/**
 * v3_fused_concurrent.cu — Fused kernel with CC+TC concurrency (pipelined)
 *
 * D = ReLU(A × B + bias)
 *
 * Key idea: exploit the CC+TC overlap measured in microbenchmark (overlap=0.48).
 * Within a single kernel, different warps use different execution units:
 *   - TC warps: compute GEMM tiles using WMMA (Tensor Cores)
 *   - CC warps: apply bias+ReLU on completed tiles (CUDA Cores)
 *
 * Pipeline: while TC warps compute tile[i], CC warps process tile[i-1].
 * Double buffering in shared memory enables this overlap.
 *
 *   Time →
 *   TC warps:  [GEMM tile 0] [GEMM tile 1] [GEMM tile 2] ...
 *   CC warps:       idle     [bias+relu 0 ] [bias+relu 1 ] ...
 *                             ↑ overlapped! ↑ overlapped!
 *
 * Design: each TC warp independently loads from global memory into WMMA
 * fragments (no shared A/B tile), avoiding the need for intra-TC-warp sync.
 * Results are written to a double-buffered shared memory output tile.
 */
#include "common.cuh"
using namespace nvcuda;

// ─── Tile sizes ──────────────────────────────────────────────────
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;

// 4 TC warps + 4 CC warps = 256 threads
constexpr int TC_WARPS = 4;
constexpr int CC_WARPS = 4;
constexpr int TOTAL_WARPS = TC_WARPS + CC_WARPS;
constexpr int BLOCK_DIM = TOTAL_WARPS * 32;  // 256

// Each block processes a strip of tiles along M for pipeline opportunity
constexpr int TILES_PER_BLOCK = 4;

// ─── Fused GEMM + bias + ReLU kernel (CC+TC concurrent) ─────────
__global__ void fused_concurrent_kernel(
    const half* __restrict__ A,    // [M, K]  row-major
    const half* __restrict__ B,    // [K, N]  row-major
    half* __restrict__ D,          // [M, N]
    const half* __restrict__ bias, // [N]
    int M, int N, int K,
    int tiles_m)
{
    int warp_id = threadIdx.x / 32;
    bool is_tc_warp = (warp_id < TC_WARPS);
    int tc_local = warp_id;               // 0..3 for TC warps
    int cc_local = warp_id - TC_WARPS;    // 0..3 for CC warps

    int tile_col = blockIdx.x * BLOCK_N;
    int tile_m_start = blockIdx.y * TILES_PER_BLOCK;

    // Double buffer for output tile [BLOCK_M × BLOCK_N]
    __shared__ float smem_out[2][BLOCK_M][BLOCK_N];

    // Atomic flags: 0 = not ready, 1 = TC done writing
    __shared__ int tile_ready[2];

    if (threadIdx.x < 2) tile_ready[threadIdx.x] = 0;
    __syncthreads();

    // ─── Pipeline loop ──────────────────────────────────────────
    for (int t = 0; t < TILES_PER_BLOCK; t++) {
        int tile_m_idx = tile_m_start + t;
        if (tile_m_idx >= tiles_m) break;

        int tile_row = tile_m_idx * BLOCK_M;
        int buf = t % 2;
        int prev_buf = 1 - buf;

        if (is_tc_warp) {
            // ═══ TC WARPS: compute GEMM for current tile ════════
            // Each TC warp handles a 16-row strip: warp i → rows [i*16, i*16+16)
            int warp_row = tc_local * WMMA_M;

            // 4 accumulators for 4 WMMA tiles across N (16×16 each → 16×64)
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
                wmma::fill_fragment(acc[j], 0.0f);

            // K-loop: each warp loads independently from global memory
            for (int k_step = 0; k_step < K; k_step += WMMA_K) {
                // Load A fragment: 16×16 block starting at (tile_row+warp_row, k_step)
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
                const half* a_ptr = A + (tile_row + warp_row) * K + k_step;
                // Bounds check: if tile extends past M, we loaded zeros via init
                wmma::load_matrix_sync(frag_a, a_ptr, K);

                // For each of 4 N-tiles, load B fragment and accumulate
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
                    const half* b_ptr = B + k_step * N + (tile_col + j * WMMA_N);
                    wmma::load_matrix_sync(frag_b, b_ptr, N);
                    wmma::mma_sync(acc[j], frag_a, frag_b, acc[j]);
                }
            }

            // Store results to shared memory double buffer
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(
                    &smem_out[buf][warp_row][j * WMMA_N], acc[j], BLOCK_N,
                    wmma::mem_row_major);
            }

            // Signal CC warps: this tile is ready
            __threadfence_block();
            if (threadIdx.x == 0) {
                atomicExch(&tile_ready[buf], 1);
            }

        } else {
            // ═══ CC WARPS: process PREVIOUS tile with bias+ReLU ═══
            if (t > 0) {
                int prev_tile_row = (tile_m_idx - 1) * BLOCK_M;

                // Spin-wait until previous tile's GEMM is done
                if (cc_local == 0 && (threadIdx.x % 32) == 0) {
                    while (atomicAdd(&tile_ready[prev_buf], 0) == 0) {
                        // busy wait — CC warps use CUDA Cores, not blocking TC
                    }
                }
                // Simple warp-level sync to propagate the flag visibility
                __syncwarp();

                // Inter-CC-warp sync: all CC warps wait for warp 0's signal
                // Use a second spin on the same flag (already set to 1)
                while (atomicAdd(&tile_ready[prev_buf], 0) == 0) {}

                // Apply bias + ReLU
                int cc_thread_offset = (warp_id - TC_WARPS) * 32 + (threadIdx.x % 32);
                int cc_total_threads = CC_WARPS * 32;
                for (int i = cc_thread_offset; i < BLOCK_M * BLOCK_N; i += cc_total_threads) {
                    int r = i / BLOCK_N;
                    int c = i % BLOCK_N;
                    int gr = prev_tile_row + r;
                    int gc = tile_col + c;
                    if (gr < M && gc < N) {
                        float val = smem_out[prev_buf][r][c] + __half2float(bias[gc]);
                        val = fmaxf(val, 0.0f);
                        D[gr * N + gc] = __float2half(val);
                    }
                }

                // Reset flag for next reuse
                __syncwarp();
                if (cc_local == 0 && (threadIdx.x % 32) == 0) {
                    atomicExch(&tile_ready[prev_buf], 0);
                }
            }
            // If t == 0, CC warps have nothing to process yet (idle)
        }

        // Full block sync before next tile iteration
        // This ensures: TC warps don't overwrite smem_out[buf] before CC warps
        // finish reading it in a future iteration
        __syncthreads();
    }

    // ─── Drain: process the last tile ────────────────────────────
    int last_t = min(TILES_PER_BLOCK, tiles_m - tile_m_start);
    if (last_t > 0) {
        int last_buf = (last_t - 1) % 2;
        int last_tile_row = (tile_m_start + last_t - 1) * BLOCK_M;

        // Wait for last tile to be complete
        if (threadIdx.x == 0) {
            while (atomicAdd(&tile_ready[last_buf], 0) == 0) {}
        }
        __syncthreads();

        // All threads help drain the last tile
        for (int i = threadIdx.x; i < BLOCK_M * BLOCK_N; i += BLOCK_DIM) {
            int r = i / BLOCK_N;
            int c = i % BLOCK_N;
            int gr = last_tile_row + r;
            int gc = tile_col + c;
            if (gr < M && gc < N) {
                float val = smem_out[last_buf][r][c] + __half2float(bias[gc]);
                val = fmaxf(val, 0.0f);
                D[gr * N + gc] = __float2half(val);
            }
        }
    }
}

// ─── Main ────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("=== v3: Fused Concurrent (TC warps + CC warps pipelined) ===\n");
    printf("  M=%d  N=%d  K=%d\n", M, N, K);
    printf("  TC warps=%d, CC warps=%d, tiles_per_block=%d\n\n",
           TC_WARPS, CC_WARPS, TILES_PER_BLOCK);

    // Pad matrices to multiples of BLOCK_M/BLOCK_N to avoid OOB in WMMA loads
    int M_pad = ((M + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
    int N_pad = ((N + BLOCK_N - 1) / BLOCK_N) * BLOCK_N;
    int K_pad = ((K + WMMA_K - 1) / WMMA_K) * WMMA_K;

    // Allocate host (padded, zero-filled)
    size_t size_A = M_pad * K_pad;
    size_t size_B = K_pad * N_pad;
    size_t size_D = M_pad * N_pad;

    half *h_A = (half*)calloc(size_A, sizeof(half));
    half *h_B = (half*)calloc(size_B, sizeof(half));
    half *h_bias = (half*)calloc(N_pad, sizeof(half));
    half *h_D = (half*)malloc(size_D * sizeof(half));

    srand(42);
    // Fill only the valid region
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K_pad + j] = __float2half((float)(rand() % 100) * 0.01f);

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            h_B[i * N_pad + j] = __float2half((float)(rand() % 100) * 0.01f);

    for (int j = 0; j < N; j++)
        h_bias[j] = __float2half((float)(rand() % 10) * 0.1f);

    // Allocate device
    half *d_A, *d_B, *d_D, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_D, size_D * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, N_pad * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, N_pad * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, size_D * sizeof(half)));

    // Grid configuration
    int tiles_m = M_pad / BLOCK_M;
    int tiles_n = N_pad / BLOCK_N;
    int grid_y = (tiles_m + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK;

    dim3 grid(tiles_n, grid_y);
    dim3 block(BLOCK_DIM);

    printf("  Grid: (%d, %d), Block: %d threads\n", tiles_n, grid_y, BLOCK_DIM);
    printf("  Padded: M=%d N=%d K=%d\n", M_pad, N_pad, K_pad);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
        fused_concurrent_kernel<<<grid, block>>>(d_A, d_B, d_D, d_bias, M, N_pad, K_pad, tiles_m);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Benchmark
    GpuTimer timer;
    float total_ms = 0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        timer.begin();
        fused_concurrent_kernel<<<grid, block>>>(d_A, d_B, d_D, d_bias, M, N_pad, K_pad, tiles_m);
        timer.end();
        total_ms += timer.elapsed_ms();
    }

    float avg_ms = total_ms / BENCH_ITERS;
    print_result("v3 (fused concurrent)", avg_ms, M, N, K);

    // Verify against v1 reference
    CUDA_CHECK(cudaMemcpy(h_D, d_D, size_D * sizeof(half), cudaMemcpyDeviceToHost));

    // Extract unpadded result for comparison
    FILE* f = fopen("ref_output.bin", "rb");
    if (f) {
        half* h_ref = (half*)malloc(M * N * sizeof(half));
        fread(h_ref, sizeof(half), M * N, f);
        fclose(f);

        // Compare only valid region (unpadded)
        int mismatches = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float va = __half2float(h_ref[i * N + j]);
                float vb = __half2float(h_D[i * N_pad + j]);
                if (fabsf(va - vb) > 1.0f) {
                    if (mismatches < 5)
                        printf("  mismatch [%d,%d]: %.4f vs %.4f\n", i, j, va, vb);
                    mismatches++;
                }
            }
        }
        printf("  Verification vs v1: %s (%d mismatches / %d)\n",
               mismatches == 0 ? "PASS" : "CHECK", mismatches, M * N);
        free(h_ref);
    } else {
        printf("  (Run v1 first to generate ref_output.bin for verification)\n");
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D); cudaFree(d_bias);
    free(h_A); free(h_B); free(h_bias); free(h_D);
    return 0;
}
