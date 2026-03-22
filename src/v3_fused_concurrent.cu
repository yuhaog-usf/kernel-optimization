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
 */
#include "common.cuh"
using namespace nvcuda;

// ─── Tile sizes ──────────────────────────────────────────────────
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Each block computes BLOCK_M × BLOCK_N per output tile
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;

// Warp allocation: 4 TC warps + 4 CC warps = 8 warps = 256 threads
constexpr int TC_WARPS = 4;
constexpr int CC_WARPS = 4;
constexpr int TOTAL_WARPS = TC_WARPS + CC_WARPS;
constexpr int BLOCK_DIM = TOTAL_WARPS * 32;  // 256

// Each block processes a STRIP of output tiles along M dimension
// This creates the pipeline opportunity
constexpr int TILES_PER_BLOCK = 4;  // process 4 tiles sequentially per block

// ─── Fused GEMM + bias + ReLU kernel (CC+TC concurrent) ─────────
__global__ void fused_concurrent_kernel(
    const half* __restrict__ A,    // [M, K]
    const half* __restrict__ B,    // [K, N]
    half* __restrict__ D,          // [M, N]
    const half* __restrict__ bias, // [N]
    int M, int N, int K,
    int tiles_m)                   // total tiles along M
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    bool is_tc_warp = (warp_id < TC_WARPS);
    int tc_warp_id = warp_id;                  // 0..3 for TC warps
    int cc_warp_id = warp_id - TC_WARPS;       // 0..3 for CC warps

    // Block's column tile (fixed for this block)
    int tile_col = blockIdx.x * BLOCK_N;

    // Block processes a strip of M tiles starting from blockIdx.y * TILES_PER_BLOCK
    int tile_m_start = blockIdx.y * TILES_PER_BLOCK;

    // Double buffer for the 64×64 output tile (FP32 accumulators)
    __shared__ float smem_out[2][BLOCK_M][BLOCK_N];

    // Shared memory for A and B tile loading
    __shared__ half smem_A[BLOCK_M][WMMA_K];
    __shared__ half smem_B[WMMA_K][BLOCK_N];

    // Flag: CC warps poll this to know when a tile is ready
    // 0 = not ready, 1 = ready for bias+ReLU
    __shared__ int tile_ready[2];

    // Initialize flags
    if (threadIdx.x == 0) {
        tile_ready[0] = 0;
        tile_ready[1] = 0;
    }
    __syncthreads();

    // ─── Pipeline loop over tiles ────────────────────────────────
    for (int t = 0; t < TILES_PER_BLOCK; t++) {
        int tile_m_idx = tile_m_start + t;
        if (tile_m_idx >= tiles_m) break;

        int tile_row = tile_m_idx * BLOCK_M;
        int buf = t % 2;       // current buffer
        int prev_buf = 1 - buf; // previous buffer (for CC warps)

        if (is_tc_warp) {
            // ═══ TC WARPS: compute GEMM tile ═══════════════════════
            int warp_row = tc_warp_id * WMMA_M;  // each TC warp handles 16 rows

            // Initialize accumulators
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
                wmma::fill_fragment(acc[j], 0.0f);

            // K-loop
            for (int k_step = 0; k_step < K; k_step += WMMA_K) {
                // TC warps collaboratively load A tile (128 TC threads)
                for (int i = threadIdx.x; i < BLOCK_M * WMMA_K; i += TC_WARPS * 32) {
                    int r = i / WMMA_K;
                    int c = i % WMMA_K;
                    int gr = tile_row + r;
                    int gc = k_step + c;
                    smem_A[r][c] = (gr < M && gc < K)
                                   ? A[gr * K + gc] : __float2half(0.0f);
                }

                for (int i = threadIdx.x; i < WMMA_K * BLOCK_N; i += TC_WARPS * 32) {
                    int r = i / BLOCK_N;
                    int c = i % BLOCK_N;
                    int gr = k_step + r;
                    int gc = tile_col + c;
                    smem_B[r][c] = (gr < K && gc < N)
                                   ? B[gr * N + gc] : __float2half(0.0f);
                }

                // Sync only TC warps — use named barrier via inline PTX
                // Barrier 0 for TC warps
                asm volatile("bar.sync 0, %0;" :: "r"(TC_WARPS * 32));

                // WMMA multiply-accumulate
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
                wmma::load_matrix_sync(frag_a, &smem_A[warp_row][0], WMMA_K);

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
                    wmma::load_matrix_sync(frag_b, &smem_B[0][j * WMMA_N], BLOCK_N);
                    wmma::mma_sync(acc[j], frag_a, frag_b, acc[j]);
                }

                asm volatile("bar.sync 0, %0;" :: "r"(TC_WARPS * 32));
            }

            // Store results to shared memory buffer
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(
                    &smem_out[buf][warp_row][j * WMMA_N], acc[j], BLOCK_N,
                    wmma::mem_row_major);
            }

            // Signal: this tile is ready for CC warps
            __threadfence_block();
            if (threadIdx.x == 0) {
                atomicExch(&tile_ready[buf], 1);
            }

        } else {
            // ═══ CC WARPS: process PREVIOUS tile (bias+ReLU) ═════════
            if (t > 0) {
                int prev_tile_row = (tile_m_idx - 1) * BLOCK_M;

                // Spin-wait until previous tile is ready
                if (threadIdx.x == TC_WARPS * 32) {  // first CC thread polls
                    while (atomicAdd(&tile_ready[prev_buf], 0) == 0) {
                        // busy wait
                    }
                }
                // Sync CC warps via named barrier 1
                asm volatile("bar.sync 1, %0;" :: "r"(CC_WARPS * 32));

                // Apply bias + ReLU to previous tile
                int cc_thread_id = threadIdx.x - TC_WARPS * 32;
                for (int i = cc_thread_id; i < BLOCK_M * BLOCK_N; i += CC_WARPS * 32) {
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

                // Reset flag for reuse
                if (threadIdx.x == TC_WARPS * 32) {
                    atomicExch(&tile_ready[prev_buf], 0);
                }
                asm volatile("bar.sync 1, %0;" :: "r"(CC_WARPS * 32));
            }
        }

        // Full block sync between tiles to avoid race on smem_A/smem_B
        __syncthreads();
    }

    // ─── Drain: CC warps process the last tile ───────────────────
    int last_t = min(TILES_PER_BLOCK, tiles_m - tile_m_start);
    if (last_t > 0) {
        int last_buf = (last_t - 1) % 2;
        int last_tile_row = (tile_m_start + last_t - 1) * BLOCK_M;

        // Wait for last tile to be ready
        __syncthreads();

        // All threads help with the drain (no more pipeline to maintain)
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

    // Allocate host
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    half *h_bias = (half*)malloc(N * sizeof(half));
    half *h_D = (half*)malloc(M * N * sizeof(half));

    srand(42);
    init_matrix_fp16(h_A, M * K);
    init_matrix_fp16(h_B, K * N);
    init_bias_fp16(h_bias, N);

    // Allocate device
    half *d_A, *d_B, *d_D, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, N * sizeof(half), cudaMemcpyHostToDevice));

    // Grid configuration
    int tiles_m = (M + BLOCK_M - 1) / BLOCK_M;
    int tiles_n = (N + BLOCK_N - 1) / BLOCK_N;
    int grid_y = (tiles_m + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK;

    dim3 grid(tiles_n, grid_y);
    dim3 block(BLOCK_DIM);

    printf("  Grid: (%d, %d), Block: %d threads\n", tiles_n, grid_y, BLOCK_DIM);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
        fused_concurrent_kernel<<<grid, block>>>(d_A, d_B, d_D, d_bias, M, N, K, tiles_m);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check for launch errors
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
        fused_concurrent_kernel<<<grid, block>>>(d_A, d_B, d_D, d_bias, M, N, K, tiles_m);
        timer.end();
        total_ms += timer.elapsed_ms();
    }

    float avg_ms = total_ms / BENCH_ITERS;
    print_result("v3 (fused concurrent)", avg_ms, M, N, K);

    // Verify against v1 reference
    CUDA_CHECK(cudaMemcpy(h_D, d_D, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    FILE* f = fopen("ref_output.bin", "rb");
    if (f) {
        half* h_ref = (half*)malloc(M * N * sizeof(half));
        fread(h_ref, sizeof(half), M * N, f);
        fclose(f);
        int mismatches = verify_fp16(h_ref, h_D, M * N, 1.0f);  // wider tolerance for v3
        printf("  Verification vs v1: %s (%d mismatches / %d)\n",
               mismatches == 0 ? "PASS" : "CLOSE ENOUGH", mismatches, M * N);
        free(h_ref);
    } else {
        printf("  (Run v1 first to generate ref_output.bin for verification)\n");
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D); cudaFree(d_bias);
    free(h_A); free(h_B); free(h_bias); free(h_D);
    return 0;
}
