/**
 * v2_fused_serial.cu — Fused kernel: WMMA GEMM (TC) then bias+ReLU (CC)
 *
 * D = ReLU(A × B + bias)
 *
 * Single kernel, serial execution within each block:
 *   1. All 8 warps do tiled GEMM using WMMA (Tensor Cores)
 *   2. All 8 warps do bias + ReLU (CUDA Cores)
 *
 * Fair comparison with v3: same 8 warps, 256 threads, 4 tiles per block.
 * Only difference: v2 is serial (all warps do TC then CC),
 *                  v3 is concurrent (TC warps + CC warps pipeline).
 */
#include "common.cuh"
using namespace nvcuda;

// ─── Tile sizes (SAME as v3 for fair comparison) ─────────────────
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;

// 8 warps = 256 threads (SAME as v3)
constexpr int NUM_WARPS = 8;
constexpr int BLOCK_DIM = NUM_WARPS * 32;  // 256 threads

// 4 tiles per block (SAME as v3)
constexpr int TILES_PER_BLOCK = 4;

// For GEMM, only first 4 warps have WMMA work (64 rows / 16 = 4 warps)
// The other 4 warps help with data loading and bias+ReLU
constexpr int GEMM_WARPS = 4;

// ─── Fused GEMM + bias + ReLU kernel (serial) ───────────────────
__global__ void fused_serial_kernel(
    const half* __restrict__ A,   // [M, K]
    const half* __restrict__ B,   // [K, N]
    half* __restrict__ D,         // [M, N]
    const half* __restrict__ bias, // [N]
    int M, int N, int K,
    int tiles_m)
{
    int warp_id = threadIdx.x / 32;

    int tile_col = blockIdx.x * BLOCK_N;
    int tile_m_start = blockIdx.y * TILES_PER_BLOCK;

    // Shared memory for loading A and B tiles
    __shared__ half smem_A[BLOCK_M][WMMA_K];
    __shared__ half smem_B[WMMA_K][BLOCK_N];
    __shared__ float smem_C[BLOCK_M][BLOCK_N];

    // ─── Process TILES_PER_BLOCK tiles sequentially ─────────────
    for (int t = 0; t < TILES_PER_BLOCK; t++) {
        int tile_m_idx = tile_m_start + t;
        if (tile_m_idx >= tiles_m) break;

        int tile_row = tile_m_idx * BLOCK_M;

        // ═══ Phase 1: ALL warps do GEMM (Tensor Cores) ══════════
        // Only warps 0-3 do WMMA (each handles 16 rows of 64-row tile)
        // Warps 4-7 help with loading but don't do WMMA
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4];
        if (warp_id < GEMM_WARPS) {
            #pragma unroll
            for (int j = 0; j < 4; j++)
                wmma::fill_fragment(acc[j], 0.0f);
        }

        int warp_row = warp_id * WMMA_M;  // only meaningful for warps 0-3

        for (int k_step = 0; k_step < K; k_step += WMMA_K) {
            // All 256 threads collaboratively load A tile [64×16]
            for (int i = threadIdx.x; i < BLOCK_M * WMMA_K; i += BLOCK_DIM) {
                int r = i / WMMA_K;
                int c = i % WMMA_K;
                int global_r = tile_row + r;
                int global_c = k_step + c;
                smem_A[r][c] = (global_r < M && global_c < K)
                               ? A[global_r * K + global_c]
                               : __float2half(0.0f);
            }

            // All 256 threads collaboratively load B tile [16×64]
            for (int i = threadIdx.x; i < WMMA_K * BLOCK_N; i += BLOCK_DIM) {
                int r = i / BLOCK_N;
                int c = i % BLOCK_N;
                int global_r = k_step + r;
                int global_c = tile_col + c;
                smem_B[r][c] = (global_r < K && global_c < N)
                               ? B[global_r * N + global_c]
                               : __float2half(0.0f);
            }

            __syncthreads();

            // Only warps 0-3 do WMMA multiply-accumulate
            if (warp_id < GEMM_WARPS) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
                wmma::load_matrix_sync(frag_a, &smem_A[warp_row][0], WMMA_K);

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
                    wmma::load_matrix_sync(frag_b, &smem_B[0][j * WMMA_N], BLOCK_N);
                    wmma::mma_sync(acc[j], frag_a, frag_b, acc[j]);
                }
            }

            __syncthreads();
        }

        // Warps 0-3 store GEMM results to shared memory
        if (warp_id < GEMM_WARPS) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(
                    &smem_C[warp_row][j * WMMA_N], acc[j], BLOCK_N,
                    wmma::mem_row_major);
            }
        }

        __syncthreads();

        // ═══ Phase 2: ALL 256 threads do bias + ReLU (CUDA Cores) ═
        for (int i = threadIdx.x; i < BLOCK_M * BLOCK_N; i += BLOCK_DIM) {
            int r = i / BLOCK_N;
            int c = i % BLOCK_N;
            int global_r = tile_row + r;
            int global_c = tile_col + c;
            if (global_r < M && global_c < N) {
                float val = smem_C[r][c] + __half2float(bias[global_c]);
                val = fmaxf(val, 0.0f);
                D[global_r * N + global_c] = __float2half(val);
            }
        }

        __syncthreads();
    }
}

// ─── Main ────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("=== v2: Fused Serial (ALL warps: GEMM then bias+ReLU) ===\n");
    printf("  M=%d  N=%d  K=%d\n", M, N, K);
    printf("  Warps=%d, Threads=%d, tiles_per_block=%d\n\n",
           NUM_WARPS, BLOCK_DIM, TILES_PER_BLOCK);

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

    // Grid configuration (SAME as v3)
    int tiles_m = (M + BLOCK_M - 1) / BLOCK_M;
    int tiles_n = (N + BLOCK_N - 1) / BLOCK_N;
    int grid_y = (tiles_m + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK;

    dim3 grid(tiles_n, grid_y);
    dim3 block(BLOCK_DIM);

    printf("  Grid: (%d, %d), Block: %d threads\n", tiles_n, grid_y, BLOCK_DIM);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
        fused_serial_kernel<<<grid, block>>>(d_A, d_B, d_D, d_bias, M, N, K, tiles_m);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    float total_ms = 0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        timer.begin();
        fused_serial_kernel<<<grid, block>>>(d_A, d_B, d_D, d_bias, M, N, K, tiles_m);
        timer.end();
        total_ms += timer.elapsed_ms();
    }

    float avg_ms = total_ms / BENCH_ITERS;
    print_result("v2 (fused serial)", avg_ms, M, N, K);

    // Verify against v1 reference
    CUDA_CHECK(cudaMemcpy(h_D, d_D, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    FILE* f = fopen("ref_output.bin", "rb");
    if (f) {
        half* h_ref = (half*)malloc(M * N * sizeof(half));
        fread(h_ref, sizeof(half), M * N, f);
        fclose(f);
        int mismatches = verify_fp16(h_ref, h_D, M * N, 2.0f);
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
