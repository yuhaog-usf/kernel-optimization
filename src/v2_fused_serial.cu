/**
 * v2_fused_serial.cu — Fused kernel: WMMA GEMM (TC) then bias+ReLU (CC)
 *
 * D = ReLU(A × B + bias)
 *
 * Single kernel, serial execution within each block:
 *   1. Tiled GEMM using WMMA (Tensor Cores)
 *   2. Apply bias + ReLU (CUDA Cores)
 *
 * Benefit over v1: no global memory round-trip for intermediate C.
 * The GEMM result stays in registers/shared memory → directly processed.
 */
#include "common.cuh"
using namespace nvcuda;

// ─── Tile sizes ──────────────────────────────────────────────────
// Each block computes a BLOCK_M × BLOCK_N output tile.
// WMMA operates on 16×16×16 fragments.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block tile: 64×64, computed by 4×4 = 16 WMMA tiles
// We use 4 warps (128 threads), each warp handles 4 WMMA tiles (a 16×64 row)
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int NUM_WARPS = 4;
constexpr int BLOCK_DIM = NUM_WARPS * 32;  // 128 threads

// ─── Fused GEMM + bias + ReLU kernel (serial) ───────────────────
__global__ void fused_serial_kernel(
    const half* __restrict__ A,   // [M, K]
    const half* __restrict__ B,   // [K, N]
    half* __restrict__ D,         // [M, N]
    const half* __restrict__ bias, // [N]
    int M, int N, int K)
{
    // Which output tile this block computes
    int tile_row = blockIdx.y * BLOCK_M;
    int tile_col = blockIdx.x * BLOCK_N;

    // Warp ID within block
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp handles a 16×64 strip of the block tile
    // warp 0: rows [0,16),  warp 1: rows [16,32), etc.
    int warp_row = warp_id * WMMA_M;

    // Shared memory for A and B tiles
    __shared__ half smem_A[BLOCK_M][WMMA_K];  // 64×16
    __shared__ half smem_B[WMMA_K][BLOCK_N];  // 16×64

    // Each warp has 4 accumulators for 4 WMMA tiles across N
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        wmma::fill_fragment(acc[i], 0.0f);

    // ─── K-loop: accumulate GEMM ────────────────────────────────
    for (int k_step = 0; k_step < K; k_step += WMMA_K) {
        // Collaborative load: all 128 threads load A tile [BLOCK_M × WMMA_K]
        // 64×16 = 1024 elements, 128 threads → 8 elements each
        for (int i = threadIdx.x; i < BLOCK_M * WMMA_K; i += BLOCK_DIM) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int global_r = tile_row + r;
            int global_c = k_step + c;
            smem_A[r][c] = (global_r < M && global_c < K)
                           ? A[global_r * K + global_c]
                           : __float2half(0.0f);
        }

        // Collaborative load: B tile [WMMA_K × BLOCK_N]
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

        // Each warp: load A fragment from its row, multiply with 4 B fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
        wmma::load_matrix_sync(frag_a, &smem_A[warp_row][0], WMMA_K);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
            wmma::load_matrix_sync(frag_b, &smem_B[0][j * WMMA_N], BLOCK_N);
            wmma::mma_sync(acc[j], frag_a, frag_b, acc[j]);
        }

        __syncthreads();
    }

    // ─── Bias + ReLU (CUDA Cores) ───────────────────────────────
    // Store accumulators to shared memory, apply bias+ReLU, write to global
    __shared__ float smem_C[BLOCK_M][BLOCK_N];

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        wmma::store_matrix_sync(
            &smem_C[warp_row][j * WMMA_N], acc[j], BLOCK_N,
            wmma::mem_row_major);
    }

    __syncthreads();

    // All threads apply bias + ReLU
    // 128 threads process 64×64 = 4096 elements → 32 elements per thread
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
}

// ─── Main ────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("=== v2: Fused Serial (WMMA GEMM then bias+ReLU) ===\n");
    printf("  M=%d  N=%d  K=%d\n\n", M, N, K);

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

    // Grid: one block per output tile
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(BLOCK_DIM);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
        fused_serial_kernel<<<grid, block>>>(d_A, d_B, d_D, d_bias, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    GpuTimer timer;
    float total_ms = 0;
    for (int i = 0; i < BENCH_ITERS; i++) {
        timer.begin();
        fused_serial_kernel<<<grid, block>>>(d_A, d_B, d_D, d_bias, M, N, K);
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
        int mismatches = verify_fp16(h_ref, h_D, M * N);
        printf("  Verification vs v1: %s (%d mismatches / %d)\n",
               mismatches == 0 ? "PASS" : "FAIL", mismatches, M * N);
        free(h_ref);
    } else {
        printf("  (Run v1 first to generate ref_output.bin for verification)\n");
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D); cudaFree(d_bias);
    free(h_A); free(h_B); free(h_bias); free(h_D);
    return 0;
}
