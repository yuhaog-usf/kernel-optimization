/**
 * v1_separate.cu — Baseline: cuBLAS GEMM (TC) + separate bias+ReLU kernel (CC)
 *
 * D = ReLU(A × B + bias)
 *
 * Two kernels, two global memory round-trips:
 *   1. cuBLAS GEMM:   C = A × B          (Tensor Cores)
 *   2. bias+ReLU:     D = ReLU(C + bias)  (CUDA Cores)
 *
 * The intermediate matrix C must be written to and read from global memory
 * between the two kernels — this is the overhead that fusing eliminates.
 */
#include "common.cuh"

// ─── Kernel: bias + ReLU (runs on CUDA Cores) ───────────────────
__global__ void bias_relu_kernel(half* C, const half* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int col = idx % N;
    float val = __half2float(C[idx]) + __half2float(bias[col]);
    val = fmaxf(val, 0.0f);          // ReLU
    C[idx] = __float2half(val);
}

// ─── Main ────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("=== v1: Separate Kernels (cuBLAS GEMM + bias_relu) ===\n");
    printf("  M=%d  N=%d  K=%d\n\n", M, N, K);

    // Allocate host
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    half *h_bias = (half*)malloc(N * sizeof(half));
    half *h_C = (half*)malloc(M * N * sizeof(half));

    srand(42);
    init_matrix_fp16(h_A, M * K);
    init_matrix_fp16(h_B, K * N);
    init_bias_fp16(h_bias, N);

    // Allocate device
    half *d_A, *d_B, *d_C, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, N * sizeof(half), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // GEMM: C = A × B  (column-major, so we compute B^T × A^T)
    half alpha_h = __float2half(1.0f);
    half beta_h  = __float2half(0.0f);

    // bias+ReLU launch config
    int threads = 256;
    int blocks_br = (M * N + threads - 1) / threads;

    // ─── Warmup ──────────────────────────────────────────────────
    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha_h,
            d_B, CUDA_R_16F, N,
            d_A, CUDA_R_16F, K,
            &beta_h,
            d_C, CUDA_R_16F, N,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        bias_relu_kernel<<<blocks_br, threads>>>(d_C, d_bias, M, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ─── Benchmark ───────────────────────────────────────────────
    GpuTimer timer;
    float total_ms = 0;

    for (int i = 0; i < BENCH_ITERS; i++) {
        timer.begin();

        // Kernel 1: GEMM on Tensor Cores (via cuBLAS)
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha_h,
            d_B, CUDA_R_16F, N,
            d_A, CUDA_R_16F, K,
            &beta_h,
            d_C, CUDA_R_16F, N,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // Kernel 2: bias + ReLU on CUDA Cores
        bias_relu_kernel<<<blocks_br, threads>>>(d_C, d_bias, M, N);

        timer.end();
        total_ms += timer.elapsed_ms();
    }

    float avg_ms = total_ms / BENCH_ITERS;
    print_result("v1 (separate)", avg_ms, M, N, K);

    // Save result for verification
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    // Write reference output for cross-version verification
    FILE* f = fopen("ref_output.bin", "wb");
    if (f) {
        fwrite(h_C, sizeof(half), M * N, f);
        fclose(f);
        printf("  Reference output saved to ref_output.bin\n");
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_bias);
    free(h_A); free(h_B); free(h_bias); free(h_C);
    return 0;
}
