#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ─── Error checking ──────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,          \
                    __LINE__, status);                                          \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ─── Constants ───────────────────────────────────────────────────
constexpr int WARMUP_ITERS = 10;
constexpr int BENCH_ITERS  = 50;

// ─── Init helpers ────────────────────────────────────────────────
static void init_matrix_fp16(half* h, int n, float scale = 0.01f) {
    for (int i = 0; i < n; i++)
        h[i] = __float2half((float)(rand() % 100) * scale);
}

static void init_bias_fp16(half* h, int n, float scale = 0.1f) {
    for (int i = 0; i < n; i++)
        h[i] = __float2half((float)(rand() % 10) * scale);
}

// ─── Verification: compare two FP16 arrays ──────────────────────
static int verify_fp16(const half* a, const half* b, int n,
                       float atol = 0.5f) {
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        float va = __half2float(a[i]);
        float vb = __half2float(b[i]);
        if (fabsf(va - vb) > atol) {
            if (mismatches < 5)
                printf("  mismatch [%d]: %.4f vs %.4f\n", i, va, vb);
            mismatches++;
        }
    }
    return mismatches;
}

// ─── Timing helper ───────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start, s)); }
    void end(cudaStream_t s = 0)   { CUDA_CHECK(cudaEventRecord(stop, s)); }
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ─── FLOPS calculation ───────────────────────────────────────────
// GEMM: 2*M*N*K  (FMA = 2 FLOP)
// bias+ReLU: 2*M*N (1 add + 1 compare)
static double gemm_flops(int M, int N, int K) {
    return 2.0 * M * N * K;
}

static double total_flops(int M, int N, int K) {
    return 2.0 * M * N * K + 2.0 * M * N;
}

static void print_result(const char* name, float ms, int M, int N, int K) {
    double flops = total_flops(M, N, K);
    double tflops = flops / (ms * 1e-3) / 1e12;
    printf("  %-30s %8.3f ms  %6.2f TFLOPS\n", name, ms, tflops);
}
