#include <cuda_runtime.h>
#include <cstdio>
#include "matmul.h"

// (M x K) @ (K x N)
// CUDA Kernel
__global__ void matmul_kernel(const float* a, const float* b, float* out, int M, int K, int N) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    
    if ((row < M) && (col < N)) {
        float outvalue = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A is (M x K): row*K + k
            // B is (K x N): k*N + col
            outvalue += a[row*K + k] * b[k*N + col];
        }
        // Out is (M x N): row*N + col
        out[row*N + col] = outvalue;
    }
}

// C++ function to call the CUDA kernel (compiled by nvcc)
void matmul_launch(const float* a, const float* b, float* out, int M, int K, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    matmul_kernel<<<blocks, threads>>>(a, b, out, M, K, N);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
