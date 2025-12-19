#include <cuda_runtime.h>
#include <cstdio>
#include "matmul.h"

// Multiply two matrices M and N and write the output to P.
// M is (m x k) and N is (k x n) so the sum is over the k dimension.
// In this naive matmul we assign one thread to each output element of P.
__global__ void naive_matmul_kernel(float* M, float* N, float* P, int m, int k, int n) {
    // This thread will compute P_row,col
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    // Check for out of bounds because the matrix dimensions
    // are probably not a multiple of the block size
    if((row<m) && (col<n)) {
        float p = 0;
        for (int j = 0; j < k; ++j) {
            // Perform dot product between row from M and col from N
            p += M[row*k+j] * N[n*j+col];
        }
        // Write out the result, recall that the matrices are
        // represented as flat array is row-major layout
        P[row*n+col] = p;
    }
}


// The previous naive kernel reads every element of M and N from global memory (DRAM), this is slow
// and leads to low arithmetic intensity. To improve on this bottleneck, we implement tiling.
// Tiling is the process of collaboratively loading subsets of the input data into shared memory.
// Tiling can only be performed if the data in the tiles can be processed independantly of other tiles
// e.g. there is no cross-tile data dependency. Additionally, shared memory is accessible only at the block level, 
// so only threads in a given block can access the shared memory for that block. 
// With these constraints, we need to make sure that we are bringing data into shared memory that will be useful 
// for all of the threads in the block. In the case of matrix multiplication, we can reduce
// global memory accesses by a factor of N for tiles of size N x N. 
// For visuals, see Fig. 5.5 and Fig 5.6 in PMPP. At a high level, each thread is computing one element
// of the output by loading in input tiles from M and N into shared memory and computing the partial
// dot product in phases. Each phase uses one set of tiles at a time. Once all tiles are processed (number of tiles is given by
// width of the matrix / tile width) the accumulated result is written to the output. In this kernel we assume
// that all matrices have dimension width x width to keep the phase calculation simple.
__global__ void tiled_matmul_kernel(float* M, float* N, float* P, int width, int tile_width) {
    // Declare shared memory segments with tile_width^2 elements
    __shared__ float M_shared[tile_width][tile_width];
    __shared__ float N_shared[tile_width][tile_width];

    // Declare aliases for thread and block ids
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Get row and column of output element for this thread
    int row = by * tile_width + ty;
    int col = bx * tile_width + tx;

    // Loop over tiles
    float Pvalue = 0;
    for (int phase = 0; phase < width/tile_width; ++phase) {
        // Load the tiles into shared memory for use by all the threads
        // in this block
        M_shared[ty][tx] = M[row*width + phase*tile_width + tx];
        N_shared[ty][tx] = N[(phase*tile_width + ty)*width + col];
        // Wait for all threads to finish loading into shared memory
        // This prevents read-after-write races
        __syncthreads();

        // Compute partial dot product
        for (int k=0; k < tile_width; ++k) {
            Pvalue += M_shared[ty][k] * N_shared[k][tx];
        }

        // This prevents write-after-read races
        __syncthreads();
    }

    P[row*width+col] = Pvalue;
}

// The tiled matmul kernel has some redundant data loading. For each tile of M that is loaded we need to load multiple tiles of N
// to comput a set of tiles for the output P. It would be more efficient if we loaded those only once rather than multiple
// times across several blocks. This may or may not be a serious problem depending on
// hardware constraints and input data size. If it is a problem, we can turn to coarsening.
// Coarsening is the design change of assigning more work to a single thread, say instead of computing one output element
// computing multiple. In the case of matrix multiplication, we can compute multiple columns of an output row with a single thread
// by loading multiple tiles of N into shared memory for a given tile of M.
__global__ void coarsened_tiled_matmul_kernel(float* M, float* N, float* P, int width, int tile_width, int coarse_factor) {
    // Declare shared memory segments with tile_width^2 elements
    __shared__ float M_shared[tile_width][tile_width];
    __shared__ float N_shared[tile_width][tile_width];

    // Declare aliases for thread and block ids
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Get row and column of output element for this thread
    int row = by * tile_width + ty;
    int col_start = bx * tile_width * coarse_factor + tx;

    // Initialize array of output values for this thread
    float Pvalue[coarse_factor];
    for (int c = 0; c < coarse_factor; ++c) {
        Pvalue[c] = 0;
    }

    // Loop over tiles
    for (int phase = 0; phase < width/tile_width; ++phase) {
        // Load M tile into shared memory
        M_shared[ty][tx] = M[row*width + phase*tile_width + tx];
        // Wait for all threads to finish loading M into shared memory
        __syncthreads();

        // Coarsening loop, process multiple N tiles for one M tile
        for (int c = 0; c < coarse_factor; ++c) {
            int col = col_start + c*tile_width;

            // Load the N tile into shared memory for this iteration
            N_shared[ty][tx] = N[(phase*tile_width + ty)*width + col];
            // Wait for all threads to finish loading into shared memory
            // This prevents read-after-write races
            __syncthreads();

            // Compute partial dot product
            for (int k=0; k < tile_width; ++k) {
                Pvalue[c] += M_shared[ty][k] * N_shared[k][tx];
            }

            // This prevents write-after-read races before loading next N tile
            __syncthreads();
        }
    }

    // Write results
    for (int c = 0; c < coarse_factor; ++c) {
        int col = col_start + c*tile_width;
        P[row*width+col] = Pvalue[c];
    }  
}

// C++ function to call the CUDA kernel (compiled by nvcc)
void matmul_launch(const float* a, const float* b, float* out, int M, int K, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    naive_matmul_kernel<<<blocks, threads>>>(const_cast<float*>(a), const_cast<float*>(b), out, M, K, N);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// C++ function to call the tiled CUDA kernel (compiled by nvcc)
void tiled_matmul_launch(const float* a, const float* b, float* out, int M, int K, int N, int tile_width) {
    // For tiled kernel, we assume square matrices (M == K == N == width)
    // and that width is divisible by tile_width
    int width = M;
    dim3 threads(tile_width, tile_width);
    dim3 blocks(width / tile_width, width / tile_width);

    tiled_matmul_kernel<<<blocks, threads>>>(const_cast<float*>(a), const_cast<float*>(b), out, width, tile_width);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
}

// C++ function to call the coarsened tiled CUDA kernel (compiled by nvcc)
void coarsened_tiled_matmul_launch(const float* a, const float* b, float* out, int M, int K, int N, int tile_width, int coarse_factor) {
    // For coarsened tiled kernel, we assume square matrices (M == K == N == width)
    // and that width is divisible by tile_width
    int width = M;
    dim3 threads(tile_width, tile_width);
    dim3 blocks((width / tile_width) / coarse_factor, width / tile_width);

    coarsened_tiled_matmul_kernel<<<blocks, threads>>>(const_cast<float*>(a), const_cast<float*>(b), out, width, tile_width, coarse_factor);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
}
