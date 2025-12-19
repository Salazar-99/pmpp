#include <cuda_runtime.h>
#include <cstdio>
#include "convolution.h"

// Declare constant memory for the convolutional filter
// using a constant FILTER_RADIUS.
// TODO: Figure out how we can make this dynamic
#define FILTER_RADIUS 2
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1]


// This is a naive CUDA kernel for 2D convolution. the only optimization is the use of constant memory for the filter F.
__global__ void naive_2D_convolution_kernel(float* input, float* output, int radius, int height, int width) {
    int outCol = blockIdx.x*blockDim.x+threadIdx.x;
    int outRow = blockIdx.y*blockDim.y+threadIdx.y;

    float outputValue = 0;
    // Row loop
    for (int filterRow=0; filterRow < 2*radius+1; filterRow++) {
        // Column loop
        for (int filterCol=0; filterCol < 2*radius+1, filterCol++) {
            // One output cell depends on all the cells of the input that overlap with the filter
            // This indexing starts processing the filter from the top left and works down the rows
            inputRow = outRow - radius + filterRow;
            inputCol = outCol - radius + filterCol;
            if ((inputRow < height) && (inputCol < width) && (inputRow > 0) && (inputCol > 0)) {
                outputValue += F[filterRow][filterCol] * input[inputRow*width+inputCol];
            }
        }
    }

    output[width*outRow+outCol] = outputValue;
}

// The naive kernel performs global memory accesses everytime it needs to access an element of the input matrix.
// As we know, global memory access is slow and we can improve this bottleneck by implementing tiling.
// Tiling means loading the input elements required by a given block into shared memory.
// In the case of convolution, the input is usually larger than the output given the radius of the filters.
// To handle this, we can either create thread blocks the size of the input or the size of the output.
// In this kernel we use thread blocks the size of the input and filter threads down while computing the output.
__global__ void tiled_2d_convolution_kernel(float* input, float* output, int radius, int height, int width, int input_tile_width, int output_tile_width) {
    // Each block is now processing a tile so we select the row and col for this thread based on the input tile dimension
    // and subtract the radius to start on the upper left edge of each tile.
    int outputRow = blockIdx.x*output_tile_width+threadIdx.x-radius;
    int outputCol = blockIdx.y*output_tile_width+threadIdx.y-radius;

    // Load the input tile into shared memory and set the "halo cells" to 0
    __shared__ float input_shared[input_tile_width][input_tile_width];
    if ((outputRow < width) && (outputCol < height)) {
        input_shared[outputRow][outputCol] = input[outputRow*width+outputCol];
    } else {
        input_shared[outputRow][outputCol] = 0;
    }
    __syncthreads()

    // TODO: Review this, I think some parts of it are wrong.
    // How are the tile dimensions related to the output dimensions?
    // Compute output by scanning over the input tile starting from the top left
    int tileCol = threadIdx.x - radius;
    int tileRow = threadIdx.y - radius;
    if (outputCol >= 0 && outputCol < output_tile_width && outputRow >= 0 && outputRow < output_tile_width) {
        if (tileCol >= 0 && tileCol < output_tile_width && tileRow >= 0 && tileRow < output_tile_width) {
            float output = 0;
            for (int filterRow=0; filterRow < 2*radius+1; filterRow++) {
                for (int filterCol=0; filterCol < 2*radius+1, filterCol++) {
                    inputRow = outRow - radius + filterRow;
                    inputCol = outCol - radius + filterCol;
                    if ((inputRow < height) && (inputCol < width) && (inputRow > 0) && (inputCol > 0)) {
                        outputValue += F[filterRow][filterCol] * input_shared[tileRow+filterRow][tileCol+filterCol];
                    }
                }
            }
        }
    }
    
    output[outputRow*width+outputCol] = output;
}

// C++ function to call the CUDA kernel (compiled by nvcc)
void naive_2D_convolution_launch(const float* input, float* output, int radius, int height, int width) {
    // Empty stub - implement kernel launch configuration here
}

