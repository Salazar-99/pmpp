#pragma once

void matmul_launch(const float* a, const float* b, float* out, int M, int K, int N);
void tiled_matmul_launch(const float* a, const float* b, float* out, int M, int K, int N, int tile_width);
void coarsened_tiled_matmul_launch(const float* a, const float* b, float* out, int M, int K, int N, int tile_width, int coarse_factor);
