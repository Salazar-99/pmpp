#pragma once

void convolution_launch(const float* input, const float* weight, float* output, 
                        int batch_size, int in_channels, int out_channels,
                        int input_height, int input_width,
                        int kernel_height, int kernel_width,
                        int padding_h, int padding_w,
                        int stride_h, int stride_w);

