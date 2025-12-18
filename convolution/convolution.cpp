#include <torch/extension.h>
#include "convolution.h"

// C++ wrapper for convolution
at::Tensor convolution(at::Tensor input, at::Tensor weight,
                       int padding_h, int padding_w,
                       int stride_h, int stride_w) {
    // Checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (batch, channels, height, width)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (out_channels, in_channels, kernel_height, kernel_width)");
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    
    int out_channels = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_channels, "weight in_channels must match input channels");
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);
    
    // Calculate output dimensions
    int output_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    convolution_launch(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_height, kernel_width,
        padding_h, padding_w,
        stride_h, stride_w
    );

    return output;
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convolution", &convolution, "Convolution (CUDA)");
}

