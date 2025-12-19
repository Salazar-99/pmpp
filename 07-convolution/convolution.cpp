#include <torch/extension.h>
#include "convolution.h"

// C++ wrapper for convolution
// Currently only supports 2D filters with a stride of 1
// so the output is the same size as the input
at::Tensor convolution(at::Tensor input, at::Tensor filter_host) {
    // Checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(filter_host.is_cuda(), "filter must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat, "input must be float32");
    TORCH_CHECK(filter_host.dtype() == torch::kFloat, "filter must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D (height, width)");
    TORCH_CHECK(filter_host.dim() == 2, "filter must be 2D (kernel_height, kernel_width)");
    
    // Create output tensor
    auto output = torch::empty({input.size(0), input.size(1)}, input.options());

    // Write filter to constant memory on CUDA device
    // F defined in concolution.cu
    cudaMemcpyToSymbol(filter_host, F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float))

    convolution_launch(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        2,
        input.size(0),
        input.size(1)
    );

    return output;
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convolution", &convolution, "Convolution (CUDA)");
}

