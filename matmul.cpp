#include <torch/extension.h>
#include "matmul.h"

// C++ wrapper
at::Tensor matmul(at::Tensor a, at::Tensor b) {
    // Checks
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat, "b must be float32");
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "a and b shapes must match for multiplication (M,K) @ (K,N)");

    int M = a.size(0);
    int K = a.size(1); // shared dim
    int N = b.size(1);

    auto out = torch::empty({M, N}, a.options());

    matmul_launch(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N
    );

    return out;
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Matrix Multiplication (CUDA)");
}
