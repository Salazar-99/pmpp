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

// C++ wrapper for tiled matmul
at::Tensor tiled_matmul(at::Tensor a, at::Tensor b, int tile_width) {
    // Checks
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat, "b must be float32");
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "a and b shapes must match for multiplication (M,K) @ (K,N)");
    
    // Tiled kernel assumes square matrices
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    TORCH_CHECK(M == K && K == N, "tiled_matmul requires square matrices (M == K == N)");
    TORCH_CHECK(M % tile_width == 0, "matrix width must be divisible by tile_width");

    auto out = torch::empty({M, N}, a.options());

    tiled_matmul_launch(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N,
        tile_width
    );

    return out;
}

// C++ wrapper for coarsened tiled matmul
at::Tensor coarsened_tiled_matmul(at::Tensor a, at::Tensor b, int tile_width, int coarse_factor) {
    // Checks
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat, "b must be float32");
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "a and b shapes must match for multiplication (M,K) @ (K,N)");
    
    // Coarsened tiled kernel assumes square matrices
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    TORCH_CHECK(M == K && K == N, "coarsened_tiled_matmul requires square matrices (M == K == N)");
    TORCH_CHECK(M % tile_width == 0, "matrix width must be divisible by tile_width");
    TORCH_CHECK((M / tile_width) % coarse_factor == 0, "width/tile_width must be divisible by coarse_factor");

    auto out = torch::empty({M, N}, a.options());

    coarsened_tiled_matmul_launch(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M, K, N,
        tile_width,
        coarse_factor
    );

    return out;
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Matrix Multiplication (CUDA)");
    m.def("tiled_matmul", &tiled_matmul, "Tiled Matrix Multiplication (CUDA)");
    m.def("coarsened_tiled_matmul", &coarsened_tiled_matmul, "Coarsened Tiled Matrix Multiplication (CUDA)");
}
