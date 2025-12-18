import torch
from torch.utils.cpp_extension import load

# Load the extension
# This compiles the C++/CUDA code JIT
cuda_ext = load(
    name="matmul",
    sources=["matmul.cpp", "matmul.cu"],
    verbose=True,
)


def main():
    # Use a small size for testing
    M, K, N = 128, 128, 128

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    # Run naive custom kernel
    print("Testing naive_matmul_kernel...")
    c = cuda_ext.matmul(a, b)
    print(f"Naive kernel result shape: {c.shape}")

    # Run PyTorch native
    c_ref = a @ b
    print(f"PyTorch result shape: {c_ref.shape}")

    # Check naive kernel result
    try:
        is_close = torch.allclose(c, c_ref, atol=1e-3)
        print(f"Naive kernel result ok: {is_close}")
        if not is_close:
            max_diff = (c - c_ref).abs().max().item()
            print(f"Max difference: {max_diff}")
    except Exception as e:
        print(f"Naive kernel check failed: {e}")

    # Run tiled custom kernel
    print("\nTesting tiled_matmul_kernel...")
    tile_width = 16
    c_tiled = cuda_ext.tiled_matmul(a, b, tile_width)
    print(f"Tiled kernel result shape: {c_tiled.shape}")

    # Check tiled kernel result
    try:
        is_close_tiled = torch.allclose(c_tiled, c_ref, atol=1e-3)
        print(f"Tiled kernel result ok: {is_close_tiled}")
        if not is_close_tiled:
            max_diff_tiled = (c_tiled - c_ref).abs().max().item()
            print(f"Max difference: {max_diff_tiled}")
    except Exception as e:
        print(f"Tiled kernel check failed: {e}")

    # Run coarsened tiled custom kernel
    print("\nTesting coarsened_tiled_matmul_kernel...")
    tile_width = 16
    coarse_factor = 2  # Each thread computes 2 output elements
    c_coarsened = cuda_ext.coarsened_tiled_matmul(a, b, tile_width, coarse_factor)
    print(f"Coarsened tiled kernel result shape: {c_coarsened.shape}")

    # Check coarsened tiled kernel result
    try:
        is_close_coarsened = torch.allclose(c_coarsened, c_ref, atol=1e-3)
        print(f"Coarsened tiled kernel result ok: {is_close_coarsened}")
        if not is_close_coarsened:
            max_diff_coarsened = (c_coarsened - c_ref).abs().max().item()
            print(f"Max difference: {max_diff_coarsened}")
    except Exception as e:
        print(f"Coarsened tiled kernel check failed: {e}")


if __name__ == "__main__":
    main()
