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

    # Run custom kernel
    c = cuda_ext.matmul(a, b)
    print(f"Kernel result: {c}")

    # Run PyTorch native
    c_ref = a @ b
    print(f"Pytorch result: {c_ref}")

    # Check result (Note: Kernel is currently empty/TODO, so this will likely fail or be random)
    print(f"Output shape: {c.shape}")
    try:
        is_close = torch.allclose(c, c_ref, atol=1e-3)
        print(f"Result ok: {is_close}")
    except Exception as e:
        print(f"Check failed: {e}")


if __name__ == "__main__":
    main()
