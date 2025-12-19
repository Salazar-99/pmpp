from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# Get the directory containing this file for absolute paths to sources
_THIS_DIR = Path(__file__).parent.resolve()

# Load the extension
# This compiles the C++/CUDA code JIT
cuda_ext = load(
    name="convolution",
    sources=[str(_THIS_DIR / "convolution.cpp"), str(_THIS_DIR / "convolution.cu")],
    verbose=True,
)


def main():
    # Use small sizes for testing
    input_height = 32
    input_width = 32

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Create input tensor (batch, channels, height, width)
    input_tensor = torch.randn(
        input_height,
        input_width,
        device="cuda",
        dtype=torch.float32,
    )

    # Create filter tensor (out_channels, in_channels, kernel_height, kernel_width)
    filter_tensor = torch.randn(
        device="cuda",
        dtype=torch.float32,
    )

    # Run custom convolution kernel
    print("Testing convolution_kernel...")
    output = cuda_ext.convolution(input_tensor, filter_tensor)
    print(f"Custom kernel result shape: {output.shape}")

    # Run PyTorch native convolution for comparison
    output_ref = torch.nn.functional.conv2d(
        input_tensor,
        filter_tensor,
        padding=0,
        stride=1,
    )
    print(f"PyTorch result shape: {output_ref.shape}")

    # Check custom kernel result
    try:
        is_close = torch.allclose(output, output_ref, atol=1e-3)
        print(f"Custom kernel result ok: {is_close}")
        if not is_close:
            max_diff = (output - output_ref).abs().max().item()
            print(f"Max difference: {max_diff}")
    except Exception as e:
        print(f"Custom kernel check failed: {e}")


if __name__ == "__main__":
    main()
