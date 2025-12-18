import torch
from torch.utils.cpp_extension import load

# Load the extension
# This compiles the C++/CUDA code JIT
cuda_ext = load(
    name="convolution",
    sources=["convolution.cpp", "convolution.cu"],
    verbose=True,
)


def main():
    # Use small sizes for testing
    batch_size = 1
    in_channels = 3
    out_channels = 16
    input_height = 32
    input_width = 32
    kernel_height = 3
    kernel_width = 3
    padding_h = 1
    padding_w = 1
    stride_h = 1
    stride_w = 1

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Create input tensor (batch, channels, height, width)
    input_tensor = torch.randn(
        batch_size,
        in_channels,
        input_height,
        input_width,
        device="cuda",
        dtype=torch.float32,
    )

    # Create weight tensor (out_channels, in_channels, kernel_height, kernel_width)
    weight_tensor = torch.randn(
        out_channels,
        in_channels,
        kernel_height,
        kernel_width,
        device="cuda",
        dtype=torch.float32,
    )

    # Run custom convolution kernel
    print("Testing convolution_kernel...")
    output = cuda_ext.convolution(
        input_tensor, weight_tensor, padding_h, padding_w, stride_h, stride_w
    )
    print(f"Custom kernel result shape: {output.shape}")

    # Run PyTorch native convolution for comparison
    output_ref = torch.nn.functional.conv2d(
        input_tensor,
        weight_tensor,
        padding=(padding_h, padding_w),
        stride=(stride_h, stride_w),
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
