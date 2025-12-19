import torch


def time_cuda_kernel(fn, *args, repeats=100, warmup=10):
    """
    Time a CUDA kernel function.
    
    Args:
        fn: The function to time
        *args: Arguments to pass to the function
        repeats: Number of times to repeat the measurement (default: 100)
        warmup: Number of warmup iterations (default: 10)
    
    Returns:
        Average execution time in milliseconds
    
    Example:
        avg_ms = time_cuda_kernel(lambda: my_custom_op(x), repeats=100)
        print("avg kernel time:", avg_ms, "ms")
    """
    # warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        fn(*args)
    end.record()

    torch.cuda.synchronize()  # wait for all kernels to finish
    ms = start.elapsed_time(end)  # total ms for `repeats` calls
    return ms / repeats

