import torch


# example
# avg_ms = time_cuda_op(lambda x: my_custom_op(x), x)
# print("avg kernel time:", avg_ms, "ms")
def time_cuda_kernel(fn, *args, repeats=100, warmup=10):
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
