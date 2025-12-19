# Notes on Programming Massively Parallel Processors
[Book Link](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311)

I am reading PMPP to learn GPU programming since it is one of the cornerstones of modern AI systems.
This repo contains code examples and notes from my reading. I try to thoroughly comment the code to improve my own understanding
and hopefully help others along the way.

This repo is a `uv` workspace consisting of one package for each chapter of the book starting from Chapter 6 as well as shared utilities like `timing`.
Package directories are prefixed with the chapter in the book that they correspond to.
They includes CUDA kernels as well as C++ code to link the kernels to Python for execution.
I compare the outputs of each kernel to the PyTorch implementation to ensure correctness as well as time the results to better understand performance.
Naturally, the code requires a CUDA-compatible GPU and `nvcc` installed.

See one of my other projects, [gml](https://github.com/Salazar-99/gml), for a tool to easily provision cloud GPUs for running these examples.

## Running The Examples
Each individual module can be run as a `uv` script. The script names are specified in the `[project.scripts]` field of the `pyproject.toml` of each package. For example:
```bash
uv run matmul
```

## Code Structure
Each package has the same structure: Python code calls a CUDA kernel through a layer of C++.
During execution, the CUDA and C++ code is JIT-compiled as follows:
- The `torch.utils.cpp_extension.load` method in the `.py` file trigger the compilation of the `.cpp` code
- The `.cpp` code includes the `.cu` code via the `.h` file
- The `.cu` file is compiled via `nvcc` and linked to the `.cpp` code
- The final output is a `.so` file containing both the `.cpp` code and the `.cu` code
- The Python code can then call methods from this `.so` file exposed by the `PYBIND11` call in the `.cpp` code
