# Matmul

This code contains a matrix-matrix multiplication kernel and the requisite code for calling it from PyTorch.
It uses a JIT-compile process via the `torch.utils.cpp_extension.load` method.
A more production-ready approach would be to setup a build for the CUDA and C++ code in a Python wheel and "publish" it so that it's statically linked
to the Python code during import but this is easier and faster. 

The code can be run with `uv run main.py`.