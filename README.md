# Notes on Programming Massively Parallel Processors
[Book Link](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311)

I am reading PMPP to learn GPU programming since it is one of the cornerstones of modern AI systems.
This repo contains code examples and notes from my reading. I try to thoroughly comment the code to improve my own understanding
and hopefully help others along the way.

Each directory in this repo is a self-contained Python project that can be run with `uv run main.py`.
It includes CUDA kernels as well as C++ code to link the kernels to Python for execution.
I compare the outputs of each kernel to the PyTorch implementation to ensure correctness.
Naturally, the code requires a CUDA-compatible GPU and `nvcc` installed.

See one of my other projects, [gml](https://github.com/Salazar-99/gml), for a tool to easily provision cloud GPUs for running these examples.
