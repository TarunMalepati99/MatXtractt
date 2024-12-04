# GPU-Accelerated SpMV Library

This repository provides a **Sparse Matrix-Vector Multiplication (SpMV)** computation library optimized for **GPU architectures**, leveraging **tensor cores** and **CUDA cores** to achieve high performance through automated techniques.

## Features

- Efficient SpMV computation on GPUs.
- Optimized utilization of **tensor cores** and **CUDA cores**.
- Support for both **FP16** and **FP64** precision.
- Easy-to-use build and execution process.

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- A valid `.mtx` format sparse matrix file for testing.

### Build Instructions

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
2. Build the project:
    ```bash
    bash build.sh
3. Configure FP64 of FP16 support (optional):
- To enable FP64 precision, modify the `CMakeLists.txt` file before building:
    ```bash
    option(USE_FP64 "Enable fp64 support" ON)

- The default configuration uses FP16 precision (`OFF`).

### Running the Program
- Once built, execute the program as follows:
    ```bash
    ./build/<executable_name> <path_to_graph.mtx>
- Replace <`executable_name`> with the generated executable file name.
- Replace <`path_to_graph.mtx`> with the path to your sparse matrix file in `.mtx` format.

### Example
- For a sparse matrix file located at `/path/to/graph.mtx`:
    ```bash
    ./build/spmv /path/to/graph.mtx
