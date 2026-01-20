# MatXtract: Sparsity-Aware Matrix Transformation via Cascaded Compute Density Extraction for SpMV

This repository provides a **Sparse Matrix-Vector Multiplication (SpMV)** computation library optimized for **GPU architectures**, leveraging **tensor cores** and **CUDA cores** to achieve high performance through automated techniques.

## Features

- Efficient SpMV computation on GPUs.
- Optimized utilization of **tensor cores** and **CUDA cores**.
- Support for both **FP16** and **FP64** precision.
- Easy-to-use build and execution process.

## Getting Started

### Hardware Requirements

- **CPU**: AMD EPYC 7V13 64-Core Processor
- **GPU**: NVIDIA A100 80GB PCIe (GPU driver version 570.124.06 or later)
- **Disk Space**: At least 400GB (required to store the sparse matrix dataset)

### Software Requirements

- **CUDA**: CUDA-12.8 (tested). Lower versions (down to CUDA 11.0) are supported but may negatively affect performance.
- **Compiler**: GCC-11.4.0 or newer (tested)

### Build Instructions

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
2. Build the project:
    ```bash
    mkdir -p build
    cd build
    cmake ..
    make -j2
    ```
3. Configure FP64 of FP16 support (optional):
- To enable FP64 precision, modify the `CMakeLists.txt` file before building:
    ```bash
    option(USE_FP64 "Enable fp64 support" ON)

- The default configuration uses FP16 precision (`OFF`).

### Dataset Preparation

To prepare the testing dataset, execute the following script in the `data` directory under the project root:

- **Full Dataset**: To prepare all matrices (~400GB):
```bash
bash prepare_all_dataset.sh
```

- **Sample Dataset**: To quickly test the execution on a few representative matrices:
```bash
bash prepare_sample_dataset.sh
```

The dataset will be generated in:`data/mtx`.

## Running the Program
After compilation, the executable files are located in the `build/` directory.

### MatXtract Performance Test

Run MatXtract with specific crux parameters `(global_col, local_row)`:

```bash
./matxtract_perftest (global_col) (local_row) <path_to_matrixA.mtx>
```

**Note**: `global_col` refers to the column-wise threshold $\tau_c$, and `local_row` refers to the row-wise threshold $\tau_r$ as described in the paper.


To use default crux parameters `(global_col = 0, local_row = 0)`:

```bash
./matxtract_perftest <path_to_matrixA.mtx>
```

### Bayesian Optimization

To identify approximately optimal crux parameters, use Bayesian optimization:

```bash
cd ML
bash ml_install.sh
source ml_venv/bin/activate
(ml_venv) python bayes_opt.py <path_to_matrixA.mtx>
```

For batch processing multiple matrices:

Set the matrix directory in `batch_bayes_opt.py`:

```python
MATRIX_ROOT_DIR = "path_to_mtx_dir"
```

Then run:

```bash
(ml_venv) python batch_bayes_opt.py
```

**Example Console Output (FP64):**
```bash
[INFO] Testing matrix: ../data/mtx/cnr-2000/cnr-2000.mtx
Init0 (0,0) Time = 0.047995 ms
Init1 (1,1) Time = 0.080856 ms
===========================================
        Bayesian Optimization Result       
===========================================
Best col_frac  = 0.4592
Best hot_frac  = 0.3337
Min Time (ms)  = 0.0393
```


### Baseline Comparisons

- **cuSPARSE**: To measure cuSPARSE's SpMV performance:

```bash
./cuda_perftest <path_to_matrixA.mtx>
```

- **CSR5 and Merge-SpMV**: Both are integrated into `/baselines`.



