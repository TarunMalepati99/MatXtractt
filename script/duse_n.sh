#!/bin/bash

cd "$(dirname "$0")"
command="../build/matxtract_perftest"

# Define data directory
data_dir="../../../data/mtx/"

# Define output files
csv_file="./res/results_n.csv"
error_log="./res/error_n.log"

# Initialize CSV file with header row
echo "Matrix Name,TC_nnz_ratio,SpMV_X (ms),tcspmv_kernel_fp64 (ms),Launching Blocks" > "$csv_file"

# Clear error log file if it already exists
> "$error_log"

# Flag to indicate whether to start processing
start_from_n=false

# Loop through each subdirectory in the data directory
for dir in "$data_dir"/*/; do
  # Get the matrix name from the directory name
  matrix_name=$(basename "$dir")

  # Check if the matrix name starts with 'n' or a subsequent letter
  if [[ $matrix_name =~ ^[n-z] ]]; then
    start_from_n=true
  fi

  # Skip until we reach the desired starting point
  if [[ "$start_from_n" == false ]]; then
    continue
  fi

  # Construct the full path to the .mtx file
  mtx_file="$dir/$matrix_name.mtx"
  
  # Check if the matrix file exists
  if [[ -f "$mtx_file" ]]; then
    # Run the command and capture the output
    output=$($command "$mtx_file" 2>&1)
    
    # Extract the required values from the output
    tc_nnz_ratio=$(echo "$output" | grep -Po "TC_nnz_ratio\s*=\s*\K[0-9\.]+")
    spmv_x=$(echo "$output" | grep -Po "SpMV_X:\s*\K[0-9\.]+(?=\s*ms)")
    kernel_fp64=$(echo "$output" | grep -Po "tcspmv_kernel_fp64:\s*\K[0-9\.]+(?=\s*ms)")
    launching_blocks=$(echo "$output" | grep -Po "Launching tcspmv_kernel_fp64 with\s*\K[0-9]+(?=\s*blocks)")

    # Check if all required values were extracted
    if [[ -n "$tc_nnz_ratio" && -n "$spmv_x" && -n "$kernel_fp64" && -n "$launching_blocks" ]]; then
      # Write the data to the CSV file
      echo "$matrix_name,$tc_nnz_ratio,$spmv_x,$kernel_fp64,$launching_blocks" >> "$csv_file"
    else
      # Log the matrix name to the error log
      echo "Incomplete data for $matrix_name" >> "$error_log"
    fi
  else
    # Log missing matrix file to the error log
    echo "Matrix file $mtx_file not found" >> "$error_log"
  fi
done

echo "Processing completed. Results saved to $csv_file. Errors logged to $error_log."
