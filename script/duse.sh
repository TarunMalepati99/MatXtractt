#!/bin/bash

cd "$(dirname "$0")"
command="../build/TCSpMVlib_tcperftest"

# Define data directory
data_dir="../../../data/mtx/"

# Define output files
csv_file="./res/duse_fp64.csv"
error_log="./res/duse_errors_fp64.log"

# Initialize CSV file with header row
echo "Matrix Name,TC_nnz_ratio,se_spmv (ms),du_spmv (ms),Launching Blocks" > "$csv_file"

# Clear error log file if it already exists
> "$error_log"

# Loop through each subdirectory in the data directory
for dir in "$data_dir"/*/; do
  # Get the matrix name from the directory name
  matrix_name=$(basename "$dir")
  
  # Construct the full path to the .mtx file
  mtx_file="$dir/$matrix_name.mtx"
  
  # Check if the matrix file exists
  if [[ -f "$mtx_file" ]]; then
    # Run the command and capture the output
    output=$($command "$mtx_file" 2>&1)
    
    # Check if the output contains "Success!"
    if echo "$output" | grep -q "Success!"; then
      # Extract the required values from the output
      tc_nnz_ratio=$(echo "$output" | grep -Po "TC_nnz_ratio\s*=\s*\K[0-9\.]+")
      spmv_x=$(echo "$output" | grep -Po "se_spmv:\s*\K[0-9\.]+(?=\s*ms)")
      kernel_fp16=$(echo "$output" | grep -Po "du_spmv:\s*\K[0-9\.]+(?=\s*ms)")
      launching_blocks=$(echo "$output" | grep -Po "Launching tcspmv_kernel_fp64 with\s*\K[0-9]+(?=\s*blocks)")

      # Check if all required values were extracted
      if [[ -n "$tc_nnz_ratio" && -n "$spmv_x" && -n "$kernel_fp16" && -n "$launching_blocks" ]]; then
        # Write the data to the CSV file
        echo "$matrix_name,$tc_nnz_ratio,$spmv_x,$kernel_fp16,$launching_blocks" >> "$csv_file"
      else
        # Identify missing data fields
        missing_fields=""
        [[ -z "$tc_nnz_ratio" ]] && missing_fields+="TC_nnz_ratio "
        [[ -z "$spmv_x" ]] && missing_fields+="se_spmv "
        [[ -z "$kernel_fp16" ]] && missing_fields+="du_spmv "
        [[ -z "$launching_blocks" ]] && missing_fields+="Launching Blocks "

        # Log the matrix name and missing fields to the error log
        echo "Incomplete data for $matrix_name: Missing $missing_fields" >> "$error_log"
      fi
    else
      # Log the absence of "Success!" in the output to the error log
      echo "No 'Success!' detected for $matrix_name" >> "$error_log"
    fi
  else
    # Log missing matrix file to the error log
    echo "Matrix file $mtx_file not found" >> "$error_log"
  fi
done

echo "Processing completed. Results saved to $csv_file. Errors logged to $error_log."
