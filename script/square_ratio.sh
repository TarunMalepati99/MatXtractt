#!/bin/bash

cd "$(dirname "$0")"
command="../build/TCSpMVlib_tcperftest"

# Define data directory
data_dir="../../../data/large_mtx/"

# Define output files
csv_file="./res/square_ratio_large.csv"
error_log="./res/111.log"

# Initialize CSV file with header row
echo "Matrix Name,square_ratio" > "$csv_file"

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

    # Extract square_ratio from the output
    tc_nnz_ratio=$(echo "$output" | grep -oP "square_ratio = \K[0-9.]+")

    if [[ -n "$tc_nnz_ratio" ]]; then
      # Write the matrix name and square_ratio to the CSV file
      echo "$matrix_name,$tc_nnz_ratio" >> "$csv_file"
    else
      # Log missing square_ratio to the error log
      echo "square_ratio missing for $matrix_name" >> "$error_log"
      echo "Output for debugging:" >> "$error_log"
      echo "$output" >> "$error_log"
    fi
  else
    # Log missing matrix file to the error log
    echo "Matrix file $mtx_file not found" >> "$error_log"
  fi
done

echo "Processing completed. Results saved to $csv_file. Errors logged to $error_log."
