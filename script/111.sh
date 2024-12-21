#!/bin/bash

cd "$(dirname "$0")"
command="../build/TCSpMVlib_tcperftest"

# Define data directory
data_dir="../../../data/large_mtx/"

# Define output files
csv_file="./res/11111111.csv"
error_log="./res/11111111.log"

# Initialize CSV file with header row
echo "Matrix Name,Parameters (colProp, rowProp),TC_nnz_ratio,se_spmv (ms),du_spmv (ms),Launching Blocks" > "$csv_file"

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
      # Split the output by sections for each rowProp and colProp
      echo "$output" | awk -v matrix_name="$matrix_name" '
        /rowProp:/ { rowProp=$2 }
        /colProp:/ { colProp=$2 }
        /cols_ratio =/ { colRatio=$3 }
        /TC_nnz_ratio =/ { tc_nnz_ratio=$3 }
        /se_spmv:/ { se_spmv=$2 }
        /du_spmv:/ { du_spmv=$2 }
        /Launching tcspmv_kernel_fp64 with/ { launch_blocks=$5 }
        /Success!/ {
          if (rowProp != "" && colProp != "" && tc_nnz_ratio != "" && se_spmv != "" && du_spmv != "" && launch_blocks != "") {
            printf "%s,(%s, %s),%s,%s,%s,%s\n", matrix_name, colProp, rowProp, tc_nnz_ratio, se_spmv, du_spmv, launch_blocks
            rowProp=""; colProp=""; colRatio=""; tc_nnz_ratio=""; se_spmv=""; du_spmv=""; launch_blocks=""
          } else {
            print "Incomplete data for " matrix_name ": Missing fields" >> "'"$error_log"'"
          }
        }' >> "$csv_file"
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
