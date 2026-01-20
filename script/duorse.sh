#!/bin/bash

cd "$(dirname "$0")"
command="../build/matxtract_perftest"

# Define data directory
data_dir="../../../data/large_mtx/"

# Define output files
csv_file="./res/du_large_fp16_new.csv"
error_log="./res/111.log"

# Initialize CSV file with header row
echo "Matrix Name,Rows,NNZs,Parameters (colProp, rowProp),TC_nnz_ratio,du_spmv (ms),cdspmv (ms)" > "$csv_file"

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
      # Extract rows and nnzs from the output
      rows=$(echo "$output" | grep -oP "rows = \K[0-9]+")
      nnzs=$(echo "$output" | grep -oP "nnzs = \K[0-9]+")

      # Split the output by sections for each rowProp and colProp
      echo "$output" | awk -v matrix_name="$matrix_name" -v rows="$rows" -v nnzs="$nnzs" '
      /rowProp:/ { rowProp=$2 }
      /colProp:/ { colProp=$2 }
      /TC_nnz_ratio =/ { tc_nnz_ratio=$3 }
      /du_spmv:/ { du_spmv=$2 + 0 }  # Ensure to extract the numerical part
      /cdspmv:/ { cdspmv=$2 + 0 }  # Similarly process cdspmv
      /Success!/ {
        missing_fields = ""
        if (rowProp == "") missing_fields = missing_fields "rowProp "
        if (colProp == "") missing_fields = missing_fields "colProp "
        if (tc_nnz_ratio == "") missing_fields = missing_fields "TC_nnz_ratio "
        if (du_spmv == "") missing_fields = missing_fields "du_spmv "
        if (cdspmv == "") missing_fields = missing_fields "cdspmv "
        
        if (missing_fields == "") {
          # If no fields are missing, print the result
          printf "%s,%s,%s,(%s, %s),%s,%s,%s\n", matrix_name, rows, nnzs, colProp, rowProp, tc_nnz_ratio, du_spmv, cdspmv
        } else {
          # If there are missing fields, log them
          print "Incomplete data for " matrix_name ": Missing fields - " missing_fields >> "'"$error_log"'"
          print "Output for debugging: " >> "'"$error_log"'"
          print $0 >> "'"$error_log"'"
        }
        # Reset variables for the next record
        rowProp=""; colProp=""; tc_nnz_ratio=""; du_spmv=""; cdspmv=""
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
