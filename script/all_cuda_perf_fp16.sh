# #!/bin/bash

# cd "$(dirname "$0")"
# command="../build/TCSpMVlib_perf"

# # Define data directory
# data_dir="../../../data/mtx/"

# # Define output files
# csv_file="./res/results_all_cd_fp16_2048npb.csv"
# error_log="./res/error_all_cd_fp16_2048npb.log"

# # Initialize CSV file with header row
# echo "Matrix Name,cd_perf (ms),cusparse_perf (ms)" > "$csv_file"

# # Clear error log file if it already exists
# > "$error_log"

# # Loop through each subdirectory in the data directory
# for dir in "$data_dir"/*/; do
#   # Get the matrix name from the directory name
#   matrix_name=$(basename "$dir")
  
#   # Construct the full path to the .mtx file
#   mtx_file="$dir/$matrix_name.mtx"
  
#   # Check if the matrix file exists
#   if [[ -f "$mtx_file" ]]; then
#     # Run the command and capture the output
#     output=$($command "$mtx_file" 2>&1)
    
#     # Check for "Success!" keyword in the output
#     if echo "$output" | grep -q "Success!"; then
#       # Extract the required values from the output
#       cd_perf=$(echo "$output" | grep -Po "cd_perf:\s*\K[0-9\.]+(?=\s*ms)")
#       cusparse_perf=$(echo "$output" | grep -Po "cusparse_perf:\s*\K[0-9\.]+(?=\s*ms)")

#       # Check if all required values were extracted
#       if [[ -n "$cd_perf" && -n "$cusparse_perf" ]]; then
#         # Write the data to the CSV file
#         echo "$matrix_name,$cd_perf,$cusparse_perf" >> "$csv_file"
#       else
#         # Log the matrix name to the error log
#         echo "Incomplete data for $matrix_name (missing performance values)" >> "$error_log"
#       fi
#     else
#       # Log missing "Success!" keyword to the error log
#       echo "Matrix $matrix_name did not complete successfully (missing 'Success!')" >> "$error_log"
#     fi
#   else
#     # Log missing matrix file to the error log
#     echo "Matrix file $mtx_file not found" >> "$error_log"
#   fi
# done

# echo "Processing completed. Results saved to $csv_file. Errors logged to $error_log."


#!/bin/bash

cd "$(dirname "$0")"
command="../build/TCSpMVlib_perf"

# Define data directory
data_dir="../../../data/mtx/"

# Define output files
csv_file="./res/results_all_cd_fp16_12_8_pre.csv"
error_log="./res/error_all_cd_fp16_12_8_pre.log"

# Initialize CSV file with header row
echo "Matrix Name,cd_perf (ms),cusparse_perf (ms)" > "$csv_file"

# Clear error log file if it already exists
> "$error_log"

# Loop through each subdirectory in the data directory
for dir in "$data_dir"/*/; do
  # Get the matrix name from the directory name
  matrix_name=$(basename "$dir")
  
  # Only process directories where the matrix name starts with a lowercase letter a-z
  # if [[ ! $matrix_name =~ ^[a-z] ]]; then
  #   continue  # Skip if the matrix name doesn't start with a lowercase letter
  # fi

  # Construct the full path to the .mtx file
  mtx_file="$dir/$matrix_name.mtx"

  # Check if the matrix file exists
  if [[ -f "$mtx_file" ]]; then
    # Run the command and capture the output
    output=$($command "$mtx_file" 2>&1)

    # Check for "Success!" keyword in the output
    if echo "$output" | grep -q "Success!"; then
      # Extract the required values from the output
      cd_perf=$(echo "$output" | grep -Po "cd_perf:\s*\K[0-9\.]+(?=\s*ms)")
      cusparse_perf=$(echo "$output" | grep -Po "cusparse_perf:\s*\K[0-9\.]+(?=\s*ms)")

      # Check if all required values were extracted
      if [[ -n "$cd_perf" && -n "$cusparse_perf" ]]; then
        # Write the data to the CSV file
        echo "$matrix_name,$cd_perf,$cusparse_perf" >> "$csv_file"
      fi
    fi
  fi
done

echo "Processing completed. Results saved to $csv_file."
