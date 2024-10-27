#!/bin/bash



# ./build/TCSpMVlib_tcperftest ../../data/SF-graph/SNAP/mtx/roadNet-TX/roadNet-TX.mtx 
# ./build/TCSpMVlib_tcperftest ../../data/SF-graph/SNAP/mtx/p2p-Gnutella06/p2p-Gnutella06.mtx
# ./build/TCSpMVlib_tcperftest ../../data/SF-graph/SNAP/mtx/ca-HepTh/ca-HepTh.mtx
# ./build/TCSpMVlib_tcperftest ../../data/SF-graph/SNAP/mtx/wiki-Vote/wiki-Vote.mtx



# Define command to execute
command="./build/TCSpMVlib_tcperftest"

# Define data directory
data_dir="../../data/SF-graph/SNAP/mtx/"

# Log file for command output
log_file="res_cd_SNAP_16.log"

# Clear the log file if it already exists
> "$log_file"

# Loop through each subdirectory in the data directory
for dir in "$data_dir"/*/; do
  # Get the matrix name from the directory name
  matrix_name=$(basename "$dir")
  
  # Construct the full path to the .mtx file
  mtx_file="$dir/$matrix_name.mtx"
  
  # Execute the command and redirect all output (stdout and stderr) to log file
  if [[ -f "$mtx_file" ]]; then
    echo "Processing $matrix_name" | tee -a "$log_file"
    $command "$mtx_file" >> "$log_file" 2>&1
  else
    echo "Matrix file $mtx_file not found" | tee -a "$log_file"
  fi
done

echo "Log saved to $log_file"



