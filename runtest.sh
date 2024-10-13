# #!/bin/bash

# # Define command to execute
# command="./build/TCSpMVlib_tcperftest"

# # Define data directory
# data_dir="../../data/SF-graph/SNAP/mtx/"

# # Output CSV file
# output_file="results_our.csv"

# # Write header to the CSV file
# echo "Matrix Name,Runtime" > "$output_file"

# # Loop through each subdirectory in the data directory
# for dir in "$data_dir"/*/; do
#   # Get the matrix name from the directory name
#   matrix_name=$(basename "$dir")
  
#   # Construct the full path to the .mtx file
#   mtx_file="$dir/$matrix_name.mtx"
  
#   # Execute the command and capture the output
#   if [[ -f "$mtx_file" ]]; then
#     output=$($command "$mtx_file")
    
#     # Extract the runtime using grep and awk
#     runtime=$(echo "$output" | grep "SpMV CUDA kernel runtime =" | awk -F'= ' '{print $2}')
    
#     # Append the result to the CSV file
#     echo "$matrix_name,$runtime" >> "$output_file"
#   else
#     echo "Matrix file $mtx_file not found"
#   fi
# done

# echo "Results saved to $output_file"







./build/TCSpMVlib_tcperftest ../../data/SF-graph/SNAP/mtx/roadNet-TX/roadNet-TX.mtx 
./build/TCSpMVlib_tcperftest ../../data/SF-graph/SNAP/mtx/p2p-Gnutella06/p2p-Gnutella06.mtx
./build/TCSpMVlib_tcperftest ../../data/SF-graph/SNAP/mtx/ca-HepTh/ca-HepTh.mtx
./build/TCSpMVlib_tcperftest ../../data/SF-graph/SNAP/mtx/wiki-Vote/wiki-Vote.mtx