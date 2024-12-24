cd "$(dirname "$0")"
command="../build/TCSpMVlib_perf"

# Define data directory
data_dir="../../../data/large_mtx/"

# Define output file
log="./res/222.log"

# Loop through each subdirectory in the data directory
for dir in "$data_dir"*/; do
    # Get the name of the subdirectory (without the path)
    sub_dir_name=$(basename "$dir")
    
    # Construct the full path to the .mtx file with the same name as the subdirectory
    mtx_file="$dir$sub_dir_name.mtx"
    
    # Check if the .mtx file exists
    if [[ -f "$mtx_file" ]]; then
        echo "Processing: $mtx_file" | tee -a "$log"
        # Execute the command with the .mtx file and append output to the log file
        $command "$mtx_file" 2>&1 | tee -a "$log"
    else
        echo "Warning: $mtx_file not found" | tee -a "$log"
    fi
done
