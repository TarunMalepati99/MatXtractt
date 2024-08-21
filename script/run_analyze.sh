#!/bin/bash

# 设置mtx文件目录路径
mtx_dir="../../../data/SF-graph/GAP/mtx"
analyze_exec="../build/TCSpMVlib_analyze"
log_file="result/GAP_1_10.log"


# 遍历所有子文件夹
for graph_dir in "$mtx_dir"/*; do
    # 确保这是一个目录
    if [[ -d "$graph_dir" ]]; then
        # 获取子文件夹名
        graph_name=$(basename "$graph_dir")
        # 构建graph_name.mtx的完整路径
        mtx_file="$graph_dir/$graph_name.mtx"
        
        # 检查graph_name.mtx文件是否存在
        if [[ -f "$mtx_file" ]]; then
             echo "Running test case for $graph_name with $mtx_file"
            
            # 执行命令并捕获输出
            output=$("$analyze_exec" "$mtx_file")
            
            # 提取'dense ratio = '后的数字
            dense_ratio=$(echo "$output" | grep -oP 'dense ratio = \K[0-9.]+')
            
            # 将graph_name和dense_ratio写入log文件
            if [[ -n "$dense_ratio" ]]; then
                echo "$graph_name, $dense_ratio" >> "$log_file"
            else
                echo "$graph_name, N/A" >> "$log_file"
            fi
        else
            echo "No $graph_name.mtx file found in $graph_dir"
        fi
    fi
done
