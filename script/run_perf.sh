#!/bin/bash

# 设置mtx文件目录路径
mtx_dir="../../../data/SF-graph/SNAP/mtx"
spmv_exec="../build/TCSpMVlib_perf"

# 创建或清空 perf.csv 文件，并添加标题行
echo "graph_name,our_perf,cusparse_perf,our_pre,cusparse_pre" > perf.csv

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
            output=$("$spmv_exec" "$mtx_file")
            echo "$output"
            
            # 从输出中提取性能数据
            our_perf=$(echo "$output" | grep "our_perf:" | awk '{print $2}')
            our_pre=$(echo "$output" | grep "our_pre:" | awk '{print $5}')
            cusparse_perf=$(echo "$output" | grep "cusparse_perf:" | awk '{print $2}')
            cusparse_pre=$(echo "$output" | grep "cusparse_pre:" | awk '{print $5}')
            
            # 将数据写入 CSV 文件
            echo "$graph_name,$our_perf,$cusparse_perf,$our_pre,$cusparse_pre" >> result/perf.csv
        else
            echo "No $graph_name.mtx file found in $graph_dir"
        fi
    fi
done

echo "Performance data has been written to perf.csv"