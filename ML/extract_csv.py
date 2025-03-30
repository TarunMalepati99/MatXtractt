#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def extract_matching_rows():
    # 读取两个CSV文件
    test_perf_df = pd.read_csv("329_fp64.csv")
    all_df = pd.read_csv("all_fp64.csv")
    
    # 获取test_spmv_performance.csv的第一列（矩阵名）
    matrix_names = test_perf_df.iloc[:, 0].tolist()
    
    # 假设all.csv的第一列名称可能不同，所以我们获取第一列的名称
    all_first_col_name = all_df.columns[0]
    
    # 从all.csv中提取匹配的行
    all_extract_df = all_df[all_df[all_first_col_name].isin(matrix_names)]
    
    # 确保提取的行顺序与test_spmv_performance.csv相同
    all_extract_df = all_extract_df.set_index(all_first_col_name)
    all_extract_df = all_extract_df.reindex(index=matrix_names)
    all_extract_df = all_extract_df.reset_index()
    
    # 保存提取的行到all_extract.csv
    all_extract_df.to_csv("all_extract_fp64.csv", index=False)
    
    # 验证提取是否完成
    print(f"从all.csv中成功提取了 {len(all_extract_df)} 行数据到all_extract.csv")
    
    # 检查是否所有矩阵名都被找到
    found_matrices = set(all_extract_df[all_first_col_name].tolist())
    all_test_matrices = set(matrix_names)
    if found_matrices != all_test_matrices:
        missing = all_test_matrices - found_matrices
        print(f"警告：以下矩阵在all.csv中未找到：{missing}")

if __name__ == "__main__":
    extract_matching_rows()