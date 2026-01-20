#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def extract_matching_rows():
    # Read two CSV files
    test_perf_df = pd.read_csv("329_fp64.csv")
    all_df = pd.read_csv("all_fp64.csv")
    
    # Get first column (matrix names) from test_spmv_performance.csv
    matrix_names = test_perf_df.iloc[:, 0].tolist()
    
    # Get first column name from all.csv (may differ)
    all_first_col_name = all_df.columns[0]
    
    # Extract matching rows from all.csv
    all_extract_df = all_df[all_df[all_first_col_name].isin(matrix_names)]
    
    # Ensure extracted rows follow the same order as test_spmv_performance.csv
    all_extract_df = all_extract_df.set_index(all_first_col_name)
    all_extract_df = all_extract_df.reindex(index=matrix_names)
    all_extract_df = all_extract_df.reset_index()
    
    # Save extracted rows to all_extract.csv
    all_extract_df.to_csv("all_extract_fp64.csv", index=False)
    
    # Verify extraction completion
    print(f"Successfully extracted {len(all_extract_df)} rows from all.csv to all_extract.csv")
    
    # Check if all matrix names were found
    found_matrices = set(all_extract_df[all_first_col_name].tolist())
    all_test_matrices = set(matrix_names)
    if found_matrices != all_test_matrices:
        missing = all_test_matrices - found_matrices
        print(f"Warning: The following matrices were not found in all.csv: {missing}")

if __name__ == "__main__":
    extract_matching_rows()