#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge Results from Multiple Parallel Experiments
Combines CSV files from different laptops into single results file
"""

import pandas as pd
import os
import sys
import argparse
from datetime import datetime

def merge_csv_files(file_paths, output_file='experiment_records/logsd_benchmark_result_merged.csv'):
    """Merge multiple CSV result files into one"""
    
    print(f"\n{'='*70}")
    print("Merging Experiment Results")
    print(f"{'='*70}\n")
    
    all_dfs = []
    
    for idx, file_path in enumerate(file_paths, 1):
        if not os.path.exists(file_path):
            print(f"⚠️  File {idx}: {file_path} - NOT FOUND, skipping...")
            continue
        
        try:
            df = pd.read_csv(file_path)
            print(f"✓ File {idx}: {file_path}")
            print(f"  - Rows: {len(df)}")
            if 'masking_strategy' in df.columns:
                strategies = df['masking_strategy'].unique()
                print(f"  - Strategies: {', '.join(strategies)}")
            all_dfs.append(df)
        except Exception as e:
            print(f"✗ File {idx}: {file_path} - ERROR: {e}")
            continue
    
    if not all_dfs:
        print("\n❌ No valid CSV files found! Cannot merge.")
        return False
    
    # Merge all dataframes
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates (in case of accidental reruns)
    original_len = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    if len(merged_df) < original_len:
        print(f"\n⚠️  Removed {original_len - len(merged_df)} duplicate rows")
    
    # Save merged results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Merge Complete!")
    print(f"{'='*70}")
    print(f"Output file: {output_file}")
    print(f"Total rows: {len(merged_df)}")
    
    if 'masking_strategy' in merged_df.columns:
        print(f"\nStrategy breakdown:")
        strategy_counts = merged_df['masking_strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            print(f"  - {strategy}: {count} runs")
    
    print(f"\n🎯 Next step:")
    print(f"   python evaluate_and_visualize.py")
    print(f"   (Will automatically use merged results)")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Merge CSV results from parallel experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge 3 CSV files from different laptops
  python merge_results.py random.csv fixed_topk.csv statistical.csv
  
  # Merge all CSV files in current directory
  python merge_results.py *.csv
  
  # Specify output file
  python merge_results.py -o merged.csv random.csv fixed.csv stat.csv
        """
    )
    
    parser.add_argument('files', nargs='+', help='CSV files to merge')
    parser.add_argument('-o', '--output', 
                       default='experiment_records/logsd_benchmark_result.csv',
                       help='Output merged CSV file (default: experiment_records/logsd_benchmark_result.csv)')
    
    args = parser.parse_args()
    
    success = merge_csv_files(args.files, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
