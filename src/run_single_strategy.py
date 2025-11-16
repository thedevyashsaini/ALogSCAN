#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Strategy Experiment Runner - For Parallel Execution Across Multiple Machines
Run this on each laptop with a different strategy
"""

import subprocess
import sys
import time
import argparse
from datetime import datetime

def run_single_strategy(strategy, epochs=50, batch_size=64, gpu=0):
    """Run experiments for a single masking strategy"""
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║   ALogSCAN Single Strategy Experiment                       ║
    ║   Strategy: {strategy.upper():45s}║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    cmd = [
        sys.executable,
        "demo.py",
        "--datasets", "bgl",
        "--window_size", "100",
        "--session_size", "100",
        "--epoches", str(epochs),
        "--batch_size", str(batch_size),
        "--gpu", str(gpu),
        "--masking_strategy", strategy
    ]
    
    print(f"Configuration:")
    print(f"  - Strategy: {strategy}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Runs: 3 (for statistical significance)")
    print(f"  - Estimated time: ~45-60 minutes")
    print(f"  - Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'='*70}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✓ {strategy.upper()} completed successfully!")
        print(f"  Time elapsed: {elapsed/60:.2f} minutes")
        print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        print(f"\n📁 Results saved to: experiment_records/logsd_benchmark_result.csv")
        print(f"\n🔄 Next steps:")
        print(f"  1. Copy 'experiment_records/logsd_benchmark_result.csv' to main laptop")
        print(f"  2. Use merge_results.py to combine all three CSV files")
        print(f"  3. Run evaluate_and_visualize.py on merged results")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {strategy.upper()} failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ {strategy.upper()} interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run single DFLF strategy experiment')
    parser.add_argument('strategy', choices=['random', 'fixed-topk', 'statistical'],
                       help='Masking strategy to run')
    parser.add_argument('--epoches', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0, use -1 for CPU)')
    
    args = parser.parse_args()
    
    print(f"\n🚀 Starting single strategy experiment: {args.strategy.upper()}\n")
    
    success = run_single_strategy(args.strategy, epochs=args.epoches, gpu=args.gpu)
    
    if success:
        print(f"\n✅ SUCCESS! Results ready for merging.")
    else:
        print(f"\n❌ FAILED! Check error messages above.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
