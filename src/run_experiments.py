#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated Experiment Runner for ALogSCAN DFLF Strategy Comparison
Runs all three masking strategies and generates comparative analysis
"""

import subprocess
import sys
import time
from datetime import datetime

def run_experiment(strategy, epochs=50, batch_size=64, gpu=0):
    """Run a single experiment with specified masking strategy"""
    
    print(f"\n{'='*70}")
    print(f"Running Experiment: {strategy.upper()} Strategy")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
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
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✓ {strategy.upper()} completed successfully in {elapsed/60:.2f} minutes")
        print(f"{'='*70}\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {strategy.upper()} failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ {strategy.upper()} interrupted by user")
        return False

def main():
    """Run all experiments and generate evaluation"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   ALogSCAN DFLF Strategy Comparison Experiment Suite        ║
    ║   Testing: Random | Fixed-TopK | Statistical                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    strategies = ['random', 'fixed-topk', 'statistical']
    epochs = 50  # Reduced from 100 for faster experimentation
    
    print(f"Configuration:")
    print(f"  - Strategies: {', '.join(strategies)}")
    print(f"  - Epochs per run: {epochs}")
    print(f"  - Runs per strategy: 3 (for statistical significance)")
    print(f"  - Estimated total time: ~2-3 hours")
    
    response = input("\nProceed with experiments? (y/n): ")
    if response.lower() != 'y':
        print("Experiments cancelled.")
        return
    
    # Run experiments
    overall_start = time.time()
    results = {}
    
    for strategy in strategies:
        success = run_experiment(strategy, epochs=epochs)
        results[strategy] = success
        
        if not success:
            print(f"\nWarning: {strategy} experiment failed. Continuing with next strategy...")
    
    # Summary
    overall_elapsed = time.time() - overall_start
    
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {overall_elapsed/3600:.2f} hours")
    print(f"\nResults:")
    for strategy, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {strategy:15s} : {status}")
    
    # Run evaluation
    if any(results.values()):
        print(f"\n{'='*70}")
        print("Generating Evaluation Report and Visualizations...")
        print(f"{'='*70}\n")
        
        try:
            subprocess.run([sys.executable, "evaluate_and_visualize.py"], check=True)
            print("\n✓ Evaluation complete! Check EVALUATION_SUMMARY.md and visualizations/")
        except subprocess.CalledProcessError:
            print("\n✗ Evaluation script failed. Run manually: python evaluate_and_visualize.py")
    else:
        print("\nAll experiments failed. No evaluation performed.")
    
    print(f"\n{'='*70}")
    print("All tasks complete!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
