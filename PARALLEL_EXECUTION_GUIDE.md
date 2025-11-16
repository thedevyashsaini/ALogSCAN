# Parallel Experiment Execution Guide

## 🚀 Running Experiments on 3 Laptops Simultaneously

This guide shows how to run the three DFLF strategies in parallel across multiple machines and merge results.

---

## Setup (Do on ALL 3 Laptops)

```bash
# 1. Clone/copy the repository
git clone <repo-url>  # or copy via USB/network
cd ALogSCAN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
cd src
python demo.py --help  # Should show all options including --masking_strategy
```

---

## Execution Plan

### **Laptop 1: Random Strategy**
```bash
cd src
python run_single_strategy.py random --epoches 50 --gpu 0
```

### **Laptop 2: Fixed-TopK Strategy**
```bash
cd src
python run_single_strategy.py fixed-topk --epoches 50 --gpu 0
```

### **Laptop 3: Statistical Strategy**
```bash
cd src
python run_single_strategy.py statistical --epoches 50 --gpu 0
```

**Time estimate:** ~45-60 minutes per laptop (runs simultaneously)

---

## After All Laptops Finish

### **Step 1: Collect CSV Files**

Each laptop will generate:
```
experiment_records/logsd_benchmark_result.csv
```

**Copy these to your main laptop and rename:**
```bash
# On main laptop, create a temp folder
mkdir merged_results
cd merged_results

# Copy from each laptop (via USB, network share, email, etc.)
# Rename to avoid overwriting:
cp laptop1_results.csv random_results.csv
cp laptop2_results.csv fixed_topk_results.csv
cp laptop3_results.csv statistical_results.csv
```

### **Step 2: Merge Results**

```bash
cd ../src
python merge_results.py ../merged_results/random_results.csv \
                        ../merged_results/fixed_topk_results.csv \
                        ../merged_results/statistical_results.csv
```

This creates: `experiment_records/logsd_benchmark_result.csv` (merged)

### **Step 3: Generate Evaluation**

```bash
python evaluate_and_visualize.py
```

This generates:
- `visualizations/metric_comparison.png`
- `visualizations/confusion_matrix_*.png`
- `EVALUATION_SUMMARY.md`

---

## File Transfer Options

### Option A: USB Drive
```bash
# On each laptop after completion
cp experiment_records/logsd_benchmark_result.csv /path/to/usb/laptop1_results.csv
```

### Option B: Network Share (Windows)
```bash
# On each laptop
copy experiment_records\logsd_benchmark_result.csv \\MainLaptop\Share\laptop1_results.csv
```

### Option C: Email/Cloud
Just email the CSV files to yourself or upload to Google Drive/Dropbox.

### Option D: Git (Cleanest)
```bash
# On each laptop after completion
git checkout -b results-random  # (or results-fixed-topk, results-statistical)
git add experiment_records/logsd_benchmark_result.csv
git commit -m "Results for random strategy"
git push origin results-random

# On main laptop
git pull origin results-random
git pull origin results-fixed-topk
git pull origin results-statistical
# Copy each CSV to merged_results/ folder
```

---

## Verification Checklist

Before merging, verify each CSV has:
```bash
# Check file exists and has content
wc -l laptop1_results.csv  # Should show ~4-6 lines (header + 3 runs)

# Check it has the right strategy
grep "random" laptop1_results.csv  # Should find entries
grep "fixed-topk" laptop2_results.csv
grep "statistical" laptop3_results.csv
```

---

## Troubleshooting

### Problem: "File not found" when merging
**Solution:** Check file paths are correct
```bash
ls -la ../merged_results/  # Verify files are there
```

### Problem: "No masking_strategy column"
**Solution:** Re-run with correct code version
```bash
# Verify code has masking_strategy support
grep "masking_strategy" demo.py  # Should find the argument
```

### Problem: Duplicate rows in merged file
**Solution:** The merge script automatically removes duplicates, but check:
```bash
# After merging
python -c "import pandas as pd; df = pd.read_csv('experiment_records/logsd_benchmark_result.csv'); print(df['masking_strategy'].value_counts())"
# Should show 3 runs per strategy (9 total)
```

---

## Expected Timeline

| Time | Activity |
|------|----------|
| 0:00 | Start all 3 laptops simultaneously |
| 0:45-1:00 | All laptops finish |
| 1:00-1:10 | Copy CSV files to main laptop |
| 1:10-1:15 | Run merge_results.py |
| 1:15-1:20 | Run evaluate_and_visualize.py |
| **1:20** | **All done! Results ready for submission** |

**Total time: ~1.5 hours** (vs ~3 hours sequential)

---

## Quick Reference Commands

```bash
# On each laptop (different strategy)
python run_single_strategy.py [random|fixed-topk|statistical] --epoches 50

# On main laptop (after collecting CSVs)
python merge_results.py file1.csv file2.csv file3.csv
python evaluate_and_visualize.py

# Check merged results
head experiment_records/logsd_benchmark_result.csv
```

---

## What You Get

After merging and evaluation:

1. **Merged CSV:** All 9 runs (3 per strategy) in one file
2. **Comparison Table:** Mean ± std for each metric and strategy
3. **Visualizations:** Bar charts and confusion matrices
4. **Summary Report:** Auto-generated evaluation summary

Ready for submission! 🎉
