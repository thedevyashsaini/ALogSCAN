# Training and Inference Guide for ALogSCAN

This guide explains how to train ALogSCAN on custom datasets and perform inference on real log data.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Training on Existing Datasets](#training-on-existing-datasets)
3. [Training on Your Own Dataset](#training-on-your-own-dataset)
4. [Production Inference](#production-inference)
5. [Available Datasets](#available-datasets)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Train on a Pre-parsed Dataset (e.g., Apache)

```bash
# Step 1: Preprocess the dataset
cd src
python preprocess_dataset.py --dataset Apache --window_size 100

# Step 2: Train the model
python train_custom_dataset.py --datasets apache --window_size 100 --epochs 50

# Step 3: Run inference on your own logs
python production_inference.py \
    --log_file /path/to/your/logs.txt \
    --model_dir experiment_records/<model_id> \
    --output results.json
```

---

## 📚 Training on Existing Datasets

You have **16 different log datasets** available in `data/logs/`:

- **BGL** (BlueGene/L supercomputer)
- **HDFS** (Hadoop distributed file system)
- **Apache** (Apache web server)
- **Linux** (Linux system logs)
- **Mac** (macOS system logs)
- **OpenSSH** (SSH server logs)
- **OpenStack** (OpenStack cloud platform)
- **Spark** (Apache Spark)
- **Thunderbird** (Thunderbird supercomputer)
- **Windows** (Windows event logs)
- **Zookeeper** (Apache Zookeeper)
- **Android** (Android system logs)
- **Hadoop** (Hadoop framework)
- **HealthApp** (Healthcare application logs)
- **HPC** (High-performance computing)
- **Proxifier** (Proxifier proxy client)

### Step 1: Preprocess a Dataset

The datasets are in **raw CSV format** (with `EventTemplate` column). You need to preprocess them first:

```bash
cd src

# Preprocess Apache logs
python preprocess_dataset.py --dataset Apache --window_size 100

# Preprocess Linux logs
python preprocess_dataset.py --dataset Linux --window_size 50

# Preprocess multiple datasets
python preprocess_dataset.py --dataset Spark --window_size 100
python preprocess_dataset.py --dataset Zookeeper --window_size 100
```

**What this does:**
- Reads the structured CSV file from `data/logs/<dataset>/`
- Creates sliding windows of log sequences
- Splits into train/test sets (80/20 by default)
- Saves preprocessed files to `data/processed/<dataset>/`

**Output files:**
```
data/processed/apache/
  ├── data_desc_wo_duplicate_100entry.json    # Dataset metadata
  ├── session_train_wo_duplicate_100entry.pkl # Training sessions
  └── session_test_wo_duplicate_100entry.pkl  # Test sessions
```

### Step 2: Train on Preprocessed Dataset

```bash
# Train on single dataset
python train_custom_dataset.py --datasets apache --window_size 100 --epochs 50

# Train on multiple datasets
python train_custom_dataset.py --datasets apache linux spark --window_size 100

# Advanced options
python train_custom_dataset.py \
    --datasets apache \
    --window_size 100 \
    --epochs 100 \
    --batch_size 128 \
    --hidden_size 256 \
    --masking_strategy statistical \
    --run_times 3
```

**Key Parameters:**
- `--datasets`: List of dataset names (must be preprocessed first)
- `--window_size`: Must match preprocessing window size
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 64)
- `--hidden_size`: Hidden layer size (32, 64, 128, or 256)
- `--masking_strategy`: DFLF strategy (`random`, `fixed-topk`, or `statistical`)
- `--run_times`: Number of training runs for statistical significance

**Training Output:**
```
experiment_records/
  └── <unique_id>/
      ├── params.json       # Model configuration
      ├── model.ckpt        # Trained weights
      └── results.csv       # Performance metrics
```

---

## 🎯 Training on Your Own Dataset

### Option A: You Have Parsed Logs (EventTemplate CSV)

If you've already parsed your logs using **Drain**, **Spell**, or similar parsers:

**Required CSV format:**
```csv
LineId,EventTemplate,Label
1,User <*> logged in,Normal
2,Failed login attempt for user <*>,Anomaly
3,Database connection established,Normal
...
```

**Steps:**
```bash
# 1. Place your CSV in data/logs/
mkdir -p data/logs/myapp
cp your_logs_structured.csv data/logs/myapp/myapp_2k.log_structured.csv

# 2. Preprocess
python preprocess_dataset.py --dataset myapp --window_size 100

# 3. Train
python train_custom_dataset.py --datasets myapp --window_size 100 --epochs 50
```

### Option B: You Have Raw Logs (No Parsing)

If you have **raw, unparsed log files**:

**1. First, parse your logs using a log parser:**

You can use:
- **Drain** (recommended): https://github.com/logpai/Drain3
- **Spell**: https://github.com/logpai/logparser
- **Deep-loglizer**: https://github.com/logpai/deep-loglizer

**2. Or use the simple hash-based approach (for testing):**

```python
# This will work but is less accurate
python production_inference.py \
    --log_file your_raw_logs.txt \
    --model_dir experiment_records/b69e9392 \
    --use_parser hash
```

**3. Recommended: Use Deep-Loglizer for preprocessing**

Follow the instructions at: https://github.com/logpai/deep-loglizer

This will give you the same format as the existing datasets.

---

## 🔮 Production Inference

Use the `production_inference.py` script for **real anomaly detection** on your logs.

### Basic Usage

```bash
# Analyze logs with trained model
python production_inference.py \
    --log_file /var/log/application.log \
    --model_dir experiment_records/b69e9392 \
    --output anomalies.json
```

### Advanced Options

```bash
python production_inference.py \
    --log_file server.log \
    --model_dir my_model \
    --window_size 50 \
    --threshold_percentile 90 \
    --output results.csv
```

**Parameters:**
- `--log_file`: Path to raw log file
- `--model_dir`: Directory with trained model
- `--window_size`: Logs per window (default: 100)
- `--threshold_percentile`: Anomaly threshold (default: 85 = top 15%)
- `--output`: Save results to JSON or CSV
- `--use_parser`: Use `drain` or `spell` parser (requires installation)
- `--quiet`: Suppress progress messages

### Output Format

**JSON output:**
```json
{
  "metadata": {
    "log_file": "server.log",
    "total_lines": 10000,
    "total_windows": 100,
    "window_size": 100,
    "anomaly_threshold": 12.5,
    "threshold_percentile": 85
  },
  "statistics": {
    "total_anomalies": 15,
    "anomaly_rate": 0.15,
    "avg_anomaly_score": 8.2
  },
  "anomalies": [
    {
      "session_id": 42,
      "anomaly_score": 25.3,
      "start_line": 4200,
      "end_line": 4300,
      "severity": "HIGH",
      "log_sample": "ERROR: Database connection timeout..."
    }
  ]
}
```

**CSV output:**
```csv
session_id,anomaly_score,start_line,end_line,severity,log_sample
42,25.3,4200,4300,HIGH,"ERROR: Database connection timeout..."
15,18.7,1500,1600,MEDIUM,"WARN: High memory usage detected..."
```

---

## 📊 Available Datasets

All datasets in `data/logs/` are from **LogHub** (https://github.com/logpai/loghub) and are already parsed.

### Current Status

| Dataset | Status | Files | Notes |
|---------|--------|-------|-------|
| **BGL** | ✅ Preprocessed | Training ready | Supercomputer logs |
| **HDFS** | ⚠️ Need preprocessing | CSV available | Block-based sessions |
| **Apache** | ⚠️ Need preprocessing | CSV available | Web server logs |
| **Linux** | ⚠️ Need preprocessing | CSV available | System logs |
| **Mac** | ⚠️ Need preprocessing | CSV available | macOS logs |
| **OpenSSH** | ⚠️ Need preprocessing | CSV available | SSH logs |
| **OpenStack** | ⚠️ Need preprocessing | CSV available | Cloud platform |
| **Spark** | ⚠️ Need preprocessing | CSV available | Spark framework |
| **Thunderbird** | ⚠️ Need preprocessing | CSV available | Supercomputer |
| **Windows** | ⚠️ Need preprocessing | CSV available | Windows events |
| **Zookeeper** | ⚠️ Need preprocessing | CSV available | Zookeeper logs |
| **Android** | ⚠️ Need preprocessing | CSV available | Android system |
| **Hadoop** | ⚠️ Need preprocessing | CSV available | Hadoop framework |
| **HealthApp** | ⚠️ Need preprocessing | CSV available | Healthcare app |
| **HPC** | ⚠️ Need preprocessing | CSV available | HPC logs |
| **Proxifier** | ⚠️ Need preprocessing | CSV available | Proxy client |

### Preprocessing All Datasets

```bash
# Preprocess all at once
cd src
for dataset in Apache Linux Mac OpenSSH OpenStack Spark Thunderbird Windows Zookeeper Android Hadoop HealthApp HPC Proxifier; do
    python preprocess_dataset.py --dataset $dataset --window_size 100
done
```

---

## 🔧 Troubleshooting

### Issue: "Preprocessed data not found"

**Solution:**
```bash
# Make sure you preprocess first
python preprocess_dataset.py --dataset <your_dataset> --window_size 100

# Then train
python train_custom_dataset.py --datasets <your_dataset> --window_size 100
```

### Issue: "EventTemplate column not found"

**Solution:**
Your CSV needs an `EventTemplate` column. Parse your raw logs using:
- Drain: https://github.com/logpai/Drain3
- Deep-loglizer: https://github.com/logpai/deep-loglizer

### Issue: "Out of memory during training"

**Solution:**
```bash
# Reduce batch size
python train_custom_dataset.py --datasets mydata --batch_size 32

# Or reduce window size
python preprocess_dataset.py --dataset mydata --window_size 50
python train_custom_dataset.py --datasets mydata --window_size 50
```

### Issue: "Model not generalizing to my logs"

**Solution:**
The model was trained on specific log types. For best results:

1. **Train on similar logs**: If analyzing web server logs, train on Apache/OpenStack
2. **Use more data**: Larger training sets improve generalization
3. **Adjust window size**: Match your log patterns (e.g., smaller for rapid events)
4. **Fine-tune threshold**: Use `--threshold_percentile 90` or `95` for stricter detection

### Issue: "Inference is slow"

**Solution:**
```bash
# Use GPU if available
python production_inference.py --log_file logs.txt --model_dir model/ --gpu 0

# Or process in smaller batches
python production_inference.py --log_file logs.txt --model_dir model/ --window_size 50
```

---

## 📈 Performance Tips

### Training

- **Start small**: Use `--epochs 50` for initial experiments
- **Use GPU**: Set `--gpu 0` if CUDA is available
- **Statistical strategy**: Use `--masking_strategy statistical` (best performance)
- **Multiple runs**: Use `--run_times 3` for reliable metrics

### Inference

- **Batch processing**: Split large log files into chunks
- **Adjust threshold**: Lower percentile = more sensitive detection
- **Use parsers**: Install Drain for better template extraction

---

## 🎓 Understanding the Pipeline

### Training Pipeline
```
Raw CSV Logs
    ↓
[preprocess_dataset.py] → Creates sessions with sliding windows
    ↓
Preprocessed PKL files
    ↓
[train_custom_dataset.py] → Trains neural network
    ↓
Trained Model (params.json + model.ckpt)
```

### Inference Pipeline
```
Raw Log File
    ↓
[production_inference.py] → Parse & extract templates
    ↓
Create windows
    ↓
Neural network inference
    ↓
Anomaly scores & detection
    ↓
JSON/CSV results
```

---

## 📞 Need Help?

1. **Check the original demo scripts**: `demo.py`, `inference_demo.py`
2. **Review example datasets**: Look at BGL preprocessing in `data/processed/bgl/`
3. **Read the paper**: ALogSCAN uses DFLF (Dynamic Frequency-based Log Filtering)
4. **Inspect your data**: Make sure EventTemplate column exists in CSVs

---

## ✅ Summary

**To train on existing datasets:**
```bash
python preprocess_dataset.py --dataset Apache --window_size 100
python train_custom_dataset.py --datasets apache --window_size 100
```

**To use your own data:**
1. Parse logs to CSV with EventTemplate column (use Drain/Spell)
2. Place in `data/logs/yourdata/`
3. Preprocess and train as above

**To detect anomalies:**
```bash
python production_inference.py \
    --log_file your_logs.txt \
    --model_dir experiment_records/<model_id> \
    --output results.json
```

---

**Happy log mining! 🚀**
