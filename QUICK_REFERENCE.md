# Quick Reference: ALogSCAN Commands

## 🎯 Most Common Commands

### Train on an Existing Dataset
```bash
cd src
python preprocess_dataset.py --dataset Apache --window_size 100
python train_custom_dataset.py --datasets apache --window_size 100 --epochs 50
```

### Run Inference on Your Logs
```bash
python production_inference.py \
    --log_file /path/to/your/logs.txt \
    --model_dir experiment_records/b69e9392 \
    --output results.json
```

### Interactive Mode
```bash
python quick_start.py
```

---

## 📦 Preprocessing Commands

### Basic Preprocessing
```bash
python preprocess_dataset.py --dataset Apache --window_size 100
```

### Custom Parameters
```bash
python preprocess_dataset.py \
    --dataset Linux \
    --window_size 50 \
    --test_ratio 0.3 \
    --output_dir ../data/processed/linux_custom
```

### Batch Preprocess All Datasets
```bash
for dataset in Apache Linux Spark; do
    python preprocess_dataset.py --dataset $dataset --window_size 100
done
```

---

## 🚀 Training Commands

### Quick Training
```bash
python train_custom_dataset.py --datasets apache --window_size 100 --epochs 50
```

### Multiple Datasets
```bash
python train_custom_dataset.py --datasets apache linux spark --window_size 100
```

### Advanced Training
```bash
python train_custom_dataset.py \
    --datasets apache \
    --window_size 100 \
    --epochs 100 \
    --batch_size 128 \
    --hidden_size 256 \
    --masking_strategy statistical \
    --run_times 3 \
    --gpu 0
```

---

## 🔮 Inference Commands

### Basic Inference
```bash
python production_inference.py \
    --log_file server.log \
    --model_dir experiment_records/b69e9392
```

### Save Results to JSON
```bash
python production_inference.py \
    --log_file app.log \
    --model_dir experiment_records/b69e9392 \
    --output anomalies.json
```

### Save Results to CSV
```bash
python production_inference.py \
    --log_file system.log \
    --model_dir experiment_records/b69e9392 \
    --output anomalies.csv
```

### Custom Window Size and Threshold
```bash
python production_inference.py \
    --log_file logs.txt \
    --model_dir my_model \
    --window_size 50 \
    --threshold_percentile 90 \
    --output results.json
```

### Quiet Mode (No Progress Messages)
```bash
python production_inference.py \
    --log_file logs.txt \
    --model_dir my_model \
    --quiet \
    --output results.json
```

---

## 📊 Available Datasets

Run preprocessing on any of these:

```bash
# System logs
python preprocess_dataset.py --dataset Linux --window_size 100
python preprocess_dataset.py --dataset Mac --window_size 100
python preprocess_dataset.py --dataset Windows --window_size 100
python preprocess_dataset.py --dataset Android --window_size 100

# Server/Application logs
python preprocess_dataset.py --dataset Apache --window_size 100
python preprocess_dataset.py --dataset OpenSSH --window_size 100
python preprocess_dataset.py --dataset OpenStack --window_size 100
python preprocess_dataset.py --dataset Proxifier --window_size 100

# Big Data / HPC logs
python preprocess_dataset.py --dataset HDFS --window_size 100
python preprocess_dataset.py --dataset Hadoop --window_size 100
python preprocess_dataset.py --dataset Spark --window_size 100
python preprocess_dataset.py --dataset BGL --window_size 100
python preprocess_dataset.py --dataset Thunderbird --window_size 100
python preprocess_dataset.py --dataset HPC --window_size 100

# Other
python preprocess_dataset.py --dataset Zookeeper --window_size 100
python preprocess_dataset.py --dataset HealthApp --window_size 100
```

---

## 🎓 Parameter Reference

### Preprocessing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | Required | Dataset name (Apache, Linux, etc.) |
| `--window_size` | 100 | Logs per session window |
| `--test_ratio` | 0.2 | Ratio of test data (0.0-1.0) |
| `--session_level` | entry | Session level (entry/secs) |
| `--without_duplicate` | True | Remove consecutive duplicates |
| `--output_dir` | Auto | Output directory for processed files |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--datasets` | Required | List of datasets to train on |
| `--window_size` | 100 | Window size (must match preprocessing) |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--hidden_size` | 128 | Hidden layer size (32/64/128/256) |
| `--masking_strategy` | statistical | DFLF strategy (random/fixed-topk/statistical) |
| `--run_times` | 3 | Training runs for statistics |
| `--gpu` | 0 | GPU device ID (-1 for CPU) |

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--log_file` | Required | Path to log file to analyze |
| `--model_dir` | Required | Trained model directory |
| `--window_size` | 100 | Logs per window |
| `--threshold_percentile` | 85 | Anomaly threshold (0-100) |
| `--output` | None | Output file (.json or .csv) |
| `--use_parser` | None | Log parser (drain/spell) |
| `--quiet` | False | Suppress progress messages |

---

## 🔍 File Locations

### Input Data
- Raw datasets: `data/logs/<dataset>/`
- Preprocessed data: `data/processed/<dataset>/`

### Model Outputs
- Trained models: `experiment_records/<model_id>/`
- Model files: `params.json`, `model.ckpt`
- Results: `results.csv`, `benchmark_result.csv`

### Inference Outputs
- Default: Console output
- With `--output`: JSON or CSV file

---

## 💡 Tips and Tricks

### Speed Up Training
```bash
# Reduce epochs for quick experiments
python train_custom_dataset.py --datasets apache --epochs 30

# Increase batch size if you have GPU memory
python train_custom_dataset.py --datasets apache --batch_size 256 --gpu 0

# Reduce run times for single experiment
python train_custom_dataset.py --datasets apache --run_times 1
```

### Adjust Anomaly Sensitivity
```bash
# More sensitive (detect more anomalies)
python production_inference.py --log_file logs.txt --model_dir model/ --threshold_percentile 70

# Less sensitive (detect only high-confidence anomalies)
python production_inference.py --log_file logs.txt --model_dir model/ --threshold_percentile 95
```

### Work with Large Log Files
```bash
# Use smaller windows for faster processing
python production_inference.py --log_file huge.log --model_dir model/ --window_size 50

# Or split the file first
split -l 10000 huge.log chunk_
for file in chunk_*; do
    python production_inference.py --log_file $file --model_dir model/ --output ${file}.json
done
```

---

## 📞 Common Issues

### "Preprocessed data not found"
→ Run preprocessing first: `python preprocess_dataset.py --dataset <name>`

### "EventTemplate column not found"
→ Your CSV needs parsing. Use Drain or Deep-loglizer first.

### "Out of memory"
→ Reduce `--batch_size` or `--window_size`

### "Model doesn't generalize to my logs"
→ Train on similar log types or use larger datasets

---

## 📖 More Information

- Full guide: `TRAINING_AND_INFERENCE_GUIDE.md`
- Interactive mode: `python quick_start.py`
- Example workflow: `bash example_workflow.sh` (or `.ps1` on Windows)
