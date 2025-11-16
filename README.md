# ALogSCAN

## 🧠 Description

`ALogSCAN` is a self-supervised log anomaly detection framework designed for cloud environments. It leverages a **dual-network architecture** and a novel technique called **Dynamic Frequency-based Log Filtering (DFLF)** to effectively learn from normal patterns in unlabeled log data. 

By separating learning into **AE (AutoEncoder)** and **EO (Encoder Only)** streams and applying frequency-based masking, ALogSCAN adapts to dynamic log distributions and achieves high detection performance with minimal supervision.

We evaluate ALogSCAN on benchmark datasets (HDFS, BGL, and ERDC), showing superior performance in terms of both accuracy and inference time.

## 🔍 Features
- Dynamic Frequency-based Log Filtering (DFLF)
- Dual-network architecture: AE & EO
- Self-supervised learning, no labels required

---

## 📁 Project Structure
```
ALogSCAN/ ├── data/ # Log instances and processed input data ├── src/ │ ├── models/ # AE and EO architectures and losses │ ├── utils/ # Preprocessing, masking, and helpers │ ├── train.py # Training pipeline for ALogSCAN │ └── test.py # Evaluation script ├── logs/ # Log files, metrics, and checkpoints └── README.md
```

## ⚙️ Environment

**Key Packages:**

- Python 3.8+
- PyTorch ≥ 1.11.0 (with CUDA 11.3 support)
- scikit-learn
- numpy
- pandas

   Run `pip install -r requirements.txt` to install dependencies.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train on a dataset:
   ```bash
   cd src
   # Preprocess dataset
   python preprocess_dataset.py --dataset Apache --window_size 100
   
   # Train model
   python train_custom_dataset.py --datasets apache --window_size 100 --epochs 50
   ```

4. Run inference on your logs:
   ```bash
   python production_inference.py \
       --log_file /path/to/your/logs.txt \
       --model_dir experiment_records/<model_id> \
       --output results.json
   ```

📖 **For detailed instructions**, see [TRAINING_AND_INFERENCE_GUIDE.md](TRAINING_AND_INFERENCE_GUIDE.md)

---

## 📚 New Features

### Training on Custom Datasets

You can now train ALogSCAN on any of the 16 available datasets or your own data:

```bash
# Preprocess any dataset
python preprocess_dataset.py --dataset Linux --window_size 100

# Train on multiple datasets
python train_custom_dataset.py --datasets apache linux spark --window_size 100
```

### Production Inference

Use the new `production_inference.py` script for real anomaly detection:

```bash
python production_inference.py \
    --log_file server.log \
    --model_dir experiment_records/b69e9392 \
    --window_size 100 \
    --output anomalies.json
```

This performs **actual neural network inference** on your log files and outputs:
- Anomaly scores for each log window
- Detected anomalies with severity levels
- Detailed reports in JSON or CSV format

See `TRAINING_AND_INFERENCE_GUIDE.md` for complete documentation.

---

## Data

The public log datasets used in the paper can be found in the repo [loghub](https://github.com/logpai/loghub).
In this repository, the BGL dataset under 100logs setting is proposed for a quick hands-up.

For generating the data files, please refer to the implementation repo of [deep-loglizer](https://github.com/logpai/deep-loglizer).

**📦 Available Datasets:** This repository includes 16 pre-parsed log datasets in `data/logs/`:
- BGL, HDFS, Apache, Linux, Mac, OpenSSH, OpenStack, Spark, Thunderbird, Windows, Zookeeper, Android, Hadoop, HealthApp, HPC, Proxifier

See `TRAINING_AND_INFERENCE_GUIDE.md` for details on training with these datasets.

## 🚀 Getting Started

### Quick Start (Interactive)

```bash
# Run the interactive quick start tool
cd src
python quick_start.py
```

This will guide you through:
1. Preprocessing datasets
2. Training models
3. Running inference on your logs

### Manual Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/MahsaRz/ALogSCAN.git
   cd ALogSCAN
