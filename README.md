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

> Run `pip install -r requirements.txt` to install dependencies.

---

## Data

The public log datasets used in the paper can be found in the repo [loghub](https://github.com/logpai/loghub).
In this repository, the BGL dataset under 100logs setting is proposed for a quick hands-up.

For generating the data files, please refer to the implementation repo of [deep-loglizer](https://github.com/logpai/deep-loglizer).



## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/MahsaRz/ALogSCAN.git
   cd ALogSCAN
