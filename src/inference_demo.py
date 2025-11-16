#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Inference Demo for ALogSCAN
This script demonstrates anomaly detection on BGL log data using a trained model.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.append("../")

from logAnalyzer.models.ss_network import SS_Net
from logAnalyzer.common.preprocess import FeatureExtractor
from logAnalyzer.common.dataloader import load_sessions, log_dataset
from logAnalyzer.common.utils import seed_everything


def load_trained_model(model_path, params_path):
    """Load a trained model from checkpoint"""
    # Load parameters
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    print(f"Loading model trained on: {params['dataset']}")
    print(f"Model configuration:")
    print(f"  - Hidden size: {params['hidden_size']}")
    print(f"  - Embedding dim: {params['embedding_dim']}")
    print(f"  - Encoder type: {params['encoder_type']}")
    print(f"  - Window size: {params['window_size']}")
    
    return params


def run_inference_demo(model_dir="experiment_records/b69e9392"):
    """Run interactive inference demo"""
    
    # Paths
    params_path = os.path.join(model_dir, "params.json")
    model_path = os.path.join(model_dir, "model.ckpt")
    
    if not os.path.exists(params_path) or not os.path.exists(model_path):
        print(f"Error: Model files not found in {model_dir}")
        return
    
    # Load model parameters
    params = load_trained_model(model_path, params_path)
    seed_everything(params["random_seed"])
    
    # Load test data
    print("\nLoading test data...")
    data_path = os.path.join(params['data_dir'], params['dataset'])
    
    raw_session_train, raw_session_test, session_size = load_sessions(
        data_dir=data_path,
        without_duplicate=params['without_duplicate'],
        session_size=params['session_size'],
        sampling_size=params['sampling_size'],
        session_level=params['session_level'],
        semi_supervised=params['semi_supervised'],
        show_statistics=False
    )
    
    # Feature extraction
    print("Extracting features...")
    ext = FeatureExtractor(**params)
    session_train = ext.fit_transform(raw_session_train)
    session_test = ext.transform(raw_session_test, datatype="test")
    
    # Create test dataset
    dataset_test = log_dataset(session_test, feature_type=params["feature_type"])
    labels_test = [sequence["window_anomalies"] for sequence in dataset_test.flatten_data_list]
    
    print(f"\nTest dataset statistics:")
    print(f"  - Total test sequences: {len(dataset_test)}")
    print(f"  - Anomaly ratio: {np.array(labels_test).sum() / len(labels_test):.4f}")
    
    # Create dataloader
    dataloader_test = DataLoader(dataset_test, batch_size=params['batch_size'], 
                                  shuffle=False, pin_memory=True)
    
    # Initialize model
    print("\nInitializing model...")
    model = SS_Net(meta_data=ext.meta_data, model_save_path=os.path.dirname(model_path), **params)
    
    # Load trained weights
    print(f"Loading trained weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=model.device))
    model.to(model.device)  # Ensure all model components are on the correct device
    model.eval()
    
    # Run inference
    print("\n" + "="*60)
    print("Running anomaly detection on test data...")
    print("="*60 + "\n")
    
    from collections import defaultdict
    from sklearn.metrics import (f1_score, recall_score, precision_score, 
                                  roc_auc_score, average_precision_score, 
                                  precision_recall_curve, auc, accuracy_score)
    import time
    
    store_dict = defaultdict(list)
    anomaly_scores = []
    
    with torch.no_grad():
        for idx, batch_input in enumerate(dataloader_test):
            # Move to device
            batch_input_device = {k: v.to(model.device) for k, v in batch_input.items()}
            
            # Forward pass
            return_dict = model.forward(batch_input_device)
            
            # Get anomaly scores (y_pred is already the anomaly score per sample)
            anomaly_score = return_dict["y_pred"]  # Higher score = more anomalous
            
            # Store results
            store_dict["session_idx"].extend(batch_input["session_idx"].cpu().numpy().flatten())
            store_dict["window_anomalies"].extend(batch_input["window_anomalies"].cpu().numpy().flatten())
            store_dict["anomaly_scores"].extend(anomaly_score.cpu().numpy().flatten())
            anomaly_scores.extend(anomaly_score.cpu().numpy().flatten())
    
    # Aggregate by session
    import pandas as pd
    store_df = pd.DataFrame(store_dict)
    
    # For session-level: sum anomaly scores and check if any window was anomalous
    session_df = store_df[["session_idx", "window_anomalies", "anomaly_scores"]].groupby("session_idx", as_index=False).sum()
    
    # Find optimal threshold using the anomaly scores
    y = (session_df["window_anomalies"] > 0).astype(int)
    
    # Use percentile-based threshold for prediction
    threshold = np.percentile(session_df["anomaly_scores"], 90)  # Top 10% as anomalies
    pred = (session_df["anomaly_scores"] > threshold).astype(int)
    
    # Calculate metrics
    precision, recall, _ = precision_recall_curve(y, pred)
    
    print("RESULTS:")
    print("="*60)
    print(f"  F1 Score:        {f1_score(y, pred):.4f}")
    print(f"  Recall:          {recall_score(y, pred):.4f}")
    print(f"  Precision:       {precision_score(y, pred):.4f}")
    print(f"  ROC-AUC:         {roc_auc_score(y, pred):.4f}")
    print(f"  PR-AUC:          {auc(recall, precision):.4f}")
    print(f"  Accuracy:        {accuracy_score(y, pred):.4f}")
    print("="*60)
    
    # Show some example predictions
    print("\nSample Predictions (diverse examples):")
    print("-"*60)
    print(f"{'Session':<10} {'True Label':<12} {'Prediction':<12} {'Anomaly Score':<15} {'Status':<10}")
    print("-"*60)
    
    # Show diverse examples
    examples_shown = 0
    
    # 1. True Positives (correctly detected anomalies)
    tp_indices = session_df[(y == 1) & (pred == 1)].head(3).index
    for idx in tp_indices:
        true_label = "Anomaly"
        pred_label = "Anomaly"
        score = session_df.loc[idx, 'anomaly_scores']
        status = "✓ Correct"
        session_id = int(session_df.loc[idx, 'session_idx'])
        print(f"{session_id:<10} {true_label:<12} {pred_label:<12} {score:<15.4f} {status:<10}")
        examples_shown += 1
    
    # 2. False Positives (false alarms)
    fp_indices = session_df[(y == 0) & (pred == 1)].head(2).index
    for idx in fp_indices:
        true_label = "Normal"
        pred_label = "Anomaly"
        score = session_df.loc[idx, 'anomaly_scores']
        status = "✗ Wrong"
        session_id = int(session_df.loc[idx, 'session_idx'])
        print(f"{session_id:<10} {true_label:<12} {pred_label:<12} {score:<15.4f} {status:<10}")
        examples_shown += 1
    
    # 3. False Negatives (missed anomalies)
    fn_indices = session_df[(y == 1) & (pred == 0)].head(2).index
    for idx in fn_indices:
        true_label = "Anomaly"
        pred_label = "Normal"
        score = session_df.loc[idx, 'anomaly_scores']
        status = "✗ Wrong"
        session_id = int(session_df.loc[idx, 'session_idx'])
        print(f"{session_id:<10} {true_label:<12} {pred_label:<12} {score:<15.4f} {status:<10}")
        examples_shown += 1
    
    # 4. True Negatives (correctly identified normal)
    tn_indices = session_df[(y == 0) & (pred == 0)].head(3).index
    for idx in tn_indices:
        true_label = "Normal"
        pred_label = "Normal"
        score = session_df.loc[idx, 'anomaly_scores']
        status = "✓ Correct"
        session_id = int(session_df.loc[idx, 'session_idx'])
        print(f"{session_id:<10} {true_label:<12} {pred_label:<12} {score:<15.4f} {status:<10}")
        examples_shown += 1
    
    print("-"*60)
    print(f"\nNote: Anomaly threshold = {threshold:.4f} (scores above this are flagged as anomalies)")
    
    # Summary
    total_sessions = len(session_df)
    anomalies_detected = pred.sum()
    true_anomalies = y.sum()
    
    print(f"\nSummary:")
    print(f"  Total sessions analyzed: {total_sessions}")
    print(f"  True anomalies: {true_anomalies}")
    print(f"  Detected anomalies: {anomalies_detected}")
    print(f"  Correctly identified: {(y == pred).sum()}/{total_sessions}")
    
    return model, session_df, y, pred


if __name__ == "__main__":
    print("="*60)
    print("ALogSCAN - Anomaly Detection Inference Demo")
    print("="*60)
    print()
    print("📝 About the data:")
    print("   - Dataset: BGL (BlueGene/L supercomputer logs)")
    print("   - Source: IBM BlueGene/L system at Lawrence Livermore National Lab")
    print("   - Contains: System logs with hardware/software failures marked")
    print("   - You're testing on the HOLDOUT test set (unseen during training)")
    print()
    
    # You can change this to use different trained models
    model_dir = "experiment_records/b69e9392"
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found!")
        print("Please train a model first using demo.py")
        sys.exit(1)
    
    run_inference_demo(model_dir)
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
