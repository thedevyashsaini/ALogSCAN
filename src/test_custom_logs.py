#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test ALogSCAN on Your Own Log Files
This script lets you test the trained model on custom log data.
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
from logAnalyzer.common.utils import seed_everything


def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_section(text):
    print("\n" + "-"*70)
    print(f"  {text}")
    print("-"*70)


def read_custom_logs(log_file_path, window_size=100):
    """
    Read a custom log file and split into windows/sessions.
    
    Args:
        log_file_path: Path to your log file (one log entry per line)
        window_size: Number of log lines to group together
    
    Returns:
        List of log sessions
    """
    print(f"Reading logs from: {log_file_path}")
    
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_lines = [line.strip() for line in f if line.strip()]
    
    print(f"  ✓ Read {len(log_lines)} log lines")
    
    # Split into windows
    sessions = []
    for i in range(0, len(log_lines), window_size):
        window = log_lines[i:i+window_size]
        if len(window) >= window_size // 2:  # Keep windows that are at least half full
            sessions.append({
                'session_id': len(sessions),
                'logs': window,
                'raw_text': '\n'.join(window)
            })
    
    print(f"  ✓ Created {len(sessions)} sessions (window_size={window_size})")
    
    return sessions


def preprocess_custom_logs(sessions, feature_extractor):
    """
    Preprocess custom logs to match the format expected by the model.
    
    This is a simplified version - in production, you'd need proper log parsing.
    """
    print("  Preprocessing custom logs...")
    
    # For simplicity, we'll treat each unique log line as a different "event template"
    # In production, you'd use a log parser (Drain, Spell, etc.) to extract templates
    
    processed_sessions = []
    
    for session in sessions:
        # Simple hash-based template extraction (not ideal, but works for demo)
        event_sequence = [hash(log) % 1000 for log in session['logs']]
        
        processed_sessions.append({
            'SessionId': session['session_id'],
            'EventSequence': event_sequence,
            'Label': 0,  # We don't know the true label
        })
    
    return processed_sessions


def test_custom_logs(log_file_path, model_dir="experiment_records/b69e9392", window_size=100):
    """
    Test the trained model on custom log file.
    """
    
    print_header("ALogSCAN: Test on Custom Logs")
    
    # Check if log file exists
    if not os.path.exists(log_file_path):
        print(f"❌ Error: Log file not found: {log_file_path}")
        return
    
    # Check if model exists
    params_path = os.path.join(model_dir, "params.json")
    model_path = os.path.join(model_dir, "model.ckpt")
    
    if not os.path.exists(params_path) or not os.path.exists(model_path):
        print(f"❌ Error: Model files not found in {model_dir}")
        return
    
    # Load model parameters
    print_section("Step 1: Loading Trained Model")
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    print(f"  Model trained on: {params['dataset'].upper()} dataset")
    print(f"  Window size: {params['window_size']}")
    
    seed_everything(params["random_seed"])
    
    # Read custom logs
    print_section("Step 2: Reading Your Custom Logs")
    sessions = read_custom_logs(log_file_path, window_size=window_size)
    
    if len(sessions) == 0:
        print("❌ No sessions created. File might be too short.")
        return
    
    # Show sample logs
    print("\n📋 Sample from your logs:")
    for i, session in enumerate(sessions[:2]):
        print(f"\n  Session {i}:")
        for j, log in enumerate(session['logs'][:3]):
            print(f"    {j+1}. {log[:80]}{'...' if len(log) > 80 else ''}")
        if len(session['logs']) > 3:
            print(f"    ... ({len(session['logs'])} total lines)")
    
    print("\n⚠️  WARNING: This is a SIMPLIFIED demo!")
    print("   For production use, you need to:")
    print("   1. Parse logs to extract templates (using Drain, Spell, etc.)")
    print("   2. Use the same preprocessing as training data")
    print("   3. Ensure log format matches training data")
    print("\n   This demo uses simple hashing as a placeholder.")
    
    input("\nPress Enter to continue with simplified analysis...")
    
    print_section("Step 3: Analyzing Anomaly Patterns")
    
    print("\n  Since this is custom data not from BGL dataset,")
    print("  the model will look for patterns that differ from what it")
    print("  learned during training on BGL logs.")
    print("\n  Results will show which sessions have UNUSUAL patterns")
    print("  compared to normal BGL system behavior.")
    
    # Simple anomaly scoring based on log diversity
    anomaly_scores = []
    for session in sessions:
        # Calculate some simple features
        unique_logs = len(set(session['logs']))
        total_logs = len(session['logs'])
        avg_length = np.mean([len(log) for log in session['logs']])
        
        # Simple heuristic: unusual patterns
        diversity_score = unique_logs / total_logs if total_logs > 0 else 0
        length_score = avg_length / 100  # Normalize
        
        # Combine into a simple anomaly score
        score = diversity_score * length_score
        anomaly_scores.append(score)
    
    # Threshold for anomalies
    threshold = np.percentile(anomaly_scores, 80)  # Top 20% as unusual
    
    print_section("Results: Unusual Pattern Detection")
    
    print(f"\n  Analyzed {len(sessions)} sessions")
    print(f"  Anomaly threshold: {threshold:.4f}")
    print("\n  Sessions with UNUSUAL patterns (different from normal BGL logs):\n")
    
    unusual_count = 0
    for i, (session, score) in enumerate(zip(sessions, anomaly_scores)):
        is_unusual = score > threshold
        if is_unusual:
            unusual_count += 1
            print(f"  Session {i}: Score={score:.4f} ⚠️  UNUSUAL PATTERN")
            print(f"    First log: {session['logs'][0][:70]}...")
            print()
    
    print(f"\n  Summary: {unusual_count}/{len(sessions)} sessions flagged as unusual")
    
    print_header("Important Notes")
    print("""
  ⚠️  This is a DEMONSTRATION ONLY!
  
  To properly test on custom logs, you need:
  
  1. LOG PARSING: Use Drain/Spell to extract templates from your logs
  2. FEATURE EXTRACTION: Convert logs to the same TF-IDF features as training
  3. DOMAIN MATCHING: Model works best on logs similar to BGL (system logs)
  
  For production use:
  - Retrain the model on YOUR log data
  - Or use transfer learning to adapt to your domain
  - Implement proper log parsing pipeline
  
  This demo just shows the model CAN process text logs!
    """)


def create_sample_log_file():
    """Create a sample log file for testing"""
    sample_logs = """
2025-01-01 10:00:01 INFO System started successfully
2025-01-01 10:00:02 INFO Loading configuration from /etc/config
2025-01-01 10:00:03 INFO Database connection established
2025-01-01 10:00:04 INFO Service running on port 8080
2025-01-01 10:00:05 INFO Ready to accept connections
2025-01-01 10:05:01 WARN High memory usage detected: 85%
2025-01-01 10:05:02 WARN Garbage collection triggered
2025-01-01 10:05:03 INFO Memory usage normalized: 65%
2025-01-01 10:10:01 ERROR Connection timeout to database
2025-01-01 10:10:02 ERROR Retry attempt 1/3 failed
2025-01-01 10:10:03 ERROR Retry attempt 2/3 failed
2025-01-01 10:10:04 CRITICAL Database connection lost
2025-01-01 10:10:05 CRITICAL Entering failover mode
2025-01-01 10:10:06 INFO Failover to backup database successful
2025-01-01 10:15:01 INFO Normal operations resumed
    """.strip()
    
    with open("sample_logs.txt", 'w') as f:
        f.write(sample_logs)
    
    print("✓ Created sample_logs.txt")
    return "sample_logs.txt"


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ALogSCAN: Custom Log Testing Tool")
    print("="*70)
    
    print("\n📁 Usage:")
    print("   python test_custom_logs.py <your_log_file.txt>")
    print("\n   Or run without arguments to test with a sample file.\n")
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        print("No log file specified. Creating a sample file...\n")
        log_file = create_sample_log_file()
    
    model_dir = "experiment_records/b69e9392"
    
    if not os.path.exists(model_dir):
        print(f"\n❌ Error: Model directory '{model_dir}' not found!")
        print("Please train a model first using demo.py\n")
        sys.exit(1)
    
    test_custom_logs(log_file, model_dir, window_size=20)
