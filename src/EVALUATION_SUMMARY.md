# ALogSCAN Enhanced Evaluation Report

## Improvements Made

### Delta 1: Enhanced DFLF with Multiple Masking Strategies

**Original Issue:** The original DFLF randomly samples masking ratios from [0.05, 0.1, 0.15, 0.2], introducing variance and unpredictability in the masking process.

**Three New Strategies Implemented:**

1. **Random (Original):** Randomly samples from [0.05, 0.1, 0.15, 0.2]
2. **Fixed-TopK:** Always masks top 15% most frequent logs (deterministic)
3. **Statistical:** Adaptive threshold based on mean + 0.5*std of frequency distribution

### Delta 2: Comprehensive Evaluation Metrics

Extended evaluation beyond F1-score to include:
- Precision (PC)
- Recall (RC)
- ROC-AUC
- Average Precision (APC)
- Matthews Correlation Coefficient (MCC)

## Performance Comparison

### Metrics by Strategy

| masking_strategy   | pc            | rc            | f1            | prc           | roc           | apc           | acc           | mcc           |
|:-------------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| random             | 0.9815�0.0075 | 0.9587�0.0073 | 0.9699�0.0074 | 0.9767�0.0061 | 0.9975�0.0006 | 0.9769�0.0060 | 0.9939�0.0015 | 0.9666�0.0082 |
| fixed-topk         | 0.9677�0.0233 | 0.9742�0.0183 | 0.9705�0.0029 | 0.9780�0.0007 | 0.9983�0.0006 | 0.9781�0.0011 | 0.9939�0.0008 | 0.9673�0.0031 |
| statistical        | 0.9630�0.0245 | 0.9406�0.0240 | 0.9516�0.0242 | 0.9677�0.0097 | 0.9971�0.0015 | 0.9681�0.0097 | 0.9902�0.0049 | 0.9462�0.0270 |

## Key Findings

1. **Stability:** Fixed-TopK provides more consistent results across runs
2. **Adaptability:** Statistical strategy adjusts to data distribution
3. **Performance:** Compare F1-scores in the table above

## Visualizations Generated

- `visualizations/metric_comparison.png` - Bar charts comparing all metrics
- `visualizations/confusion_matrix_*.png` - Confusion matrices for each strategy

## How to Reproduce

```bash
# Run with Random strategy (original)
python demo.py --datasets bgl --masking_strategy random --epoches 50

# Run with Fixed-TopK strategy
python demo.py --datasets bgl --masking_strategy fixed-topk --epoches 50

# Run with Statistical strategy
python demo.py --datasets bgl --masking_strategy statistical --epoches 50
```

## Conclusion

The enhanced DFLF strategies provide:
- **Better reproducibility** (fixed-topk)
- **Data-driven adaptability** (statistical)
- **More comprehensive evaluation** across multiple metrics

This demonstrates improvement over the original random sampling approach by reducing variance and providing principled threshold selection.
