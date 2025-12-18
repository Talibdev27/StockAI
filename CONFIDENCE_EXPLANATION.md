# Confidence Calculation Explanation

## How Confidence is Calculated

### Individual Model Confidences (0-1 range)

Each model calculates its own confidence score during training:

1. **Linear Regression**: Uses R² score (coefficient of determination)
   - Formula: `confidence = R²` (0-1 range)
   - Higher R² = better fit = higher confidence

2. **LSTM**: Uses validation RMSE (Root Mean Squared Error)
   - Formula: `confidence = 1.0 / (1.0 + val_rmse)`
   - Lower RMSE = higher confidence (0-1 range)

3. **ARIMA**: Uses AIC (Akaike Information Criterion)
   - Formula: `confidence = 1.0 / (1.0 + abs(aic) / 1000)`
   - Lower AIC = better model = higher confidence (0-1 range)

4. **XGBoost/Decision Tree/SVM**: Similar to LSTM
   - Formula: `confidence = 1.0 / (1.0 + rmse)`
   - Lower RMSE = higher confidence (0-1 range)

5. **Prophet**: Uses test error
   - Formula: `confidence = 1 - normalized_error`
   - Lower error = higher confidence (0-1 range)

### Ensemble Confidence (Weighted Average)

The ensemble combines individual model confidences:

```python
ensemble_confidence = sum(
    individual_confidence[model] * weight[model]
    for each model
)
```

Where:
- `individual_confidence[model]` = confidence from that model (0-1)
- `weight[model]` = weight assigned to that model (0-1, sums to 1.0)

### Final Output

The backend multiplies by 100 to convert to percentage:
- `confidence = ensemble_confidence * 100` → Returns 0-100%

**Example:**
- Linear: 0.85 (85%)
- LSTM: 0.99 (99%)
- ARIMA: 0.50 (50%)
- Ensemble (equal weights): (0.85 + 0.99 + 0.50) / 3 = 0.78 → **78%**

## Direction vs. Price: Balancing Objectives

The system now explicitly balances **price accuracy** and **directional accuracy**:

- LSTM uses a **dual-head architecture**:
  - `price` head (regression, MSE loss).
  - `direction` head (Up vs non-Up, binary cross-entropy loss).
  - Combined loss: `0.6 * direction + 0.4 * price` to emphasize correct sign.
- The ensemble weighting gives **more weight** to models with better:
  - Directional hit rate (especially for 1d interval).
  - Normalized RMSE.

This helps reduce the previous mean-reversion bias where daily predictions
were systematically a few percent below price.

## Post-Processing Guardrails for 1d Predictions

For **daily (1d) predictions**, the backend adds a conservative filter:

- If the ensemble predicts **Down** with **confidence ≥ 75%**, and
- The current price is within about **±2% of the 5‑day moving average**,

then the system:

1. Shrinks the predicted move toward the current price.
2. Often downgrades the direction to **Neutral**.
3. Lowers the reported confidence.

The API exposes this in the `postProcessing` field:

- `rawPredictedPrice`, `rawConfidence`
- `predictedDirectionRaw`, `predictedDirectionAdjusted`
- `adjusted` (boolean), `adjustmentReason`

These values are useful for debugging and for explaining why some overly
bearish signals are softened in the final output.

