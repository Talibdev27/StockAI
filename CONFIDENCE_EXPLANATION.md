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

## Bug Fix Needed

The frontend is multiplying confidence by 100 again:
- Backend returns: `79.0` (meaning 79%)
- Frontend displays: `79.0 * 100 = 7900%` ❌

**Fix:** Remove `* 100` in frontend since confidence is already a percentage.

