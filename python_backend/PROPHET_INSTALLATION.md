# Prophet Model Installation Guide

## Quick Install

```bash
# Activate virtual environment
cd python_backend
source .venv/bin/activate

# Install Prophet (may take 2-5 minutes)
pip install prophet>=1.1.5
```

## Verify Installation

```bash
python -c "from prophet import Prophet; print('Prophet OK')"
```

Expected output: `Prophet OK`

## If Installation Fails

Prophet requires compiled dependencies. Try these solutions:

### Option 1: Install with conda (if available)
```bash
conda install -c conda-forge prophet
```

### Option 2: Install dependencies first
```bash
# Install C++ compiler tools (macOS)
xcode-select --install

# Then install Prophet
pip install prophet
```

### Option 3: Install older version
```bash
pip install prophet==1.1.0
```

### Option 4: Skip Prophet (graceful degradation)
If Prophet won't install, the ensemble will work without it:
- Prophet is **optional** like XGBoost
- System degrades gracefully with try/except
- You'll still have 7+ other models

## Common Issues

### Issue: "No module named 'pystan'"
**Solution:**
```bash
pip install pystan==2.19.1.1
pip install prophet
```

### Issue: Takes forever to install
**Solution:**
- Be patient, first install compiles C++ code (5-10 minutes)
- Subsequent installs are faster

## Testing Prophet

After installation, test with:

```bash
cd python_backend
python models/prophet_model.py
```

Or test in ensemble:
```bash
python -c "
from models.ensemble import EnsembleModel
import numpy as np
ensemble = EnsembleModel(interval='1d')
print('Prophet available:', ensemble.prophet is not None)
"
```

## Performance Notes

- **First run:** Slow (5-10 seconds) due to compilation
- **Subsequent runs:** Fast (<1 second)
- **Training time:** 10-30 seconds per stock
- **Worth it:** +5-8% accuracy improvement

## Expected Behavior

- **With Prophet installed:** Ensemble uses 7 models total (6 base models + Prophet)
  - Base models: linear, lstm, arima, xgboost, decision_tree, svm
  - Prophet: time series specialist model
- **Without Prophet:** Ensemble uses 6 models (graceful degradation)
- **Error handling:** Prophet failures don't crash the ensemble

## Verification (All Models Working)

To verify all models are working correctly:

```bash
cd python_backend
source .venv/bin/activate
python3 -c "
from models.ensemble import EnsembleModel
ensemble = EnsembleModel(interval='1d')
print('Models in ensemble:', list(ensemble.models.keys()))
print('Prophet available:', ensemble.prophet is not None and not ensemble.prophet_failed)
print('Total models:', len(ensemble.models) + (1 if ensemble.prophet and not ensemble.prophet_failed else 0))
"
```

Expected output:
- Models in ensemble: ['linear', 'lstm', 'arima', 'xgboost', 'decision_tree', 'svm']
- Prophet available: True
- Total models: 7

