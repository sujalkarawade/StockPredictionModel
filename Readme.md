# 📈 Stock Price Prediction

A Flask web app for uploading stock data and visualizing closing price trends using multiple ML models — with a multi-model comparison dashboard.

---

## Features

- Upload stock data as CSV, XLS, or XLSX
- Choose from 4 ML models via a custom styled dropdown
- View Actual vs Predicted chart with a data summary panel
- Multi-Model Comparison Dashboard — runs all 4 models at once, shows forecast overlay and accuracy metrics (R², MAE, RMSE)
- Light / Dark theme toggle that persists across pages (respects OS preference on first visit)
- Auto-opens browser on `python app.py`

---

## ML Models

| Model | Notes |
|---|---|
| Linear Regression | Fast baseline, straight-line trend |
| Decision Tree Regressor | Non-linear, fits local patterns |
| Random Forest Regressor | Ensemble, robust to noise |
| XGBoost Regressor | Gradient boosting, high accuracy |

---

## Input Format

File must contain at least these two columns:

| Date | Close |
|---|---|
| 2024-01-01 | 150.00 |
| 2024-01-02 | 152.50 |

Extra columns are ignored.

---

## Project Structure

```
StockPredictionFlask/
├── app.py                  # Flask routes (/, /compare)
├── model.py                # Data prep, ML training, chart generation
├── uploads/                # Uploaded files (auto-created)
├── static/
│   ├── graph.png           # Single-model chart output
│   ├── compare_overlay.png # Multi-model overlay chart
│   ├── compare_accuracy.png# Accuracy bar chart
│   ├── styles_index.css
│   └── styles_result.css
└── templates/
    ├── index.html          # Upload page
    ├── result.html         # Single model result page
    └── compare.html        # Multi-model comparison dashboard
```

---

## Installation

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
pip install flask pandas scikit-learn matplotlib openpyxl xgboost werkzeug numpy
```

---

## Running

```bash
python app.py
```

Browser opens automatically at `http://127.0.0.1:5000`.

---

## Technologies

| | |
|---|---|
| Python + Flask | Backend and routing |
| Pandas + NumPy | Data loading and preprocessing |
| Scikit-learn | Linear Regression, Decision Tree, Random Forest, metrics |
| XGBoost | XGBoost Regressor |
| Matplotlib | Chart generation |
| HTML / CSS / JS | Frontend UI with theme support |

---

## Author

**Sujal Karawade** — Engineering Student

---

## License

Open-source, free to use for learning and educational purposes.
