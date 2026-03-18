# 📈 Stock Price Prediction

A Flask web app that lets you upload stock data and visualize closing price trends using multiple machine learning models.

---

## Features

- Upload stock data as **CSV, XLS, or XLSX**
- Choose from **4 ML models** via a custom styled dropdown
- View an **Actual vs Predicted** chart with a summary panel
- **Light / Dark theme** toggle that persists across sessions (respects OS preference on first visit)

---

## ML Models

| Model | Key trait |
|---|---|
| Linear Regression | Fast baseline, straight-line trend |
| Decision Tree Regressor | Non-linear, fits local patterns |
| Random Forest Regressor | Ensemble, more robust to noise |
| XGBoost Regressor | Gradient boosting, high accuracy |

---

## Input Format

Your file must contain at least these two columns:

| Date | Close |
|---|---|
| 2024-01-01 | 150.00 |
| 2024-01-02 | 152.50 |

Any extra columns are ignored.

---

## Project Structure

```
StockPredictionFlask/
├── app.py                  # Flask routes
├── model.py                # Data processing + ML + chart generation
├── uploads/                # Uploaded files (auto-created)
├── static/
│   ├── graph.png           # Generated chart output
│   ├── styles_index.css
│   └── styles_result.css
└── templates/
    ├── index.html          # Upload page
    └── result.html         # Results page
```

---

## Installation

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
pip install flask pandas scikit-learn matplotlib openpyxl xgboost werkzeug
```

---

## Running

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## Technologies

| | |
|---|---|
| Python + Flask | Backend and routing |
| Pandas | Data loading and preprocessing |
| Scikit-learn | Linear Regression, Decision Tree, Random Forest |
| XGBoost | XGBoost Regressor |
| Matplotlib | Chart generation |
| HTML / CSS / JS | Frontend UI with theme support |

---

## Author

**Sujal Karawade** — Engineering Student

---

## License

Open-source, free to use for learning and educational purposes.
