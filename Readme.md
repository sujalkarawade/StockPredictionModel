# 📈 Stock Price Prediction using Linear Regression

## 📌 Project Overview

This project is a **Machine Learning web application** built using **Python and Flask** that predicts stock price trends using the **Linear Regression algorithm**.

Users can upload a **stock dataset in Excel format (.xlsx)**, and the application will:

* Train a Linear Regression model
* Predict stock price trends
* Display a graph comparing **actual vs predicted prices**

This project demonstrates the integration of **Machine Learning with a web interface**.

---

## 🚀 Features

* Upload stock dataset in **Excel format**
* Machine Learning model using **Linear Regression**
* Automatic **data preprocessing**
* Visualization of **Actual vs Predicted stock prices**
* Web interface built using **Flask**

---

## 🧠 Machine Learning Model

The prediction is based on **Linear Regression**, which models the relationship between time and stock price.

Regression equation:

y = mx + b

Where:

* **x** → Independent variable (time in days)
* **y** → Predicted stock price
* **m** → Slope of the line
* **b** → Intercept

---

## 🛠 Technologies Used

| Technology   | Purpose                |
| ------------ | ---------------------- |
| Python       | Programming language   |
| Flask        | Web framework          |
| Pandas       | Data processing        |
| Scikit-learn | Machine learning model |
| Matplotlib   | Graph visualization    |
| OpenPyXL     | Reading Excel files    |
| HTML         | Frontend interface     |

---

## 📂 Project Structure

```
StockPredictionFlask
│
├── app.py
├── model.py
├── uploads
├── static
│   └── graph.png
└── templates
    └── index.html
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/stock-price-prediction.git
```

### 2️⃣ Navigate to project folder

```
cd stock-price-prediction
```

### 3️⃣ Install dependencies

```
pip install flask pandas scikit-learn matplotlib openpyxl
```

---

## ▶️ Running the Application

Run the Flask application:

```
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

Upload an Excel file containing stock data.

---

## 📊 Input Dataset Format

The Excel file should contain the following columns:

| Date       | Close |
| ---------- | ----- |
| 2024-01-01 | 150   |
| 2024-01-02 | 152   |
| 2024-01-03 | 153   |

---

## 📉 Output

The application generates a graph showing:

* **Actual stock prices**
* **Predicted trend line**

---

## 🔮 Future Improvements

Possible enhancements:

* Add **Random Forest and XGBoost models**
* Predict **future stock prices**
* Use **real-time stock data from APIs**
* Create **interactive graphs using Plotly**
* Deploy the application on **cloud platforms**

---

## 👨‍💻 Author

**Sujal Karawade**
Engineering Student

---

## 📜 License

This project is open-source and available for learning and educational purposes.
