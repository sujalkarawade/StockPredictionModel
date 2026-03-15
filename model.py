import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
def run_model(filepath, model_type="Linear Regression"):
    extension = os.path.splitext(filepath)[1].lower()

    if extension == ".csv":
        data = pd.read_csv(filepath)
    else:
        data = pd.read_excel(filepath)

    required_columns = {"Date", "Close"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing_list}.")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"]).sort_values("Date").copy()

    if len(data) < 2:
        raise ValueError("At least two valid rows with Date and Close values are required.")

    data["Days"] = (data["Date"] - data["Date"].min()).dt.days

    X = data[["Days"]]
    y = data["Close"]

    if model_type == "Decision Tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "Support Vector Regression (SVR)":
        model = SVR(kernel='rbf')
    elif model_type == "XGBoost":
        model = XGBRegressor(random_state=42)
    else:
        model = LinearRegression()

    model.fit(X, y)

    predictions = model.predict(X)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    ax.scatter(
        data["Date"],
        data["Close"],
        label="Actual close",
        color="#0f766e",
        s=48,
        alpha=0.9,
        edgecolors="#2a2a2a",
        linewidths=0.9,
        zorder=3,
    )
    ax.plot(
        data["Date"],
        predictions,
        color="#f97316",
        linewidth=2.6,
        label=f"{model_type} forecast",
        zorder=4,
    )

    title_model_str = model_type if model_type == "Support Vector Regression (SVR)" else getattr(model, "__class__").__name__ if model_type != "Linear Regression" else "Linear Regression"
    
    ax.set_title(f"Closing Price Trend and {model_type} Forecast", fontsize=15, pad=14, color="#e0e0e0")
    ax.set_xlabel("Date", fontsize=11, color="#a0a0a0")
    ax.set_ylabel("Closing Price", fontsize=11, color="#a0a0a0")
    ax.grid(True, color="#333333", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#1e1e1e",
        edgecolor="#333333",
        fontsize=10,
        labelcolor="#e0e0e0"
    )

    ax.tick_params(colors="#a0a0a0")

    for spine in ax.spines.values():
        spine.set_color("#333333")

    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()

    graph_path = os.path.join("static", "graph.png")
    fig.savefig(graph_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "points": len(data),
        "start_date": data["Date"].min().strftime("%d %b %Y"),
        "end_date": data["Date"].max().strftime("%d %b %Y"),
        "latest_close": f"{data['Close'].iloc[-1]:.2f}",
    }

    return graph_path, summary
