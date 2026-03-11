import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


def run_model(filepath):
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

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="#f4f7fb")
    ax.set_facecolor("#ffffff")

    ax.scatter(
        data["Date"],
        data["Close"],
        label="Actual close",
        color="#0f766e",
        s=48,
        alpha=0.9,
        edgecolors="#dff7f2",
        linewidths=0.9,
        zorder=3,
    )
    ax.plot(
        data["Date"],
        predictions,
        color="#f97316",
        linewidth=2.6,
        label="Regression forecast",
        zorder=4,
    )

    ax.set_title("Closing Price Trend and Linear Regression Forecast", fontsize=15, pad=14, color="#10233f")
    ax.set_xlabel("Date", fontsize=11, color="#44556d")
    ax.set_ylabel("Closing Price", fontsize=11, color="#44556d")
    ax.grid(True, color="#d8e1ed", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#d8e1ed",
        fontsize=10,
    )

    for spine in ax.spines.values():
        spine.set_color("#d8e1ed")

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
