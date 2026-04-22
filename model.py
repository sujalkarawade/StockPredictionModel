import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from ml_models import (
    LinearRegressionModel,
    MultipleRegressionModel,
    LogisticRegressionModel,
    PolynomialRegressionModel,
    KNNRegressorModel,
    KMeansClusteringModel,
    IsolationForestModel,
    _load_and_prep,
    _calculate_volatility_and_changes
)

def run_model(filepath, model_type="Linear Regression", start_date=None, end_date=None):
    data = _load_and_prep(filepath, start_date, end_date)
    
    # Feature selection logic
    features = ["Days"]
    if model_type == "Multiple Regression":
        features = ["Days", "MA_20", "MA_50"]
        data_filtered = data.dropna(subset=features)
        if len(data_filtered) < 2:
            raise ValueError("Not enough data points for Multiple Regression after calculating indicators.")
        X, y = data_filtered[features], data_filtered["Close"]
        plot_dates = data_filtered["Date"]
    else:
        X, y = data[features], data["Close"]
        plot_dates = data["Date"]

    is_unsupervised = model_type in ["K-Means", "Isolation Forest"]

    if model_type == "Multiple Regression":
        model = MultipleRegressionModel()
    elif model_type == "Logistic Regression":
        model = LogisticRegressionModel()
        y_binary = (y > y.median()).astype(int)
        model.fit(X, y_binary)
        predictions = model.predict_proba(X)[:, 1] * (y.max() - y.min()) + y.min()
    elif model_type == "Polynomial":
        model = PolynomialRegressionModel(degree=2)
    elif model_type == "KNN":
        model = KNNRegressorModel(n_neighbors=5)
    elif model_type == "K-Means":
        model = KMeansClusteringModel(n_clusters=3)
    elif model_type == "Isolation Forest":
        model = IsolationForestModel(contamination=0.05)
    else:
        model = LinearRegressionModel()

    if is_unsupervised:
        X_unsup = data[["Close"]]
        model.fit(X_unsup)
        results_labels = model.predict(X_unsup)
    elif model_type != "Logistic Regression":
        model.fit(X, y)
        predictions = model.predict(X)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(13, 7), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    if is_unsupervised:
        if model_type == "K-Means":
            scatter = ax.scatter(data["Date"], data["Close"], c=results_labels, cmap="viridis", 
                                 s=48, alpha=0.9, edgecolors="#2a2a2a", linewidths=0.9, zorder=3, label="Clusters")
            legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Clusters", 
                               frameon=True, facecolor="#1e1e1e", edgecolor="#333333", labelcolor="#e0e0e0")
            ax.add_artist(legend1)
        else: # Isolation Forest
            anomalies = results_labels == -1
            ax.scatter(data.loc[~anomalies, "Date"], data.loc[~anomalies, "Close"], 
                       color="#0f766e", s=48, alpha=0.9, edgecolors="#2a2a2a", linewidths=0.9, zorder=3, label="Normal")
            ax.scatter(data.loc[anomalies, "Date"], data.loc[anomalies, "Close"], 
                       color="#ef4444", s=60, alpha=1.0, edgecolors="white", linewidths=1.2, zorder=4, label="Anomaly")
    else:
        # Plot moving averages
        ma20_mask = data["MA_20"].notna()
        ma50_mask = data["MA_50"].notna()
        ma200_mask = data["MA_200"].notna()
        
        ax.plot(data.loc[ma20_mask, "Date"], data.loc[ma20_mask, "MA_20"], 
                color="#60a5fa", linewidth=1.5, label="MA 20", alpha=0.8, zorder=2)
        ax.plot(data.loc[ma50_mask, "Date"], data.loc[ma50_mask, "MA_50"], 
                color="#fbbf24", linewidth=1.5, label="MA 50", alpha=0.8, zorder=2)
        ax.plot(data.loc[ma200_mask, "Date"], data.loc[ma200_mask, "MA_200"], 
                color="#f87171", linewidth=1.5, label="MA 200", alpha=0.8, zorder=2)

        ax.scatter(data["Date"], data["Close"], label="Actual close",
                   color="#0f766e", s=48, alpha=0.9, edgecolors="#2a2a2a", linewidths=0.9, zorder=3)
        ax.plot(plot_dates, predictions, color="#f97316", linewidth=2.6,
                label=f"{model_type} forecast", zorder=4)

    title_suffix = "Analysis" if is_unsupervised else "Forecast"
    ax.set_title(f"Closing Price Trend and {model_type} {title_suffix}", 
                 fontsize=15, pad=14, color="#e0e0e0")
    ax.set_xlabel("Date", fontsize=11, color="#a0a0a0")
    ax.set_ylabel("Closing Price", fontsize=11, color="#a0a0a0")
    ax.grid(True, color="#333333", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.legend(loc="upper left", frameon=True, facecolor="#1e1e1e",
              edgecolor="#333333", fontsize=9, labelcolor="#e0e0e0", ncol=2)
    ax.tick_params(colors="#a0a0a0")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()

    graph_path = os.path.join("static", "graph.png")
    fig.savefig(graph_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    metrics = _calculate_volatility_and_changes(data)

    summary = {
        "points": len(data),
        "start_date": data["Date"].min().strftime("%d %b %Y"),
        "end_date": data["Date"].max().strftime("%d %b %Y"),
        "latest_close": f"{data['Close'].iloc[-1]:.2f}",
        "avg_daily_change": metrics["avg_daily_change"],
        "7day_change": metrics["7day_change_%"],
        "30day_change": metrics["30day_change_%"],
        "overall_change": metrics["overall_change_%"],
    }
    return graph_path, summary

def run_all_models(filepath):
    data = _load_and_prep(filepath)
    X, y = data[["Days"]], data["Close"]

    models = {
        "Linear Regression": LinearRegressionModel(),
        "Multiple Regression": MultipleRegressionModel(),
        "Polynomial": PolynomialRegressionModel(degree=2),
        "KNN": KNNRegressorModel(n_neighbors=5),
    }
    colors = {
        "Linear Regression": "#3b82f6",
        "Multiple Regression": "#f97316",
        "Polynomial": "#22c55e",
        "KNN": "#a855f7",
    }

    results = {}
    predictions_map = {}

    for name, m in models.items():
        if name == "Multiple Regression":
            features = ["Days", "MA_20", "MA_50"]
            data_m = data.dropna(subset=features)
            X_m, y_m = data_m[features], data_m["Close"]
            m.fit(X_m, y_m)
            preds = m.predict(X_m)
            full_preds = np.full(len(data), np.nan)
            full_preds[data.index.get_indexer(data_m.index)] = preds
            predictions_map[name] = full_preds
            r2 = r2_score(y_m, preds)
            mae = mean_absolute_error(y_m, preds)
            rmse = np.sqrt(mean_squared_error(y_m, preds))
        else:
            m.fit(X, y)
            preds = m.predict(X)
            predictions_map[name] = preds
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            
        results[name] = {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}

    plt.style.use("seaborn-v0_8-darkgrid")
    fig1, ax1 = plt.subplots(figsize=(13, 6.5), facecolor="#1e1e1e")
    ax1.set_facecolor("#1e1e1e")

    ax1.scatter(data["Date"], data["Close"], label="Actual close",
                color="#0f766e", s=28, alpha=0.7, edgecolors="#2a2a2a", linewidths=0.6, zorder=3)
    for name, preds in predictions_map.items():
        mask = ~np.isnan(preds)
        ax1.plot(data["Date"][mask], preds[mask], color=colors[name], linewidth=2.2, label=name, zorder=4)

    ax1.set_title("All Models — Forecast Overlay", fontsize=15, pad=14, color="#e0e0e0")
    ax1.set_xlabel("Date", fontsize=11, color="#a0a0a0")
    ax1.set_ylabel("Closing Price", fontsize=11, color="#a0a0a0")
    ax1.grid(True, color="#333333", linestyle="--", linewidth=0.8, alpha=0.9)
    ax1.legend(loc="upper left", frameon=True, facecolor="#1e1e1e",
               edgecolor="#333333", fontsize=10, labelcolor="#e0e0e0")
    ax1.tick_params(colors="#a0a0a0")
    for spine in ax1.spines.values():
        spine.set_color("#333333")
    fig1.autofmt_xdate(rotation=35)
    fig1.tight_layout()
    overlay_path = os.path.join("static", "compare_overlay.png")
    fig1.savefig(overlay_path, dpi=160, bbox_inches="tight")
    plt.close(fig1)

    names = list(results.keys())
    r2_vals   = [results[n]["r2"]   for n in names]
    mae_vals  = [results[n]["mae"]  for n in names]
    rmse_vals = [results[n]["rmse"] for n in names]
    bar_colors = [colors[n] for n in names]

    fig2, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#1e1e1e")
    fig2.suptitle("Model Accuracy Comparison", fontsize=15, color="#e0e0e0", y=1.01)

    metrics = [
        (axes[0], r2_vals,   "R² Score (higher = better)",  True),
        (axes[1], mae_vals,  "MAE (lower = better)",         False),
        (axes[2], rmse_vals, "RMSE (lower = better)",        False),
    ]

    for ax, vals, title, higher_better in metrics:
        ax.set_facecolor("#1e1e1e")
        bars = ax.bar(names, vals, color=bar_colors, width=0.55, zorder=3)
        ax.set_title(title, fontsize=11, color="#e0e0e0", pad=10)
        ax.tick_params(colors="#a0a0a0", labelsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8, color="#a0a0a0")
        ax.grid(axis="y", color="#333333", linestyle="--", linewidth=0.8, alpha=0.9, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#e0e0e0")
        best_idx = vals.index(max(vals) if higher_better else min(vals))
        bars[best_idx].set_edgecolor("#ffffff")
        bars[best_idx].set_linewidth(2)

    fig2.patch.set_facecolor("#1e1e1e")
    fig2.tight_layout()
    accuracy_path = os.path.join("static", "compare_accuracy.png")
    fig2.savefig(accuracy_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "points":     len(data),
        "start_date": data["Date"].min().strftime("%d %b %Y"),
        "end_date":   data["Date"].max().strftime("%d %b %Y"),
        "latest_close": f"{data['Close'].iloc[-1]:.2f}",
    }

    return overlay_path, accuracy_path, results, summary
