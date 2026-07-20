"""
visualization.py
Menghasilkan plot matplotlib/seaborn sebagai string Base64 PNG.
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend non-interaktif untuk server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.pipeline import Pipeline

from src.preprocessing import FITUR_TERPILIH

# Warna konsisten
WARNA_SEHAT       = "#2ecc71"
WARNA_TIDAK_SEHAT = "#e74c3c"
WARNA_PALETTE     = [WARNA_TIDAK_SEHAT, WARNA_SEHAT]

LABEL_TEKS = {0: "Tidak Sehat", 1: "Sehat"}


def fig_to_base64(fig: plt.Figure) -> str:
    """Helper: konversi matplotlib Figure ke data URL base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def plot_class_distribution(y: pd.Series) -> str:
    """Pie chart distribusi Sehat vs Tidak Sehat → base64 PNG."""
    counts = y.value_counts().sort_index()
    labels = [LABEL_TEKS.get(idx, str(idx)) for idx in counts.index]
    colors = [WARNA_PALETTE[idx] for idx in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
        autotext.set_color("white")
    ax.set_title("Distribusi Kelas Kondisi Keuangan UMKM", fontsize=14, fontweight="bold", pad=15)
    return fig_to_base64(fig)


def plot_correlation_heatmap(X: pd.DataFrame) -> str:
    """Heatmap korelasi 6 fitur → base64 PNG."""
    # Rename kolom agar label lebih pendek
    label_map = {
        "Monthly_Revenue":        "Revenue",
        "Net_Profit_Margin":      "NPM (%)",
        "Burn_Rate_Ratio":        "Burn Rate",
        "Transaction_Count":      "Transaksi",
        "Repeat_Order_Rate":      "Repeat Order",
        "Business_Tenure_Months": "Tenure",
    }
    X_plot = X.rename(columns=label_map)
    corr   = X_plot.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Heatmap Korelasi Antar Fitur", fontsize=14, fontweight="bold", pad=15)
    return fig_to_base64(fig)


def plot_kde_per_feature(X: pd.DataFrame, y: pd.Series) -> str:
    """KDE plot per fitur, dipisah per kelas → base64 PNG."""
    n_fitur = len(FITUR_TERPILIH)
    n_cols  = 3
    n_rows  = (n_fitur + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    label_friendly = {
        "Monthly_Revenue":        "Pendapatan Bulanan (Rp)",
        "Net_Profit_Margin":      "Net Profit Margin (%)",
        "Burn_Rate_Ratio":        "Burn Rate Ratio",
        "Transaction_Count":      "Jumlah Transaksi",
        "Repeat_Order_Rate":      "Repeat Order Rate (%)",
        "Business_Tenure_Months": "Lama Usaha (bulan)",
    }

    for i, fitur in enumerate(FITUR_TERPILIH):
        ax = axes[i]
        for kelas, warna in [(0, WARNA_TIDAK_SEHAT), (1, WARNA_SEHAT)]:
            data_kelas = X.loc[y == kelas, fitur]
            sns.kdeplot(data=data_kelas, ax=ax, color=warna,
                        fill=True, alpha=0.4, label=LABEL_TEKS[kelas])
        ax.set_title(label_friendly.get(fitur, fitur), fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Sembunyikan subplot kosong
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribusi Fitur per Kelas Kondisi Keuangan", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_confusion_matrix(cm: np.ndarray) -> str:
    """Confusion matrix heatmap dengan label TN/FP/FN/TP → base64 PNG."""
    labels = ["Tidak Sehat (0)", "Sehat (1)"]
    annot  = np.array([
        [f"TN\n{cm[0,0]:,}", f"FP\n{cm[0,1]:,}"],
        [f"FN\n{cm[1,0]:,}", f"TP\n{cm[1,1]:,}"],
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_xlabel("Prediksi", fontsize=12, labelpad=10)
    ax.set_ylabel("Aktual",   fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    return fig_to_base64(fig)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> str:
    """ROC Curve dengan nilai AUC di legenda → base64 PNG."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2,
            label=f"Gaussian NB (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2980b9")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Gaussian Naïve Bayes", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def plot_feature_importance(pipeline: Pipeline) -> str:
    """
    Bar chart feature importance GaussianNB berdasarkan Fisher-like score:
    importance[i] = |mean_sehat[i] - mean_tidak_sehat[i]| / sqrt(avg_var[i])
    → base64 PNG
    """
    gnb    = pipeline.named_steps["gnb"]
    # theta_.shape == (2, 6): [kelas][fitur]
    # var_.shape   == (2, 6)
    delta_mean = np.abs(gnb.theta_[1] - gnb.theta_[0])
    avg_var    = (gnb.var_[1] + gnb.var_[0]) / 2.0
    importance = delta_mean / np.sqrt(avg_var + 1e-10)

    # Normalisasi ke [0, 1]
    max_imp    = importance.max()
    if max_imp > 0:
        importance = importance / max_imp

    label_friendly = {
        "Monthly_Revenue":        "Pendapatan\nBulanan",
        "Net_Profit_Margin":      "Net Profit\nMargin",
        "Burn_Rate_Ratio":        "Burn Rate\nRatio",
        "Transaction_Count":      "Jumlah\nTransaksi",
        "Repeat_Order_Rate":      "Repeat Order\nRate",
        "Business_Tenure_Months": "Lama\nUsaha",
    }
    labels = [label_friendly.get(f, f) for f in FITUR_TERPILIH]
    scores = importance.tolist()

    # Urutkan dari tertinggi
    sorted_pairs = sorted(zip(scores, labels), reverse=True)
    scores_sorted, labels_sorted = zip(*sorted_pairs)

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(scores_sorted)))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels_sorted, scores_sorted, color=colors, edgecolor="white")
    for bar, val in zip(bars, scores_sorted):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Skor Kepentingan (Normalized Fisher Score)", fontsize=11)
    ax.set_title("Feature Importance — Gaussian Naïve Bayes", fontsize=14, fontweight="bold", pad=15)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_cross_val_scores(scores: list) -> str:
    """Bar chart akurasi tiap fold CV → base64 PNG."""
    n     = len(scores)
    folds = [f"Fold {i+1}" for i in range(n)]
    mean  = np.mean(scores)

    colors = [WARNA_SEHAT if s >= mean else "#f39c12" for s in scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(folds, scores, color=colors, edgecolor="white", width=0.6)
    ax.axhline(mean, color="#2980b9", linestyle="--", linewidth=2,
               label=f"Rata-rata: {mean:.4f}")
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Akurasi", fontsize=12)
    ax.set_title("Akurasi per Fold — 10-Fold Stratified Cross Validation",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)
