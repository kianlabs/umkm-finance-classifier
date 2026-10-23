"""
visualization.py
Menghasilkan plot matplotlib/seaborn sebagai string Base64 PNG.
Versi multiclass: 4 kelas (Critical, Elite, Growth, Struggling), 5 fitur SBS.
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend non-interaktif untuk server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

from src.preprocessing import FITUR_TERPILIH, KELAS, LABEL_TEKS

# Warna konsisten per kelas
WARNA_KELAS = {
    "Critical":   "#e74c3c",
    "Elite":      "#8e44ad",
    "Growth":     "#2ecc71",
    "Struggling": "#f39c12",
}
WARNA_PALETTE = [WARNA_KELAS[k] for k in KELAS]

LABEL_FRIENDLY = {
    "Net_Profit_Margin":      "Net Profit Margin (%)",
    "Burn_Rate_Ratio":        "Burn Rate Ratio",
    "Avg_Historical_Rating":  "Rating Historis",
    "Business_Tenure_Months": "Lama Usaha (bulan)",
    "Repeat_Order_Rate":      "Repeat Order Rate (%)",
}
LABEL_SHORT = {
    "Net_Profit_Margin":      "NPM (%)",
    "Burn_Rate_Ratio":        "Burn Rate",
    "Avg_Historical_Rating":  "Rating",
    "Business_Tenure_Months": "Tenure",
    "Repeat_Order_Rate":      "Repeat Order",
}


def fig_to_base64(fig: plt.Figure) -> str:
    """Helper: konversi matplotlib Figure ke data URL base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def plot_class_distribution(y: pd.Series) -> str:
    """Pie chart distribusi 4 kelas → base64 PNG."""
    counts = y.value_counts().sort_index()
    labels = [LABEL_TEKS.get(int(idx), str(idx)) for idx in counts.index]
    colors = [WARNA_KELAS.get(LABEL_TEKS.get(int(idx), ""), "#999999") for idx in counts.index]

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
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
        autotext.set_color("white")
    ax.set_title("Distribusi Kelas Kondisi Keuangan UMKM (4 Kelas)", fontsize=14, fontweight="bold", pad=15)
    return fig_to_base64(fig)


def plot_correlation_heatmap(X: pd.DataFrame) -> str:
    """Heatmap korelasi 5 fitur SBS → base64 PNG."""
    X_plot = X.rename(columns=LABEL_SHORT)
    corr = X_plot.corr()

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
    ax.set_title("Heatmap Korelasi Antar Fitur (SBS)", fontsize=14, fontweight="bold", pad=15)
    return fig_to_base64(fig)


def plot_kde_per_feature(X: pd.DataFrame, y: pd.Series) -> str:
    """KDE plot per fitur, dipisah per kelas (4) → base64 PNG."""
    n_fitur = len(FITUR_TERPILIH)
    n_cols = 3
    n_rows = (n_fitur + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, fitur in enumerate(FITUR_TERPILIH):
        ax = axes[i]
        for kode, nama in LABEL_TEKS.items():
            data_kelas = X.loc[y == kode, fitur]
            if len(data_kelas) > 1:
                sns.kdeplot(data=data_kelas, ax=ax, color=WARNA_KELAS[nama],
                            fill=True, alpha=0.3, label=nama)
        ax.set_title(LABEL_FRIENDLY.get(fitur, fitur), fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    for j in range(n_fitur, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribusi Fitur per Kelas Kondisi Keuangan", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_confusion_matrix(cm: np.ndarray) -> str:
    """Confusion matrix 4x4 heatmap → base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=KELAS,
        yticklabels=KELAS,
        linewidths=1,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_xlabel("Prediksi", fontsize=12, labelpad=10)
    ax.set_ylabel("Aktual", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix (4 Kelas)", fontsize=14, fontweight="bold", pad=15)
    return fig_to_base64(fig)


def plot_roc_curves(roc_data: dict) -> str:
    """
    ROC Curve One-vs-Rest multiclass → base64 PNG.
    roc_data: {nama_kelas: {"fpr": [...], "tpr": [...], "auc": float}}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for nama in KELAS:
        d = roc_data.get(nama)
        if not d:
            continue
        ax.plot(d["fpr"], d["tpr"], color=WARNA_KELAS[nama], lw=2,
                label=f"{nama} (AUC = {d['auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve One-vs-Rest — Gaussian Naïve Bayes", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def plot_roc_curve(fpr, tpr, auc_score: float) -> str:
    """Legacy biner — dipertahankan untuk kompatibilitas."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"Gaussian NB (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Gaussian Naïve Bayes", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def plot_feature_importance(pipeline: Pipeline) -> str:
    """
    Bar chart feature importance GaussianNB multiclass berdasarkan
    Fisher-like score: simpangan baku mean antar-kelas / sqrt(rata-rata varians dalam kelas).
    → base64 PNG
    """
    gnb = pipeline.named_steps["gnb"]
    # theta_.shape == (4, 5), var_.shape == (4, 5)
    between = np.std(gnb.theta_, axis=0)
    within = np.sqrt(np.mean(gnb.var_, axis=0))
    importance = between / (within + 1e-10)

    max_imp = importance.max()
    if max_imp > 0:
        importance = importance / max_imp

    labels = [LABEL_FRIENDLY.get(f, f) for f in FITUR_TERPILIH]
    scores = importance.tolist()

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
    ax.set_title("Feature Importance — Gaussian Naïve Bayes (Multiclass)", fontsize=14, fontweight="bold", pad=15)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_cross_val_scores(scores: list) -> str:
    """Bar chart akurasi tiap fold CV → base64 PNG."""
    n = len(scores)
    folds = [f"Fold {i+1}" for i in range(n)]
    mean = np.mean(scores)

    colors = ["#2ecc71" if s >= mean else "#f39c12" for s in scores]

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
