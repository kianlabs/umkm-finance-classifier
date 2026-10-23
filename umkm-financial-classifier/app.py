"""
app.py
Entry point Flask Application — Sistem Klasifikasi Kondisi Keuangan UMKM
Jalankan: python app.py  →  http://localhost:5000
"""

import os
import sys
import json
import uuid
import traceback
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify, redirect, url_for

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import validate_input, FITUR_TERPILIH, KELAS
from src.model import load_pipeline, predict_single
from src.visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_kde_per_feature,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_cross_val_scores,
)

# ── Konfigurasi ──────────────────────────────────────────────────────────────

app = Flask(__name__)

MODEL_PATH   = os.path.join("models",  "gnb_pipeline.pkl")
METRICS_PATH = os.path.join("models",  "metrics.json")
HISTORY_PATH = os.path.join("data",    "prediction_history.json")
DATASET_PATH = os.path.join("data",    "raw", "dataset_umkm.csv")

# Cache model (dimuat sekali saat startup)
_pipeline = None


def get_pipeline():
    """Muat pipeline sekali dan cache ke variabel global."""
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline(MODEL_PATH)
    return _pipeline


def load_metrics() -> dict:
    """Muat metrik dari file JSON yang dihasilkan train.py."""
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def load_history() -> list:
    """Muat riwayat prediksi dari JSON."""
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def append_history(input_data: dict, output_data: dict) -> None:
    """Tambahkan satu prediksi ke riwayat."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    history = load_history()
    entry = {
        "id":        str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "input":     {k: input_data.get(k) for k in FITUR_TERPILIH},
        "output":    output_data,
    }
    history.append(entry)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ── Halaman HTML ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/klasifikasi")
def klasifikasi():
    return render_template("klasifikasi.html")


@app.route("/visualisasi")
def visualisasi():
    return render_template("visualisasi.html")


@app.route("/riwayat")
def riwayat():
    return render_template("riwayat.html")


# ── REST API ─────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Terima JSON body dengan 5 fitur SBS, kembalikan prediksi 4 kelas dan rekomendasi.
    """
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "errors": ["Request body harus JSON."]}), 400

    is_valid, errors = validate_input(data)
    if not is_valid:
        return jsonify({"success": False, "errors": errors}), 400

    try:
        result = predict_single(pipeline, data)
        result["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        append_history(data, result)
        return jsonify({"success": True, **result})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": "Terjadi kesalahan saat prediksi."}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """
    Kembalikan metrik model yang disimpan saat training.
    """
    metrics = load_metrics()
    if not metrics:
        return jsonify({
            "success": False,
            "error": "Metrik belum tersedia. Jalankan python train.py terlebih dahulu."
        }), 503

    return jsonify({
        "success":           True,
        "accuracy":          metrics.get("accuracy"),
        "precision":         metrics.get("precision_macro", metrics.get("precision")),
        "precision_macro":   metrics.get("precision_macro", metrics.get("precision")),
        "recall":            metrics.get("recall_macro", metrics.get("recall")),
        "recall_macro":      metrics.get("recall_macro", metrics.get("recall")),
        "f1":                metrics.get("f1_macro", metrics.get("f1")),
        "f1_macro":          metrics.get("f1_macro", metrics.get("f1")),
        "f1_weighted":       metrics.get("f1_weighted"),
        "roc_auc":           metrics.get("roc_auc"),
        "roc_auc_per_class": metrics.get("roc_auc_per_class", {}),
        "per_class":         metrics.get("per_class", {}),
        "classes":           metrics.get("classes", KELAS),
        "cv_mean_acc":       metrics.get("mean_acc"),
        "cv_std_acc":        metrics.get("std_acc"),
        "cv_mean_f1":        metrics.get("mean_f1"),
        "cv_std_f1":         metrics.get("std_f1"),
        "cv_scores":         metrics.get("scores", []),
        "total_train":       metrics.get("total_train"),
        "total_test":        metrics.get("total_test"),
        "training_time":     metrics.get("training_time_seconds"),
    })


@app.route("/api/visualizations", methods=["GET"])
def api_visualizations():
    """
    Generate semua plot dan kembalikan sebagai base64 PNG.
    """
    import pandas as pd
    import numpy as np

    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503

    metrics = load_metrics()
    if not metrics:
        return jsonify({
            "success": False,
            "error": "Metrik belum tersedia. Jalankan python train.py terlebih dahulu."
        }), 503

    try:
        # Muat dataset untuk EDA
        from src.preprocessing import load_dataset, select_features, split_data
        df = load_dataset(DATASET_PATH)
        X, y = select_features(df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        cm = np.array(metrics["confusion_matrix"])
        sample_idx = X_train.sample(min(5000, len(X_train)), random_state=42).index

        viz = {
            "success":          True,
            "class_dist":       plot_class_distribution(y),
            "correlation":      plot_correlation_heatmap(X),
            "kde":              plot_kde_per_feature(X_train.loc[sample_idx],
                                                     y_train.loc[sample_idx]),
            "confusion_matrix": plot_confusion_matrix(cm),
            "roc_curve":        plot_roc_curves(metrics["roc_curves"]),
            "feature_importance": plot_feature_importance(pipeline),
            "cv_scores":        plot_cross_val_scores(metrics["scores"]),
        }
        return jsonify(viz)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Kembalikan riwayat prediksi.
    Query params: page (default 1), per_page (default 20)
    """
    history = load_history()
    # Urutkan terbaru di atas
    history = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)

    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    start    = (page - 1) * per_page
    end      = start + per_page

    return jsonify({
        "success":     True,
        "predictions": history[start:end],
        "total":       len(history),
        "page":        page,
        "per_page":    per_page,
        "total_pages": (len(history) + per_page - 1) // per_page,
    })


@app.route("/api/history", methods=["DELETE"])
def api_history_clear():
    """Hapus semua riwayat prediksi."""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w") as f:
            json.dump([], f)
    return jsonify({"success": True, "message": "Riwayat prediksi dihapus."})


@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Trigger ulang training pipeline secara asinkron (endpoint sederhana).
    """
    import subprocess
    import threading
    import time

    global _pipeline

    def run_training():
        global _pipeline
        subprocess.run([sys.executable, "train.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
        _pipeline = None  # Reset cache agar model baru dimuat

    t = threading.Thread(target=run_training, daemon=True)
    t.start()

    return jsonify({
        "success": True,
        "message": "Training dimulai di background. Refresh halaman dalam beberapa menit."
    })


# ── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint tidak ditemukan."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Kesalahan internal server."}), 500


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pastikan direktori data ada
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Pre-load model jika sudah ada
    if os.path.exists(MODEL_PATH):
        try:
            get_pipeline()
            print(f"[OK] Model dimuat dari: {MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Gagal memuat model: {e}")
    else:
        print(f"[WARN] Model belum ada. Jalankan: python train.py")

    app.run(debug=True, host="0.0.0.0", port=5000)
