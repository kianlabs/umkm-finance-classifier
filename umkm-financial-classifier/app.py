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

from src.preprocessing import validate_input, FITUR_TERPILIH
from src.model import load_pipeline, predict_single
from src.evaluation import (
    compute_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    cross_validate,
)
from src.visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_kde_per_feature,
    plot_confusion_matrix,
    plot_roc_curve,
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
    Terima JSON body dengan 6 fitur, kembalikan prediksi dan rekomendasi.
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
        "success":         True,
        "accuracy":        metrics.get("accuracy"),
        "precision":       metrics.get("precision"),
        "recall":          metrics.get("recall"),
        "f1":              metrics.get("f1_weighted"),
        "roc_auc":         metrics.get("roc_auc"),
        "cv_mean_acc":     metrics.get("mean_acc"),
        "cv_std_acc":      metrics.get("std_acc"),
        "cv_mean_f1":      metrics.get("mean_f1"),
        "cv_scores":       metrics.get("scores", []),
        "total_train":     metrics.get("total_train"),
        "total_test":      metrics.get("total_test"),
        "training_time":   metrics.get("training_time_seconds"),
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

        cm  = np.array(metrics["confusion_matrix"])
        fpr = np.array(metrics["roc_fpr"])
        tpr = np.array(metrics["roc_tpr"])

        viz = {
            "success":          True,
            "class_dist":       plot_class_distribution(y),
            "correlation":      plot_correlation_heatmap(X),
            "kde":              plot_kde_per_feature(X_train.sample(min(5000, len(X_train)), random_state=42),
                                                     y_train.loc[X_train.sample(min(5000, len(X_train)), random_state=42).index]),
            "confusion_matrix": plot_confusion_matrix(cm),
            "roc_curve":        plot_roc_curve(fpr, tpr, metrics["roc_auc"]),
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


# ── Halaman Prediksi Batch ────────────────────────────────────────────────────

@app.route("/batch")
def batch():
    """Halaman upload CSV untuk prediksi massal."""
    return render_template("batch.html")


@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    """
    Terima file CSV, validasi kolom, prediksi massal, kembalikan JSON.
    """
    import io
    import pandas as pd

    if "file" not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file yang diupload."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Nama file kosong."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"success": False, "error": "Format file harus .csv."}), 400

    try:
        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8-sig")))
    except Exception as e:
        return jsonify({"success": False, "error": f"Gagal membaca CSV: {str(e)}"}), 400

    # Raw rename (sama kyk load_dataset di preprocessing.py)
    COL_RAW_MAP_BATCH = {
        "Monthly_Revenue":        "Monthly_Revenue",
        "Net_Profit_Margin":      "Net_Profit_Margin (%)",
        "Burn_Rate_Ratio":        "Burn_Rate_Ratio",
        "Transaction_Count":      "Transaction_Count",
        "Repeat_Order_Rate":      "Repeat_Order_Rate (%)",
        "Business_Tenure_Months": "Business_Tenure_Months",
    }
    rename_map = {v: k for k, v in COL_RAW_MAP_BATCH.items() if v in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # Validasi kolom
    from src.preprocessing import validate_csv_columns, FITUR_TERPILIH, BATAS_VALIDASI
    is_valid, missing = validate_csv_columns(df)
    if not is_valid:
        return jsonify({
            "success": False,
            "error": f"Kolom tidak lengkap. Kolom yang dibutuhkan: {', '.join(FITUR_TERPILIH)}. "
                     f"Missing: {', '.join(missing)}"
        }), 400

    # Validasi nilai per baris (bersihkan baris invalid)
    valid_rows = []
    error_rows = []
    for idx, row in df.iterrows():
        data = {col: row[col] for col in FITUR_TERPILIH}
        valid, errs = validate_input(data)
        if valid:
            valid_rows.append(idx)
        else:
            error_rows.append({"baris": int(idx) + 2, "errors": errs})

    if len(valid_rows) == 0:
        return jsonify({
            "success": False,
            "error": "Tidak ada baris dengan data valid untuk diprediksi.",
            "error_rows": error_rows
        }), 400

    try:
        pipeline = get_pipeline()
        from src.model import predict_dataframe, generate_rekomendasi
        df_valid = df.loc[valid_rows].copy()
        hasil = predict_dataframe(pipeline, df_valid)

        # Tambah rekomendasi per baris
        hasil["Rekomendasi"] = hasil.apply(
            lambda r: generate_rekomendasi(r["Prediksi_Label"], [r["Prob_Tidak_Sehat"], r["Prob_Sehat"]]),
            axis=1
        )

        # Simpan ke riwayat
        for idx, row in hasil.iterrows():
            input_data = {col: row[col] for col in FITUR_TERPILIH}
            output_data = {
                "label": int(row["Prediksi_Label"]),
                "label_teks": row["Prediksi_Teks"],
                "prob_sehat": row["Prob_Sehat"],
                "prob_tidak_sehat": row["Prob_Tidak_Sehat"],
                "rekomendasi": row["Rekomendasi"],
            }
            append_history(input_data, output_data)

        # Hasil sebagai list of dict
        predictions = []
        for _, row in hasil.iterrows():
            predictions.append({
                **{col: (
                    float(row[col]) if isinstance(row[col], (int, float))
                    else str(row[col])
                ) for col in FITUR_TERPILIH},
                "label": int(row["Prediksi_Label"]),
                "label_teks": row["Prediksi_Teks"],
                "prob_sehat": round(float(row["Prob_Sehat"]), 4),
                "prob_tidak_sehat": round(float(row["Prob_Tidak_Sehat"]), 4),
                "rekomendasi": row["Rekomendasi"],
            })

        # Statistik
        total_sehat = sum(1 for p in predictions if p["label"] == 1)
        total_tidak = len(predictions) - total_sehat

        return jsonify({
            "success": True,
            "total": len(predictions),
            "total_sehat": total_sehat,
            "total_tidak": total_tidak,
            "total_invalid": len(error_rows),
            "error_rows": error_rows,
            "predictions": predictions,
        })

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": f"Kesalahan saat prediksi: {str(e)}"}), 500


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
