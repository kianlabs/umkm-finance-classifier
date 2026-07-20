/**
 * dashboard.js — Memuat metrik model dari API /api/metrics
 */
document.addEventListener("DOMContentLoaded", loadMetrics);

async function loadMetrics() {
  try {
    const res  = await fetch("/api/metrics");
    const data = await res.json();

    if (!data.success) {
      document.getElementById("error-banner").classList.remove("hidden");
      return;
    }

    setVal("val-accuracy",  data.accuracy);
    setVal("val-f1",        data.f1);
    setVal("val-auc",       data.roc_auc);
    setVal("val-precision", data.precision);
    setVal("val-recall",    data.recall);

    const cvText = data.cv_mean_acc != null
      ? `${pct(data.cv_mean_acc)} ±${(data.cv_std_acc * 100).toFixed(2)}%`
      : "—";
    document.getElementById("val-cv").textContent = cvText;

    document.querySelectorAll(".metric-card").forEach(c => c.classList.remove("skeleton"));

    setText("total-train",   data.total_train != null ? data.total_train.toLocaleString("id-ID") + " data" : "—");
    setText("total-test",    data.total_test  != null ? data.total_test.toLocaleString("id-ID")  + " data" : "—");
    setText("training-time", data.training_time != null ? `${data.training_time} detik` : "—");

  } catch (err) {
    console.error(err);
    document.getElementById("error-banner").classList.remove("hidden");
  }
}

function setVal(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value != null ? pct(value) : "—";
  const num = parseFloat(value);
  if (!isNaN(num)) {
    el.style.color = num >= 0.94 ? "#16a34a" : num >= 0.80 ? "#d97706" : "#dc2626";
  }
}

function pct(v) { return (v * 100).toFixed(2) + "%"; }
function setText(id, t) { const el = document.getElementById(id); if (el) el.textContent = t; }
