/**
 * visualisasi.js
 * Memuat semua plot dari API /api/visualizations dan render ke halaman.
 */

document.addEventListener("DOMContentLoaded", () => {
  loadVisualizations();
  setupTabs();
});

// ── Load Visualisasi ─────────────────────────────────────────────────────────

async function loadVisualizations() {
  const loading = document.getElementById("viz-loading");
  const errDiv  = document.getElementById("viz-error");
  const content = document.getElementById("viz-content");

  try {
    const res  = await fetch("/api/visualizations");
    const data = await res.json();

    loading.classList.add("hidden");

    if (!data.success) {
      document.getElementById("viz-error-text").textContent = data.error || "Gagal memuat visualisasi.";
      errDiv.classList.remove("hidden");
      return;
    }

    // EDA
    setImage("img-class-dist",  data.class_dist);
    setImage("img-correlation",  data.correlation);
    setImage("img-kde",          data.kde);

    // Model Performance
    setImage("img-confusion",    data.confusion_matrix);
    setImage("img-roc",          data.roc_curve);
    setImage("img-cv",           data.cv_scores);

    // Feature Importance
    setImage("img-importance",   data.feature_importance);

    content.classList.remove("hidden");

  } catch (err) {
    loading.classList.add("hidden");
    document.getElementById("viz-error-text").textContent =
      "Tidak dapat terhubung ke server. Pastikan server Flask berjalan.";
    errDiv.classList.remove("hidden");
    console.error(err);
  }
}

// ── Tab Navigation ───────────────────────────────────────────────────────────

function setupTabs() {
  const tabBtns   = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;

      tabBtns.forEach(b   => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.add("hidden"));

      btn.classList.add("active");
      const panel = document.getElementById(`tab-${target}`);
      if (panel) panel.classList.remove("hidden");
    });
  });
}

// ── Helper ───────────────────────────────────────────────────────────────────

function setImage(id, src) {
  const el = document.getElementById(id);
  if (el && src) {
    el.src = src;
    el.onerror = () => { el.alt = "Gagal memuat gambar."; };
  }
}
