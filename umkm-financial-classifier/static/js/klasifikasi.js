/**
 * klasifikasi.js
 * 5 fitur model hasil SBS + field pendapatan/pengeluaran sebagai helper
 * hitung otomatis Net_Profit_Margin & Burn_Rate_Ratio (pendapatan &
 * pengeluaran BUKAN fitur model — tereliminasi SBS).
 * Hasil: klasifikasi 4 kelas (Critical, Elite, Growth, Struggling)
 * dengan probabilitas per kelas.
 */

let gaugeChart = null;
let autoFillNPM = false;
let autoFillBurn = false;

const WARNA_KELAS = {
  Critical: "#dc2626",
  Elite: "#8e44ad",
  Growth: "#16a34a",
  Struggling: "#d97706",
};
const IKON_KELAS = { Critical: "🚨", Elite: "👑", Growth: "📈", Struggling: "⚠️" };
const KELAS_LIST = ["Critical", "Elite", "Growth", "Struggling"];

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("klasifikasi-form").addEventListener("submit", handleSubmit);
  document.getElementById("Monthly_Revenue").addEventListener("input", triggerAutoCalc);
  document.getElementById("pengeluaran").addEventListener("input", triggerAutoCalc);
  document.getElementById("Net_Profit_Margin").addEventListener("input", () => { autoFillNPM = false; });
  document.getElementById("Burn_Rate_Ratio").addEventListener("input", () => { autoFillBurn = false; });
});

/* ── Auto Kalkulasi (helper) ──────────────────────────────────────────────── */

function triggerAutoCalc() {
  const pendapatan = parseFloat(document.getElementById("Monthly_Revenue").value);
  const pengeluaran = parseFloat(document.getElementById("pengeluaran").value);
  const preview = document.getElementById("calc-preview");

  if (!pendapatan || !pengeluaran || isNaN(pendapatan) || isNaN(pengeluaran) || pendapatan <= 0) {
    preview.classList.add("hidden");
    return;
  }

  const laba = pendapatan - pengeluaran;
  const npm = (laba / pendapatan) * 100;
  const burn = pengeluaran / pendapatan;

  const labaEl = document.getElementById("calc-laba");
  labaEl.textContent = "Rp " + laba.toLocaleString("id-ID");
  labaEl.style.color = laba >= 0 ? "#16a34a" : "#dc2626";
  document.getElementById("calc-npm").textContent = npm.toFixed(2) + "%";
  document.getElementById("calc-burn").textContent = burn.toFixed(3);
  preview.classList.remove("hidden");

  if (autoFillNPM !== false || document.getElementById("Net_Profit_Margin").value === "") {
    autoFillNPM = true;
    document.getElementById("Net_Profit_Margin").value = npm.toFixed(2);
  }
  if (autoFillBurn !== false || document.getElementById("Burn_Rate_Ratio").value === "") {
    autoFillBurn = true;
    document.getElementById("Burn_Rate_Ratio").value = burn.toFixed(3);
  }
}

/* ── Validasi (5 fitur SBS, rentang dari dataset) ─────────────────────────── */

const VALIDASI = {
  Net_Profit_Margin:      { min: -35.0, max: 45.0,  label: "Margin Keuntungan Bersih" },
  Burn_Rate_Ratio:        { min: 0.43,  max: 1.55,  label: "Rasio Pengeluaran" },
  Avg_Historical_Rating:  { min: 1.0,   max: 5.0,   label: "Rating Historis Rata-rata" },
  Business_Tenure_Months: { min: 3,     max: 179,   label: "Lama Usaha" },
  Repeat_Order_Rate:      { min: 2.0,   max: 54.06, label: "Tingkat Pelanggan Kembali" },
};

/* ── Submit ─────────────────────────────────────────────────────────────────── */

async function handleSubmit(e) {
  e.preventDefault();
  clearErrors();

  const data = {
    Net_Profit_Margin:      parseFloat(document.getElementById("Net_Profit_Margin").value),
    Burn_Rate_Ratio:        parseFloat(document.getElementById("Burn_Rate_Ratio").value),
    Avg_Historical_Rating:  parseFloat(document.getElementById("Avg_Historical_Rating").value),
    Business_Tenure_Months: parseInt(document.getElementById("Business_Tenure_Months").value, 10),
    Repeat_Order_Rate:      parseFloat(document.getElementById("Repeat_Order_Rate").value),
  };

  let hasError = false;
  for (const [field, rule] of Object.entries(VALIDASI)) {
    const val = data[field];
    if (isNaN(val)) {
      setErr(field, `${rule.label} harus diisi dengan angka.`);
      hasError = true;
    } else if (val < rule.min || val > rule.max) {
      setErr(field, `${rule.label} harus antara ${rule.min} s.d. ${rule.max}.`);
      hasError = true;
    }
  }
  if (hasError) return;

  setLoading(true);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const result = await res.json();

    if (result.success) {
      displayResult(result, data);
    } else {
      const msg = result.errors ? result.errors.join(" | ") : (result.error || "Terjadi kesalahan.");
      displayError(msg);
    }
  } catch (err) {
    displayError("Tidak dapat terhubung ke server. Pastikan aplikasi Flask berjalan.");
    console.error(err);
  } finally {
    setLoading(false);
  }
}

/* ── Tampilkan Hasil ────────────────────────────────────────────────────────── */

function displayResult(result, data) {
  document.getElementById("result-placeholder").classList.add("hidden");
  document.getElementById("result-error").classList.add("hidden");
  document.getElementById("result-content").classList.remove("hidden");

  const kelas = result.label_teks;
  const badge = document.getElementById("result-badge");
  badge.textContent = `${IKON_KELAS[kelas] || ""} KELAS ${kelas.toUpperCase()}`;
  badge.className = "result-badge badge-" + kelas.toLowerCase();

  document.getElementById("result-tagline").textContent =
    `Keyakinan sistem: ${(result.prob_max * 100).toFixed(1)}% kelas ${kelas}`;

  renderGauge(result.probabilitas);
  renderLegend(result.probabilitas);
  renderSummary(data);

  document.getElementById("rekomendasi-text").textContent = result.rekomendasi;
  document.getElementById("result-timestamp").textContent =
    `Dianalisis pada: ${formatTs(result.timestamp)}`;
}

function renderLegend(probabilitas) {
  const el = document.getElementById("gauge-legend");
  if (!el) return;
  el.innerHTML = KELAS_LIST.map(k => `
    <div class="gl-item">
      <span class="gl-dot" style="background:${WARNA_KELAS[k]}"></span>
      <span>${IKON_KELAS[k]} ${k}:</span><strong>${((probabilitas[k] || 0) * 100).toFixed(1)}%</strong>
    </div>`).join("");
}

function renderSummary(data) {
  const pengeluaran = parseFloat(document.getElementById("pengeluaran").value);
  const pendapatan = parseFloat(document.getElementById("Monthly_Revenue").value);
  const grid = document.getElementById("summary-grid");

  const items = [
    { label: "💰 Pendapatan (helper)", value: !isNaN(pendapatan) ? "Rp " + pendapatan.toLocaleString("id-ID") : "—" },
    { label: "💸 Pengeluaran (helper)", value: !isNaN(pengeluaran) ? "Rp " + pengeluaran.toLocaleString("id-ID") : "—" },
    { label: "📊 Margin Keuntungan", value: data.Net_Profit_Margin.toFixed(2) + "%",
      color: data.Net_Profit_Margin >= 0 ? "var(--clr-sehat)" : "var(--clr-tidak)" },
    { label: "🔥 Rasio Pengeluaran", value: data.Burn_Rate_Ratio.toFixed(3),
      color: data.Burn_Rate_Ratio < 1 ? "var(--clr-sehat)" : "var(--clr-tidak)" },
    { label: "⭐ Rating Historis", value: data.Avg_Historical_Rating.toFixed(2) + " / 5" },
    { label: "🔄 Repeat Order", value: data.Repeat_Order_Rate.toFixed(1) + "%" },
    { label: "📅 Lama Usaha", value: formatTenure(data.Business_Tenure_Months) },
  ];

  grid.innerHTML = items.map(it => `
    <div class="summary-item">
      <div class="si-label">${it.label}</div>
      <div class="si-value" ${it.color ? `style="color:${it.color}"` : ""}>${it.value}</div>
    </div>
  `).join("");
}

/* ── Gauge: doughnut probabilitas 4 kelas ──────────────────────────────────── */

function renderGauge(probabilitas) {
  const ctx = document.getElementById("gauge-chart").getContext("2d");
  if (gaugeChart) gaugeChart.destroy();
  gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: KELAS_LIST,
      datasets: [{
        data: KELAS_LIST.map(k => probabilitas[k] || 0),
        backgroundColor: KELAS_LIST.map(k => WARNA_KELAS[k]),
        borderWidth: 3, borderColor: "#fff", hoverOffset: 6,
      }],
    },
    options: {
      cutout: "60%", responsive: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${(ctx.parsed * 100).toFixed(2)}%` } },
      },
    },
  });
}

/* ── Helper ─────────────────────────────────────────────────────────────────── */

function displayError(msg) {
  document.getElementById("result-placeholder").classList.add("hidden");
  document.getElementById("result-content").classList.add("hidden");
  document.getElementById("result-error").classList.remove("hidden");
  document.getElementById("result-error-text").textContent = msg;
}

function setLoading(on) {
  const btn = document.getElementById("btn-predict");
  btn.disabled = on;
  document.getElementById("btn-text").textContent = on ? "Menganalisis..." : "🔍 Analisis Kondisi Usaha Saya";
  document.getElementById("btn-spinner").classList.toggle("hidden", !on);
}

function setErr(field, msg) {
  const el = document.getElementById(`err-${field}`);
  if (el) el.textContent = msg;
  const wrap = document.getElementById(`wrap-${field}`);
  if (wrap) wrap.classList.add("invalid");
  else { const inp = document.getElementById(field); if (inp) inp.classList.add("invalid"); }
}

function clearErrors() {
  document.querySelectorAll(".field-error").forEach(el => el.textContent = "");
  document.querySelectorAll("input").forEach(el => el.classList.remove("invalid"));
  document.querySelectorAll(".input-prefix-wrap,.input-suffix-wrap").forEach(el => el.classList.remove("invalid"));
  document.getElementById("result-error").classList.add("hidden");
}

function resetForm() {
  document.getElementById("klasifikasi-form").reset();
  autoFillNPM = false;
  autoFillBurn = false;
  clearErrors();
  document.getElementById("calc-preview").classList.add("hidden");
  document.getElementById("result-placeholder").classList.remove("hidden");
  document.getElementById("result-content").classList.add("hidden");
  document.getElementById("result-error").classList.add("hidden");
  if (gaugeChart) { gaugeChart.destroy(); gaugeChart = null; }
}

function formatTenure(months) {
  const th = Math.floor(months / 12);
  const bl = months % 12;
  if (th > 0 && bl > 0) return `${th} thn ${bl} bln`;
  if (th > 0) return `${th} tahun`;
  return `${bl} bulan`;
}

function formatTs(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("id-ID", {
    day: "numeric", month: "long", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}
