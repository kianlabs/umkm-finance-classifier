/**
 * klasifikasi.js
 * 6 input sesuai laporan + field pengeluaran sebagai helper hitung otomatis.
 * Pengeluaran → auto-isi Net_Profit_Margin & Burn_Rate_Ratio (tapi masih bisa edit manual).
 */

let gaugeChart    = null;
let autoFillNPM   = false; // flag: NPM sedang diisi otomatis
let autoFillBurn  = false; // flag: Burn sedang diisi otomatis

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("klasifikasi-form").addEventListener("submit", handleSubmit);
  document.getElementById("Monthly_Revenue").addEventListener("input",  triggerAutoCalc);
  document.getElementById("pengeluaran").addEventListener("input",       triggerAutoCalc);

  // Jika user edit manual NPM / Burn → lepas auto-fill
  document.getElementById("Net_Profit_Margin").addEventListener("input", () => { autoFillNPM  = false; });
  document.getElementById("Burn_Rate_Ratio").addEventListener("input",   () => { autoFillBurn = false; });
});

/* ── Auto Kalkulasi ─────────────────────────────────────────────────────────── */

function triggerAutoCalc() {
  const pendapatan  = parseFloat(document.getElementById("Monthly_Revenue").value);
  const pengeluaran = parseFloat(document.getElementById("pengeluaran").value);
  const preview     = document.getElementById("calc-preview");

  if (!pendapatan || !pengeluaran || isNaN(pendapatan) || isNaN(pengeluaran) || pendapatan <= 0) {
    preview.classList.add("hidden");
    return;
  }

  const laba = pendapatan - pengeluaran;
  const npm  = (laba / pendapatan) * 100;
  const burn = pengeluaran / pendapatan;

  // Tampilkan preview
  const labaEl = document.getElementById("calc-laba");
  labaEl.textContent   = "Rp " + laba.toLocaleString("id-ID");
  labaEl.style.color   = laba >= 0 ? "#16a34a" : "#dc2626";
  document.getElementById("calc-npm").textContent  = npm.toFixed(2) + "%";
  document.getElementById("calc-burn").textContent = burn.toFixed(3);
  preview.classList.remove("hidden");

  // Auto-isi NPM (hanya jika user belum edit manual)
  if (autoFillNPM !== false || document.getElementById("Net_Profit_Margin").value === "") {
    autoFillNPM = true;
    document.getElementById("Net_Profit_Margin").value = npm.toFixed(2);
  }

  // Auto-isi Burn Rate (hanya jika user belum edit manual)
  if (autoFillBurn !== false || document.getElementById("Burn_Rate_Ratio").value === "") {
    autoFillBurn = true;
    document.getElementById("Burn_Rate_Ratio").value = burn.toFixed(3);
  }
}

/* ── Validasi ───────────────────────────────────────────────────────────────── */

const VALIDASI = {
  Monthly_Revenue:        { min: 0.01,  max: Infinity, label: "Pendapatan Bulanan" },
  Net_Profit_Margin:      { min: -35.0, max: 45.0,     label: "Margin Keuntungan Bersih" },
  Burn_Rate_Ratio:        { min: 0.44,  max: 1.55,     label: "Rasio Pengeluaran" },
  Transaction_Count:      { min: 9,     max: 285,      label: "Jumlah Transaksi" },
  Repeat_Order_Rate:      { min: 2.0,   max: 54.0,     label: "Tingkat Pelanggan Kembali" },
  Business_Tenure_Months: { min: 3,     max: 179,      label: "Lama Usaha" },
};

/* ── Submit ─────────────────────────────────────────────────────────────────── */

async function handleSubmit(e) {
  e.preventDefault();
  clearErrors();

  const data = {
    Monthly_Revenue:        parseFloat(document.getElementById("Monthly_Revenue").value),
    Net_Profit_Margin:      parseFloat(document.getElementById("Net_Profit_Margin").value),
    Burn_Rate_Ratio:        parseFloat(document.getElementById("Burn_Rate_Ratio").value),
    Transaction_Count:      parseInt(document.getElementById("Transaction_Count").value, 10),
    Repeat_Order_Rate:      parseFloat(document.getElementById("Repeat_Order_Rate").value),
    Business_Tenure_Months: parseInt(document.getElementById("Business_Tenure_Months").value, 10),
  };

  let hasError = false;
  for (const [field, rule] of Object.entries(VALIDASI)) {
    const val = data[field];
    if (isNaN(val)) {
      setErr(field, `${rule.label} harus diisi dengan angka.`);
      hasError = true;
    } else if (val < rule.min || (rule.max !== Infinity && val > rule.max)) {
      const maxStr = rule.max === Infinity ? "∞" : rule.max;
      setErr(field, `${rule.label} harus antara ${rule.min} s.d. ${maxStr}.`);
      hasError = true;
    }
  }
  if (hasError) return;

  setLoading(true);

  try {
    const res    = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(data),
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

  const isSehat = result.label === 1;

  const badge = document.getElementById("result-badge");
  badge.textContent = isSehat ? "✅ KONDISI SEHAT" : "⚠️ PERLU PERHATIAN";
  badge.className   = "result-badge " + (isSehat ? "badge-sehat" : "badge-tidak");

  document.getElementById("result-tagline").textContent = isSehat
    ? `Keyakinan sistem: ${(result.prob_sehat * 100).toFixed(1)}% Sehat`
    : `Keyakinan sistem: ${(result.prob_tidak_sehat * 100).toFixed(1)}% Tidak Sehat`;

  renderGauge(result.prob_sehat, result.prob_tidak_sehat);
  document.getElementById("gauge-sehat").textContent = (result.prob_sehat * 100).toFixed(1) + "%";
  document.getElementById("gauge-tidak").textContent = (result.prob_tidak_sehat * 100).toFixed(1) + "%";

  renderSummary(data);

  document.getElementById("rekomendasi-text").textContent = result.rekomendasi;
  document.getElementById("result-timestamp").textContent =
    `Dianalisis pada: ${formatTs(result.timestamp)}`;
}

function renderSummary(data) {
  const pengeluaran = parseFloat(document.getElementById("pengeluaran").value);
  const grid = document.getElementById("summary-grid");

  const items = [
    { label: "💰 Pendapatan",        value: "Rp " + data.Monthly_Revenue.toLocaleString("id-ID") },
    { label: "💸 Pengeluaran",       value: !isNaN(pengeluaran) ? "Rp " + pengeluaran.toLocaleString("id-ID") : "—" },
    { label: "📊 Margin Keuntungan", value: data.Net_Profit_Margin.toFixed(2) + "%",
      color: data.Net_Profit_Margin >= 0 ? "var(--clr-sehat)" : "var(--clr-tidak)" },
    { label: "🔥 Rasio Pengeluaran", value: data.Burn_Rate_Ratio.toFixed(3),
      color: data.Burn_Rate_Ratio < 1 ? "var(--clr-sehat)" : "var(--clr-tidak)" },
    { label: "🛒 Jumlah Transaksi",  value: data.Transaction_Count + " transaksi" },
    { label: "🔄 Repeat Order",      value: data.Repeat_Order_Rate.toFixed(1) + "%" },
    { label: "📅 Lama Usaha",        value: formatTenure(data.Business_Tenure_Months) },
  ];

  grid.innerHTML = items.map(it => `
    <div class="summary-item">
      <div class="si-label">${it.label}</div>
      <div class="si-value" ${it.color ? `style="color:${it.color}"` : ""}>${it.value}</div>
    </div>
  `).join("");
}

/* ── Gauge ──────────────────────────────────────────────────────────────────── */

function renderGauge(probSehat, probTidak) {
  const ctx = document.getElementById("gauge-chart").getContext("2d");
  if (gaugeChart) gaugeChart.destroy();
  gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Sehat", "Tidak Sehat"],
      datasets: [{
        data: [probSehat, probTidak],
        backgroundColor: ["#16a34a", "#dc2626"],
        borderWidth: 3, borderColor: "#fff", hoverOffset: 6,
      }],
    },
    options: {
      cutout: "72%", rotation: -90, circumference: 180, responsive: false,
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
  autoFillNPM  = false;
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
