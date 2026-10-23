/**
 * riwayat.js — Riwayat prediksi dengan stat pills, filter, export, pagination
 */

let allHistory  = [];
let currentPage = 1;
const PER_PAGE  = 20;

document.addEventListener("DOMContentLoaded", loadHistory);

async function loadHistory() {
  try {
    const res  = await fetch("/api/history?per_page=10000");
    const data = await res.json();

    document.getElementById("history-loading").classList.add("hidden");

    if (!data.success || data.total === 0) {
      document.getElementById("history-empty").classList.remove("hidden");
      updateStats([]);
      return;
    }

    allHistory = data.predictions;
    updateStats(allHistory);
    renderTable(allHistory);
  } catch (err) {
    document.getElementById("history-loading").classList.add("hidden");
    console.error(err);
  }
}

const WARNA_KELAS_R = { Critical: "#dc2626", Elite: "#7c3aed", Growth: "#16a34a", Struggling: "#d97706" };
const IKON_KELAS_R = { Critical: "🚨", Elite: "👑", Growth: "📈", Struggling: "⚠️" };

function updateStats(data) {
  const total = data.length;
  const count = k => data.filter(d => d.output?.label_teks === k).length;
  setText("total-predictions", total.toLocaleString("id-ID"));
  setText("total-critical",   count("Critical").toLocaleString("id-ID"));
  setText("total-struggling", count("Struggling").toLocaleString("id-ID"));
  setText("total-growth",     count("Growth").toLocaleString("id-ID"));
  setText("total-elite",      count("Elite").toLocaleString("id-ID"));
}

/* ── Render Tabel ─────────────────────────────────────────────────────────── */

function renderTable(data) {
  const empty  = document.getElementById("history-empty");
  const scroll = document.getElementById("table-scroll");
  const tbody  = document.getElementById("history-tbody");
  const pagDiv = document.getElementById("pagination");

  if (data.length === 0) {
    scroll.classList.add("hidden");
    empty.classList.remove("hidden");
    pagDiv.innerHTML = "";
    return;
  }

  empty.classList.add("hidden");
  scroll.classList.remove("hidden");

  const totalPages = Math.ceil(data.length / PER_PAGE);
  currentPage = Math.min(Math.max(currentPage, 1), totalPages);
  const start = (currentPage - 1) * PER_PAGE;
  const slice = data.slice(start, start + PER_PAGE);

  tbody.innerHTML = slice.map((item, idx) => {
    const i = item.input  || {};
    const o = item.output || {};
    const kelas = o.label_teks || "—";
    const warna = WARNA_KELAS_R[kelas] || "#64748b";
    const badge = `<span class="badge-kelas-sm" style="background:#f1f5f9;color:${warna};border:1px solid ${warna}">${IKON_KELAS_R[kelas] || ""} ${kelas}</span>`;

    const prob = o.prob_max != null ? o.prob_max : 0;
    const probPct = (prob * 100).toFixed(1) + "%";
    const barWidth = Math.round(prob * 100);

    return `<tr>
      <td style="color:var(--clr-muted);font-size:.8rem">${start + idx + 1}</td>
      <td style="white-space:nowrap">${formatTs(item.timestamp)}</td>
      <td style="color:${i.Net_Profit_Margin >= 0 ? "var(--clr-sehat)" : "var(--clr-tidak)"};font-weight:700">
        ${i.Net_Profit_Margin != null ? fmtDec(i.Net_Profit_Margin) + "%" : "—"}
      </td>
      <td>${i.Burn_Rate_Ratio != null ? fmtDec(i.Burn_Rate_Ratio, 3) : "—"}</td>
      <td>${i.Avg_Historical_Rating != null ? fmtDec(i.Avg_Historical_Rating) + " / 5" : "—"}</td>
      <td>${i.Repeat_Order_Rate != null ? fmtDec(i.Repeat_Order_Rate) + "%" : "—"}</td>
      <td>${i.Business_Tenure_Months ?? "—"} bln</td>
      <td>${badge}</td>
      <td>
        <div class="prob-bar">
          <span style="font-weight:700;min-width:38px">${probPct}</span>
          <div style="flex:1;background:var(--clr-border);border-radius:9999px;height:6px;overflow:hidden">
            <div style="width:${barWidth}%;height:100%;background:${warna};border-radius:9999px"></div>
          </div>
        </div>
      </td>
    </tr>`;
  }).join("");

  renderPagination(data.length, totalPages);
}

/* ── Pagination ─────────────────────────────────────────────────────────────*/

function renderPagination(total, totalPages) {
  const pagDiv = document.getElementById("pagination");
  if (totalPages <= 1) { pagDiv.innerHTML = ""; return; }

  let html = `<button class="page-btn" onclick="goPage(${currentPage-1})" ${currentPage===1?"disabled":""}>‹ Prev</button>`;
  for (let p = 1; p <= totalPages; p++) {
    if (p===1||p===totalPages||Math.abs(p-currentPage)<=2)
      html += `<button class="page-btn ${p===currentPage?"active-page":""}" onclick="goPage(${p})">${p}</button>`;
    else if (Math.abs(p-currentPage)===3)
      html += `<span style="padding:.4rem .3rem;color:var(--clr-muted)">…</span>`;
  }
  html += `<button class="page-btn" onclick="goPage(${currentPage+1})" ${currentPage===totalPages?"disabled":""}>Next ›</button>`;
  pagDiv.innerHTML = html;
}

function goPage(p) {
  currentPage = p;
  renderTable(getFiltered());
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ── Filter ──────────────────────────────────────────────────────────────── */

function filterHistory() {
  currentPage = 1;
  renderTable(getFiltered());
}

function getFiltered() {
  const status = document.getElementById("filter-status").value;
  const search = document.getElementById("search-history").value.toLowerCase();
  return allHistory.filter(item => {
    if (status !== "all" && item.output?.label_teks !== status) return false;
    if (search) {
      const ts = (item.timestamp || "").toLowerCase();
      const id = (item.id        || "").toLowerCase();
      if (!ts.includes(search) && !id.includes(search)) return false;
    }
    return true;
  });
}

/* ── Export PDF ──────────────────────────────────────────────────────────── */

function exportPDF() {
  const data = getFiltered();
  if (!data.length) { alert("Tidak ada data untuk diekspor."); return; }

  // Buka window baru untuk print
  const printWindow = window.open('', '_blank');
  
  const rows = data.map((item, idx) => {
    const i = item.input || {}; const o = item.output || {};
    const prob = o.prob_max != null ? (o.prob_max * 100).toFixed(1) + "%" : "-";
    const warna = WARNA_KELAS_R[o.label_teks] || "#1e293b";
    return `
      <tr>
        <td>${idx + 1}</td>
        <td>${formatTs(item.timestamp)}</td>
        <td>${i.Net_Profit_Margin != null ? i.Net_Profit_Margin + "%" : "-"}</td>
        <td>${i.Burn_Rate_Ratio ?? "-"}</td>
        <td>${i.Avg_Historical_Rating ?? "-"}</td>
        <td>${i.Repeat_Order_Rate != null ? i.Repeat_Order_Rate + "%" : "-"}</td>
        <td>${i.Business_Tenure_Months ?? "-"}</td>
        <td style="color:${warna};font-weight:bold">${o.label_teks || "-"}</td>
        <td>${prob}</td>
      </tr>
    `;
  }).join("");

  const total = data.length;
  const count = k => data.filter(d => d.output?.label_teks === k).length;
  setText("total-predictions", total.toLocaleString("id-ID"));
  setText("total-critical",   count("Critical").toLocaleString("id-ID"));
  setText("total-struggling", count("Struggling").toLocaleString("id-ID"));
  setText("total-growth",     count("Growth").toLocaleString("id-ID"));
  setText("total-elite",      count("Elite").toLocaleString("id-ID"));
}

/* ── Render Tabel ─────────────────────────────────────────────────────────── */

function renderTable(data) {
  const empty  = document.getElementById("history-empty");
  const scroll = document.getElementById("table-scroll");
  const tbody  = document.getElementById("history-tbody");
  const pagDiv = document.getElementById("pagination");

  if (data.length === 0) {
    scroll.classList.add("hidden");
    empty.classList.remove("hidden");
    pagDiv.innerHTML = "";
    return;
  }

  empty.classList.add("hidden");
  scroll.classList.remove("hidden");

  const totalPages = Math.ceil(data.length / PER_PAGE);
  currentPage = Math.min(Math.max(currentPage, 1), totalPages);
  const start = (currentPage - 1) * PER_PAGE;
  const slice = data.slice(start, start + PER_PAGE);

  tbody.innerHTML = slice.map((item, idx) => {
    const i = item.input  || {};
    const o = item.output || {};
    const kelas = o.label_teks || "—";
    const warna = WARNA_KELAS_R[kelas] || "#64748b";
    const badge = `<span class="badge-kelas-sm" style="background:#f1f5f9;color:${warna};border:1px solid ${warna}">${IKON_KELAS_R[kelas] || ""} ${kelas}</span>`;

    const prob = o.prob_max != null ? o.prob_max : 0;
    const probPct = (prob * 100).toFixed(1) + "%";
    const barWidth = Math.round(prob * 100);

    return `<tr>
      <td style="color:var(--clr-muted);font-size:.8rem">${start + idx + 1}</td>
      <td style="white-space:nowrap">${formatTs(item.timestamp)}</td>
      <td style="color:${i.Net_Profit_Margin >= 0 ? "var(--clr-sehat)" : "var(--clr-tidak)"};font-weight:700">
        ${i.Net_Profit_Margin != null ? fmtDec(i.Net_Profit_Margin) + "%" : "—"}
      </td>
      <td>${i.Burn_Rate_Ratio != null ? fmtDec(i.Burn_Rate_Ratio, 3) : "—"}</td>
      <td>${i.Avg_Historical_Rating != null ? fmtDec(i.Avg_Historical_Rating) + " / 5" : "—"}</td>
      <td>${i.Repeat_Order_Rate != null ? fmtDec(i.Repeat_Order_Rate) + "%" : "—"}</td>
      <td>${i.Business_Tenure_Months ?? "—"} bln</td>
      <td>${badge}</td>
      <td>
        <div class="prob-bar">
          <span style="font-weight:700;min-width:38px">${probPct}</span>
          <div style="flex:1;background:var(--clr-border);border-radius:9999px;height:6px;overflow:hidden">
            <div style="width:${barWidth}%;height:100%;background:${warna};border-radius:9999px"></div>
          </div>
        </div>
      </td>
    </tr>`;
  }).join("");

  renderPagination(data.length, totalPages);
}

/* ── Pagination ─────────────────────────────────────────────────────────────*/

function renderPagination(total, totalPages) {
  const pagDiv = document.getElementById("pagination");
  if (totalPages <= 1) { pagDiv.innerHTML = ""; return; }

  let html = `<button class="page-btn" onclick="goPage(${currentPage-1})" ${currentPage===1?"disabled":""}>‹ Prev</button>`;
  for (let p = 1; p <= totalPages; p++) {
    if (p===1||p===totalPages||Math.abs(p-currentPage)<=2)
      html += `<button class="page-btn ${p===currentPage?"active-page":""}" onclick="goPage(${p})">${p}</button>`;
    else if (Math.abs(p-currentPage)===3)
      html += `<span style="padding:.4rem .3rem;color:var(--clr-muted)">…</span>`;
  }
  html += `<button class="page-btn" onclick="goPage(${currentPage+1})" ${currentPage===totalPages?"disabled":""}>Next ›</button>`;
  pagDiv.innerHTML = html;
}

function goPage(p) {
  currentPage = p;
  renderTable(getFiltered());
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ── Filter ──────────────────────────────────────────────────────────────── */

function filterHistory() {
  currentPage = 1;
  renderTable(getFiltered());
}

function getFiltered() {
  const status = document.getElementById("filter-status").value;
  const search = document.getElementById("search-history").value.toLowerCase();
  return allHistory.filter(item => {
    if (status !== "all" && item.output?.label_teks !== status) return false;
    if (search) {
      const ts = (item.timestamp || "").toLowerCase();
      const id = (item.id        || "").toLowerCase();
      if (!ts.includes(search) && !id.includes(search)) return false;
    }
    return true;
  });
}

/* ── Export PDF ──────────────────────────────────────────────────────────── */

function exportPDF() {
  const data = getFiltered();
  if (!data.length) { alert("Tidak ada data untuk diekspor."); return; }

  // Buka window baru untuk print
  const printWindow = window.open('', '_blank');
  
  const rows = data.map((item, idx) => {
    const i = item.input || {}; const o = item.output || {};
    const prob = o.prob_max != null ? (o.prob_max * 100).toFixed(1) + "%" : "-";
    const warna = WARNA_KELAS_R[o.label_teks] || "#1e293b";
    return `
      <tr>
        <td>${idx + 1}</td>
        <td>${formatTs(item.timestamp)}</td>
        <td>${i.Net_Profit_Margin != null ? i.Net_Profit_Margin + "%" : "-"}</td>
        <td>${i.Burn_Rate_Ratio ?? "-"}</td>
        <td>${i.Avg_Historical_Rating ?? "-"}</td>
        <td>${i.Repeat_Order_Rate != null ? i.Repeat_Order_Rate + "%" : "-"}</td>
        <td>${i.Business_Tenure_Months ?? "-"}</td>
        <td style="color:${warna};font-weight:bold">${o.label_teks || "-"}</td>
        <td>${prob}</td>
      </tr>
    `;
  }).join("");

  const total = data.length;
  const positif = data.filter(d => ["Growth", "Elite"].includes((d.output || {}).label_teks)).length;
  const negatif = total - positif;

  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Riwayat Analisis Keuangan UMKM</title>
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; color: #1e293b; }
        .header { text-align: center; margin-bottom: 20px; border-bottom: 2px solid #2563eb; padding-bottom: 15px; }
        .header h1 { font-size: 24px; color: #1e2761; margin-bottom: 5px; }
        .header p { color: #64748b; font-size: 14px; }
        .stats { display: flex; justify-content: center; gap: 30px; margin-bottom: 20px; }
        .stat { background: #f1f5f9; padding: 10px 20px; border-radius: 8px; text-align: center; }
        .stat-label { font-size: 12px; color: #64748b; }
        .stat-value { font-size: 20px; font-weight: bold; color: #1e2761; }
        .stat-value.sehat { color: #10b981; }
        .stat-value.tidak { color: #ef4444; }
        table { width: 100%; border-collapse: collapse; font-size: 11px; }
        th { background: #2563eb; color: white; padding: 10px 8px; text-align: left; font-weight: 600; }
        td { padding: 8px; border-bottom: 1px solid #e2e8f0; }
        tr:nth-child(even) { background: #f8fafc; }
        .footer { margin-top: 20px; text-align: center; font-size: 11px; color: #94a3b8; }
        @media print { body { padding: 0; } }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>📊 Laporan Riwayat Analisis Keuangan UMKM</h1>
        <p>Skripsi Ridzkyan Buti Pratama - UMKMCek</p>
      </div>
      <div class="stats">
        <div class="stat"><div class="stat-label">Total Analisis</div><div class="stat-value">${total}</div></div>
        <div class="stat"><div class="stat-label">Positif (Growth/Elite)</div><div class="stat-value sehat">${positif}</div></div>
        <div class="stat"><div class="stat-label">Negatif (Critical/Struggling)</div><div class="stat-value tidak">${negatif}</div></div>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Waktu</th>
            <th>Margin</th>
            <th>Burn Rate</th>
            <th>Rating</th>
            <th>Repeat Order</th>
            <th>Tenure (bln)</th>
            <th>Kelas</th>
            <th>Prob Kelas</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
      <div class="footer">
        Dicetak pada: ${new Date().toLocaleString("id-ID")} | Sistem Klasifikasi Keuangan UMKM - UMKMCek
      </div>
      <script>window.onload = function() { window.print(); }<\/script>
    </body>
    </html>
  `);
  printWindow.document.close();
}

/* ── Hapus Semua ─────────────────────────────────────────────────────────── */

async function clearHistory() {
  if (!confirm("Yakin ingin menghapus semua riwayat? Tindakan ini tidak bisa dibatalkan.")) return;
  try {
    await fetch("/api/history", { method: "DELETE" });
    allHistory = [];
    updateStats([]);
    renderTable([]);
  } catch (e) { alert("Gagal menghapus riwayat."); }
}

/* ── Helper ──────────────────────────────────────────────────────────────── */

function formatTs(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("id-ID", {
      day:"2-digit", month:"short", year:"numeric", hour:"2-digit", minute:"2-digit"
    });
  } catch { return iso; }
}

function fmtDec(v, d=2) { return v != null ? Number(v).toFixed(d) : "—"; }
function setText(id, t)  { const el = document.getElementById(id); if (el) el.textContent = t; }
