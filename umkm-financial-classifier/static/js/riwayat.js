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

function updateStats(data) {
  const total = data.length;
  const sehat = data.filter(d => d.output?.label === 1).length;
  const tidak = total - sehat;
  setText("total-predictions", total.toLocaleString("id-ID"));
  setText("total-sehat",       sehat.toLocaleString("id-ID"));
  setText("total-tidak",       tidak.toLocaleString("id-ID"));
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
    const i   = item.input  || {};
    const o   = item.output || {};
    const isS = o.label === 1;

    // Hitung pendapatan & pengeluaran kembali dari fitur model
    const rev  = i.Monthly_Revenue;
    const npm  = i.Net_Profit_Margin;
    const burn = i.Burn_Rate_Ratio;
    // pengeluaran = pendapatan * burn_rate
    const exp  = rev && burn ? Math.round(rev * burn) : null;

    const badge = isS
      ? `<span class="badge-sehat-sm">✅ Sehat</span>`
      : `<span class="badge-tidak-sm">⚠️ Tidak Sehat</span>`;

    const prob     = o.prob_sehat != null ? o.prob_sehat : 0;
    const probPct  = (prob * 100).toFixed(1) + "%";
    const barWidth = Math.round(prob * 100);

    return `<tr>
      <td style="color:var(--clr-muted);font-size:.8rem">${start + idx + 1}</td>
      <td style="white-space:nowrap">${formatTs(item.timestamp)}</td>
      <td>${rev ? "Rp " + Math.round(rev).toLocaleString("id-ID") : "—"}</td>
      <td>${exp ? "Rp " + exp.toLocaleString("id-ID") : "—"}</td>
      <td style="color:${npm >= 0 ? "var(--clr-sehat)" : "var(--clr-tidak)"};font-weight:700">
        ${npm != null ? fmtDec(npm) + "%" : "—"}
      </td>
      <td>${i.Transaction_Count ?? "—"}</td>
      <td>${i.Repeat_Order_Rate != null ? fmtDec(i.Repeat_Order_Rate) + "%" : "—"}</td>
      <td>${badge}</td>
      <td>
        <div class="prob-bar">
          <span style="font-weight:700;min-width:38px">${probPct}</span>
          <div style="flex:1;background:var(--clr-border);border-radius:9999px;height:6px;overflow:hidden">
            <div style="width:${barWidth}%;height:100%;background:${prob >= 0.5 ? "var(--clr-sehat)" : "var(--clr-tidak)"};border-radius:9999px"></div>
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
    if (status === "sehat"       && item.output?.label !== 1) return false;
    if (status === "tidak-sehat" && item.output?.label !== 0) return false;
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
    const exp = i.Monthly_Revenue && i.Burn_Rate_Ratio ? Math.round(i.Monthly_Revenue * i.Burn_Rate_Ratio) : "-";
    const prob = o.prob_sehat ? (o.prob_sehat * 100).toFixed(1) + "%" : "-";
    const isS = o.label === 1;
    return `
      <tr>
        <td>${idx + 1}</td>
        <td>${formatTs(item.timestamp)}</td>
        <td>Rp ${(i.Monthly_Revenue || 0).toLocaleString("id-ID")}</td>
        <td>Rp ${typeof exp === "number" ? exp.toLocaleString("id-ID") : exp}</td>
        <td>${i.Net_Profit_Margin != null ? i.Net_Profit_Margin + "%" : "-"}</td>
        <td>${i.Burn_Rate_Ratio || "-"}</td>
        <td>${i.Transaction_Count || "-"}</td>
        <td>${i.Repeat_Order_Rate != null ? i.Repeat_Order_Rate + "%" : "-"}</td>
        <td style="color:${isS ? "#10b981" : "#ef4444"};font-weight:bold">${o.label_teks || "-"}</td>
        <td>${prob}</td>
      </tr>
    `;
  }).join("");

  const total = data.length;
  const sehat = data.filter(d => (d.output || {}).label === 1).length;
  const tidak = total - sehat;

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
        <div class="stat"><div class="stat-label">Sehat</div><div class="stat-value sehat">${sehat}</div></div>
        <div class="stat"><div class="stat-label">Tidak Sehat</div><div class="stat-value tidak">${tidak}</div></div>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Waktu</th>
            <th>Pendapatan</th>
            <th>Pengeluaran</th>
            <th>Margin</th>
            <th>Burn Rate</th>
            <th>Transaksi</th>
            <th>Repeat Order</th>
            <th>Hasil</th>
            <th>P(Sehat)</th>
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
