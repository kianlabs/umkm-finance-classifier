/**
 * riwayat.js — Riwayat prediksi dengan stat pills, filter, pagination
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
