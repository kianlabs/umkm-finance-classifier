/**
 * tren.js — Analisis Tren Tahunan (independen, gak ganggu riwayat.js)
 */
let trenChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
  // Intercept loadHistory — after render, compute trend
  const origLoad = loadHistory;
  loadHistory = function() {
    origLoad.apply(this, arguments);
    // We hook into the fetch by patching fetch
  };
});

// Override: setelah allHistory diisi, compute trend
const origRenderTable = renderTable;
renderTable = function(data) {
  origRenderTable.apply(this, arguments);
  setTimeout(() => computeTrend(allHistory), 100);
};

function computeTrend(data) {
  const section = document.getElementById("tren-section");
  if (!data || data.length === 0) {
    section.classList.add("hidden");
    return;
  }

  // Kelompokkan per bulan
  const monthly = {};
  let minDate = null, maxDate = null;

  data.forEach(item => {
    const ts = item.timestamp;
    if (!ts) return;
    const d = new Date(ts);
    if (isNaN(d.getTime())) return;
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
    if (!monthly[key]) monthly[key] = [];
    monthly[key].push(item);
    if (!minDate || d < minDate) minDate = d;
    if (!maxDate || d > maxDate) maxDate = d;
  });

  // Filter 12 bulan terakhir
  let months = Object.keys(monthly).sort();
  if (months.length === 0) {
    section.classList.add("hidden");
    return;
  }

  // Potong ke 12 bulan terakhir
  if (months.length > 12) months = months.slice(-12);

  const labels = [];
  const probAverages = [];
  const sehatCounts = [];
  const tidakCounts = [];
  const totalCounts = [];

  months.forEach(m => {
    const items = monthly[m];
    const total = items.length;
    let sehat = 0, totalProb = 0;
    items.forEach(it => {
      const prob = it.output?.prob_sehat ?? it.prob_sehat ?? 0;
      totalProb += prob;
      if ((it.output?.label ?? it.label) === 1) sehat++;
    });
    labels.push(m);
    probAverages.push(totalProb / total);
    sehatCounts.push(sehat);
    tidakCounts.push(total - sehat);
    totalCounts.push(total);
  });

  // Hitung status kesimpulan
  const totalSehat = sehatCounts.reduce((a, b) => a + b, 0);
  const totalAll = totalCounts.reduce((a, b) => a + b, 0);
  const sehatPct = totalAll > 0 ? totalSehat / totalAll : 0;

  let verdict = "";
  let verdictClass = "";
  let verdictDesc = "";

  if (sehatPct >= 0.7) {
    verdict = "✅ USAHA SEHAT — Stabil Positif";
    verdictClass = "sehat";
    verdictDesc = "Selama setahun terakhir, mayoritas analisis menunjukkan kondisi keuangan yang sehat. "
      + "Pertahankan strategi bisnis Anda dan terus pantau indikator keuangan secara berkala.";
  } else if (sehatPct <= 0.3) {
    verdict = "⚠️ USAHA TIDAK SEHAT — Perlu Perhatian";
    verdictClass = "tidak";
    verdictDesc = "Sebagian besar analisis dalam setahun terakhir menunjukkan kondisi keuangan yang kurang sehat. "
      + "Segera evaluasi strategi bisnis, terutama pada efisiensi pengeluaran dan peningkatan margin keuntungan.";
  } else {
    verdict = "📊 USAHA FLUKTUATIF — Kondisi Bervariasi";
    verdictClass = "fluk";
    verdictDesc = "Kondisi keuangan usaha Anda cenderung berubah-ubah sepanjang tahun. "
      + "Fokus pada stabilitas dengan menjaga margin keuntungan dan mengontrol rasio pengeluaran.";
  }

  // Summary stats
  const avgProb = probAverages.reduce((a, b) => a + b, 0) / probAverages.length;
  const latestProb = probAverages[probAverages.length - 1] || 0;
  const trend = probAverages.length >= 2 && probAverages[probAverages.length - 1] > probAverages[0] ? "naik" :
    probAverages.length >= 2 && probAverages[probAverages.length - 1] < probAverages[0] ? "turun" : "stabil";

  const summaryHtml = `
    <div class="tren-verdict-card ${verdictClass}">
      <div class="verdict-label">🏆 Kesimpulan Tahunan</div>
      <div class="verdict-title">${verdict}</div>
      <div class="verdict-desc">${verdictDesc}</div>
    </div>
    <div class="tren-summary">
      <div class="tren-stat">
        <div class="label">Total Analisis (12 bln)</div>
        <div class="value">${totalAll}</div>
      </div>
      <div class="tren-stat">
        <div class="label">Rata-rata Probabilitas Sehat</div>
        <div class="value ${avgProb >= 0.7 ? 'sehat' : avgProb <= 0.3 ? 'tidak' : 'fluk'}">${(avgProb * 100).toFixed(1)}%</div>
      </div>
      <div class="tren-stat">
        <div class="label">Hasil Terbaru</div>
        <div class="value ${latestProb >= 0.7 ? 'sehat' : latestProb <= 0.3 ? 'tidak' : 'fluk'}">${(latestProb * 100).toFixed(1)}%</div>
        <div class="desc">Bulan ${labels[labels.length - 1]}</div>
      </div>
      <div class="tren-stat">
        <div class="label">Tren Probabilitas</div>
        <div class="value ${trend === 'naik' ? 'sehat' : trend === 'turun' ? 'tidak' : 'fluk'}">
          ${trend === 'naik' ? '📈 Naik' : trend === 'turun' ? '📉 Turun' : '➡️ Stabil'}
        </div>
        <div class="desc">Dari ${labels[0]} ke ${labels[labels.length - 1]}</div>
      </div>
    </div>
  `;

  document.getElementById("trenSummary").innerHTML = summaryHtml;

  // Tabel bulanan
  const tbody = document.getElementById("trenTableBody");
  tbody.innerHTML = months.map((m, i) => {
    const sehat = sehatCounts[i];
    const total = totalCounts[i];
    const tidak = tidakCounts[i];
    const avg = probAverages[i];
    const status = avg >= 0.5 ? "sehat" : "tidak";
    const badge = status === "sehat" ? "✅ Sehat" : "⚠️ Tidak Sehat";
    const badgeClass = status === "sehat" ? "tren-badge-sehat" : "tren-badge-tidak";
    return `<tr>
      <td><strong>${m}</strong></td>
      <td>${total}</td>
      <td style="font-weight:700">${(avg * 100).toFixed(1)}%</td>
      <td style="color:#10b981;font-weight:700">${sehat}</td>
      <td style="color:#ef4444;font-weight:700">${tidak}</td>
      <td><span class="tren-badge ${badgeClass}">${badge}</span></td>
    </tr>`;
  }).join("");

  // Chart
  renderTrenChart(labels, probAverages, sehatCounts, tidakCounts);

  section.classList.remove("hidden");
}

function renderTrenChart(labels, probAvg, sehat, tidak) {
  if (trenChartInstance) trenChartInstance.destroy();

  const ctx = document.getElementById("trenChart").getContext("2d");

  trenChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Probabilitas Sehat (%)",
          data: probAvg.map(v => v * 100),
          borderColor: "#10b981",
          backgroundColor: "rgba(16,185,129,0.1)",
          fill: true,
          tension: 0.4,
          pointBackgroundColor: "#10b981",
          pointBorderColor: "#fff",
          pointBorderWidth: 2,
          pointRadius: 5,
          pointHoverRadius: 7,
          borderWidth: 3,
          yAxisID: "y",
        },
        {
          label: "✅ Sehat",
          data: sehat,
          backgroundColor: "rgba(16,185,129,0.2)",
          borderColor: "rgba(16,185,129,0.7)",
          borderWidth: 2,
          pointRadius: 3,
          type: "bar",
          order: 2,
          yAxisID: "y1",
        },
        {
          label: "⚠️ Tidak Sehat",
          data: tidak,
          backgroundColor: "rgba(239,68,68,0.2)",
          borderColor: "rgba(239,68,68,0.7)",
          borderWidth: 2,
          pointRadius: 3,
          type: "bar",
          order: 2,
          yAxisID: "y1",
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            usePointStyle: true,
            padding: 20,
            font: { size: 12 }
          }
        },
        tooltip: {
          callbacks: {
            label: function(ctx) {
              if (ctx.dataset.label === "Probabilitas Sehat (%)") {
                return ` ${ctx.parsed.y.toFixed(1)}%`;
              }
              return ` ${ctx.dataset.label}: ${ctx.parsed.y}`;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: "Probabilitas Sehat (%)",
            font: { size: 11 }
          },
          grid: { color: "rgba(0,0,0,0.05)" }
        },
        y1: {
          position: "right",
          beginAtZero: true,
          title: {
            display: true,
            text: "Jumlah Analisis",
            font: { size: 11 }
          },
          grid: { display: false }
        }
      }
    }
  });
}
