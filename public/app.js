/**
 * AgriSmart AI — Client Application Logic (Vercel Frontend)
 * Handles tab navigation, interactive prediction API calls, leaf disease scanning,
 * and Chart.js metric visualizations.
 */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initPresets();
  initCropForm();
  initLeafScanner();
  initFeatureChart();
  renderHistoryTable();
  initHistoryActions();
});

// ==================== 1. TAB NAVIGATION ====================
function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const targetId = btn.getAttribute('data-target');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
    });
  });
}

// ==================== 2. PRESET HANDLER ====================
function initPresets() {
  const select = document.getElementById('preset-select');
  if (!select) return;

  const presets = {
    custom: { n: 90, p: 42, k: 43, temp: 20.8, hum: 82.0, ph: 6.5, rain: 202.9 },
    paddy: { n: 90, p: 42, k: 43, temp: 24.0, hum: 82.0, ph: 6.5, rain: 240.0 },
    pulse: { n: 40, p: 68, k: 79, temp: 18.5, hum: 17.0, ph: 7.3, rain: 75.0 },
    orchard: { n: 25, p: 130, k: 200, temp: 23.0, hum: 92.0, ph: 6.0, rain: 110.0 },
    coffee: { n: 105, p: 30, k: 32, temp: 24.5, hum: 60.0, ph: 6.8, rain: 160.0 },
  };

  select.addEventListener('change', (e) => {
    const val = e.target.value;
    const data = presets[val] || presets.custom;

    document.getElementById('input-n').value = data.n;
    document.getElementById('input-p').value = data.p;
    document.getElementById('input-k').value = data.k;
    document.getElementById('input-temp').value = data.temp;
    document.getElementById('input-humidity').value = data.hum;
    document.getElementById('input-ph').value = data.ph;
    document.getElementById('input-rainfall').value = data.rain;
  });
}

// ==================== 3. CROP RECOMMENDATION FORM ====================
function initCropForm() {
  const form = document.getElementById('crop-form');
  const btn = document.getElementById('btn-predict-crop');
  const btnText = btn.querySelector('.btn-text');
  const btnSpinner = btn.querySelector('.btn-spinner');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const payload = {
      N: parseFloat(document.getElementById('input-n').value),
      P: parseFloat(document.getElementById('input-p').value),
      K: parseFloat(document.getElementById('input-k').value),
      temperature: parseFloat(document.getElementById('input-temp').value),
      humidity: parseFloat(document.getElementById('input-humidity').value),
      ph: parseFloat(document.getElementById('input-ph').value),
      rainfall: parseFloat(document.getElementById('input-rainfall').value),
    };

    // UI Loading state
    btn.disabled = true;
    btnText.textContent = 'Analyzing Farmland...';
    btnSpinner.classList.remove('hidden');

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate recommendation.');
      }

      displayCropResults(data, payload);
      saveToHistory(data, payload);

    } catch (err) {
      alert(`⚠️ Validation or Prediction Error:\n${err.message}`);
    } finally {
      btn.disabled = false;
      btnText.textContent = '🌿 Recommend Optimal Crop';
      btnSpinner.classList.add('hidden');
    }
  });
}

function displayCropResults(res, inputs) {
  document.getElementById('crop-empty-state').classList.add('hidden');
  document.getElementById('crop-results-active').classList.remove('hidden');

  const profile = res.profile;
  document.getElementById('res-category').textContent = profile.category || 'Agricultural Crop';
  document.getElementById('res-crop-name').textContent = profile.display_name || res.predicted_crop;
  
  const conf = res.confidence_pct || 90.0;
  document.getElementById('res-confidence-text').textContent = `${conf.toFixed(1)}%`;
  document.getElementById('res-confidence-bar').style.width = `${Math.min(100, Math.max(10, conf))}%`;

  // Top candidates
  const listEl = document.getElementById('candidates-list');
  listEl.innerHTML = '';
  if (res.top_candidates && res.top_candidates.length > 0) {
    res.top_candidates.forEach(cand => {
      const row = document.createElement('div');
      row.className = 'candidate-row';
      const cropName = cand.crop.charAt(0).toUpperCase() + cand.crop.slice(1);
      row.innerHTML = `<span>🌾 ${cropName}</span><strong>${cand.confidence_pct}%</strong>`;
      listEl.appendChild(row);
    });
  }

  // Profile data
  document.getElementById('prof-soil').textContent = profile.soil_type || 'Standard arable agricultural soil';
  document.getElementById('prof-water').textContent = profile.water_requirement || 'Follow regional guidelines';
  document.getElementById('prof-season').textContent = `${profile.season || 'Seasonal'} (${profile.growth_duration || 'Duration varies'})`;
  document.getElementById('prof-nutrients').textContent = profile.key_nutrients || 'Balanced NPK';
  document.getElementById('prof-tip').textContent = profile.management_tip || 'Follow standard package of practices.';
}

// ==================== 4. LEAF DISEASE SCANNER ====================
let currentSelectedImageBase64 = null;

function initLeafScanner() {
  const dropZone = document.getElementById('leaf-drop-zone');
  const fileInput = document.getElementById('leaf-file-input');
  const sampleSelect = document.getElementById('leaf-sample-select');
  const scanBtn = document.getElementById('btn-scan-leaf');

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  sampleSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val === 'none') {
      scanBtn.disabled = !currentSelectedImageBase64;
      return;
    }
    // Generate synthetic leaf specimen on canvas for instant testing
    currentSelectedImageBase64 = generateSampleLeafDataUrl(val);
    scanBtn.disabled = false;
  });

  scanBtn.addEventListener('click', async () => {
    if (!currentSelectedImageBase64) return;

    scanBtn.disabled = true;
    const btnText = scanBtn.querySelector('.btn-text');
    const btnSpinner = scanBtn.querySelector('.btn-spinner');
    btnText.textContent = 'Diagnosing Leaf Lesions...';
    btnSpinner.classList.remove('hidden');

    try {
      const response = await fetch('/api/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: currentSelectedImageBase64 }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Diagnostic scan failed.');
      }

      displayDiseaseResults(data);

    } catch (err) {
      alert(`⚠️ Diagnostic Error:\n${err.message}`);
    } finally {
      scanBtn.disabled = false;
      btnText.textContent = '🔍 Scan & Diagnose Leaf';
      btnSpinner.classList.add('hidden');
    }
  });
}

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please upload a valid image file (JPG, PNG, WEBP).');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    currentSelectedImageBase64 = e.target.result;
    document.getElementById('btn-scan-leaf').disabled = false;
    document.getElementById('leaf-sample-select').value = 'none';
  };
  reader.readAsDataURL(file);
}

function displayDiseaseResults(data) {
  document.getElementById('disease-empty-state').classList.add('hidden');
  document.getElementById('disease-results-active').classList.remove('hidden');

  document.getElementById('cv-original-img').src = currentSelectedImageBase64;
  document.getElementById('cv-segmented-img').src = data.segmented_image_data_uri;

  document.getElementById('res-disease-severity').textContent = `Severity: ${data.severity}`;
  document.getElementById('res-disease-name').textContent = data.condition;
  document.getElementById('res-disease-pathogen').textContent = data.pathogen;

  const m = data.metrics;
  document.getElementById('val-healthy-pct').textContent = `${m.healthy_tissue_pct}%`;
  document.getElementById('bar-healthy-pct').style.width = `${m.healthy_tissue_pct}%`;

  document.getElementById('val-necrotic-pct').textContent = `${m.necrotic_lesion_pct}%`;
  document.getElementById('bar-necrotic-pct').style.width = `${m.necrotic_lesion_pct}%`;

  document.getElementById('val-chlorotic-pct').textContent = `${m.chlorotic_yellowing_pct}%`;
  document.getElementById('bar-chlorotic-pct').style.width = `${m.chlorotic_yellowing_pct}%`;

  document.getElementById('res-disease-mgmt').textContent = data.management;
}

// Generate sample leaf data URLs on an in-memory canvas
function generateSampleLeafDataUrl(type) {
  const canvas = document.createElement('canvas');
  canvas.width = 300;
  canvas.height = 300;
  const ctx = canvas.getContext('2d');

  // Background
  ctx.fillStyle = '#f5f5f5';
  ctx.fillRect(0, 0, 300, 300);

  // Draw Leaf shape
  ctx.beginPath();
  ctx.ellipse(150, 150, 70, 120, 0, 0, 2 * Math.PI);

  if (type === 'healthy') {
    ctx.fillStyle = '#2d6a4f';
    ctx.fill();
    // leaf vein
    ctx.strokeStyle = '#40916c';
    ctx.lineWidth = 3;
    ctx.stroke();
  } else if (type === 'blight') {
    ctx.fillStyle = '#40916c';
    ctx.fill();
    // necrotic spots
    const spots = [[120, 130, 20], [175, 160, 24], [140, 185, 16]];
    spots.forEach(([x, y, r]) => {
      // yellow halo
      ctx.beginPath();
      ctx.arc(x, y, r + 7, 0, 2 * Math.PI);
      ctx.fillStyle = '#ffd166';
      ctx.fill();
      // dark brown necrotic center
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle = '#5c2c16';
      ctx.fill();
    });
  } else if (type === 'chlorosis') {
    ctx.fillStyle = '#d4a373';
    ctx.fill();
  }

  return canvas.toDataURL('image/png');
}

// ==================== 5. METRICS FEATURE IMPORTANCE CHART ====================
function initFeatureChart() {
  const canvas = document.getElementById('featureImportanceChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Rainfall', 'Humidity', 'Potassium (K)', 'Phosphorus (P)', 'Nitrogen (N)', 'Temperature', 'Soil pH'],
      datasets: [{
        label: 'Gini Importance (%)',
        data: [23.02, 22.42, 17.54, 15.08, 9.64, 7.24, 5.06],
        backgroundColor: [
          '#1b4332',
          '#2d6a4f',
          '#40916c',
          '#52b788',
          '#74c69d',
          '#95d5b2',
          '#b7e4c7',
        ],
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.parsed.x}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: '#f1f5f9' },
          ticks: { callback: (val) => `${val}%` },
        },
        y: {
          grid: { display: false },
        },
      },
    },
  });
}

// ==================== 6. PREDICTION HISTORY ====================
function getHistory() {
  try {
    return JSON.parse(localStorage.getItem('agrismart_history') || '[]');
  } catch (e) {
    return [];
  }
}

function saveToHistory(res, inputs) {
  const history = getHistory();
  const entry = {
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    crop: res.predicted_crop.charAt(0).toUpperCase() + res.predicted_crop.slice(1),
    confidence: `${res.confidence_pct || 90.0}%`,
    n: inputs.N,
    p: inputs.P,
    k: inputs.K,
    temp: inputs.temperature,
    humidity: inputs.humidity,
    ph: inputs.ph,
    rainfall: inputs.rainfall,
  };
  history.unshift(entry);
  if (history.length > 50) history.pop();
  localStorage.setItem('agrismart_history', JSON.stringify(history));
  renderHistoryTable();
}

function renderHistoryTable() {
  const tbody = document.getElementById('history-tbody');
  if (!tbody) return;

  const history = getHistory();
  if (history.length === 0) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="10">No recommendations generated yet.</td></tr>';
    return;
  }

  tbody.innerHTML = history.map(item => `
    <tr>
      <td>${item.time}</td>
      <td><strong>🌾 ${item.crop}</strong></td>
      <td><span style="color:#2d6a4f; font-weight:700;">${item.confidence}</span></td>
      <td>${item.n}</td>
      <td>${item.p}</td>
      <td>${item.k}</td>
      <td>${item.temp}°C</td>
      <td>${item.humidity}%</td>
      <td>${item.ph}</td>
      <td>${item.rainfall} mm</td>
    </tr>
  `).join('');
}

function initHistoryActions() {
  const exportBtn = document.getElementById('btn-export-history');
  const clearBtn = document.getElementById('btn-clear-history');

  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      const history = getHistory();
      if (history.length === 0) {
        alert('No history records to export.');
        return;
      }

      const headers = ['Time', 'Crop', 'Confidence', 'N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rainfall'];
      const csvRows = [headers.join(',')];
      history.forEach(h => {
        csvRows.push([h.time, h.crop, h.confidence, h.n, h.p, h.k, h.temp, h.humidity, h.ph, h.rainfall].join(','));
      });

      const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'agrismart_recommendations.csv';
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (confirm('Clear all session history records?')) {
        localStorage.removeItem('agrismart_history');
        renderHistoryTable();
      }
    });
  }
}
