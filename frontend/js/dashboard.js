const BASE =
  window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : window.location.origin;

const WS_BASE = BASE.replace('https://', 'wss://').replace('http://', 'ws://');

let allWorkers = [];
let allLogs = [];

const _svgWarn = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>`;
const _svgPerson = `<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>`;
const _svgCheck = `<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 12l3 3 5-5"/></svg>`;

async function loadDashboard() {
  await Promise.all([loadWorkers(), loadLogs()]);
}

async function loadWorkers() {
  const grid = document.getElementById('workersGrid');

  try {
    const res = await fetch(`${BASE}/employees`);
    if (!res.ok) {
      throw new Error(`/employees failed with status ${res.status}`);
    }

    const data = await res.json();
    console.log('employees response:', data);

    allWorkers = data.employees || [];
    renderWorkers(allWorkers);

    document.getElementById('statTotal').textContent = allWorkers.length;
    document.getElementById('statActive').textContent = allWorkers.length;
  } catch (err) {
    console.error('loadWorkers error:', err);
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1">${emptyState(_svgPerson, 'فشل تحميل سجل العمال')}</div>`;
    document.getElementById('statTotal').textContent = '0';
    document.getElementById('statActive').textContent = '0';
  }
}

async function loadLogs() {
  const list = document.getElementById('alertsList');

  try {
    const res = await fetch(`${BASE}/logs?limit=50`);
    if (!res.ok) {
      throw new Error(`/logs failed with status ${res.status}`);
    }

    const data = await res.json();
    console.log('logs response:', data);

    allLogs = data.logs || [];
    renderAlerts(allLogs);
    updateViolationStats();
  } catch (err) {
    console.error('loadLogs error:', err);
    list.innerHTML = emptyState(_svgWarn, 'فشل تحميل التنبيهات');
    document.getElementById('statViolations').textContent = '0';
    document.getElementById('statCompliance').textContent = '0%';
  }
}

function renderWorkers(workers) {
  const grid = document.getElementById('workersGrid');

  if (!workers.length) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1">${emptyState(_svgPerson, 'لا يوجد عمال')}</div>`;
    return;
  }

  grid.innerHTML = workers
    .map((w) => {
      const initials = (w.name || '?')
        .split(' ')
        .map((n) => n[0])
        .slice(0, 2)
        .join('');

      return `
        <div class="worker-card">
          <div class="worker-avatar">${initials}</div>
          <div class="worker-name">${w.name || '—'}</div>
          <div class="worker-id">${w.id || w.employee_id || '—'}</div>
          <div class="worker-dept">${w.department || '—'}</div>
          <div class="worker-status active">
            <div class="status-dot active"></div>
            <span>نشط</span>
          </div>
        </div>
      `;
    })
    .join('');
}

function renderAlerts(logs) {
  const list = document.getElementById('alertsList');
  const alerts = logs.filter((l) => l.alert_msg).slice(0, 10);

  if (!alerts.length) {
    list.innerHTML = emptyState(_svgCheck, 'لا توجد تنبيهات');
    return;
  }

  list.innerHTML = `
    <div class="alerts-list">
      ${alerts
        .map(
          (l) => `
        <div class="alert-item">
          <div class="alert-icon">${_svgWarn}</div>
          <div class="alert-content">
            <div class="alert-worker">${l.employee_id || '—'}</div>
            <div class="alert-message">${l.alert_msg || '—'}</div>
          </div>
          <div class="alert-time">${fmtTime(l.timestamp)}</div>
        </div>
      `
        )
        .join('')}
    </div>
  `;
}

function updateViolationStats() {
  const today = new Date().toDateString();
  const todayLogs = allLogs.filter((l) => new Date(l.timestamp).toDateString() === today);
  const violations = todayLogs.filter((l) => !l.ppe_compliant).length;
  const total = todayLogs.length;
  const compliance = total ? Math.round(((total - violations) / total) * 100) : 100;

  document.getElementById('statViolations').textContent = violations;
  document.getElementById('statCompliance').textContent = compliance + '%';
}

function filterWorkers(q) {
  const low = q.toLowerCase();

  renderWorkers(
    allWorkers.filter(
      (w) =>
        (w.name || '').toLowerCase().includes(low) ||
        (w.id || w.employee_id || '').toLowerCase().includes(low) ||
        (w.department || '').toLowerCase().includes(low)
    )
  );
}

function fmtTime(ts) {
  if (!ts) return '—';

  return new Date(ts).toLocaleTimeString(
    currentLang === 'ar' ? 'ar-SA' : 'en-US',
    { hour: '2-digit', minute: '2-digit' }
  );
}

function emptyState(icon, msg) {
  return `<div class="empty-state"><div class="empty-state-icon">${icon}</div><p>${msg}</p></div>`;
}

let ws;

function connectWS() {
  ws = new WebSocket(`${WS_BASE}/ws/stream`);

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.workers) {
      document.getElementById('statActive').textContent = data.workers.length;
    }
  };

  ws.onclose = () => setTimeout(connectWS, 3000);
  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
    ws.close();
  };
}

document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
  connectWS();
  setInterval(loadLogs, 30000);
});