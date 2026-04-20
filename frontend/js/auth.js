const API = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : window.location.origin;

function getCurrentUser() {
  try { return JSON.parse(localStorage.getItem('user')); } catch { return null; }
}

function isLoggedIn() {
  return getCurrentUser() !== null;
}

function logout() {
  localStorage.removeItem('user');
  window.location.href = 'login.html';
}

async function loginUser(employeeId, password) {
  const res = await fetch(`${API}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ employee_id: employeeId, password }),
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'error'); }
  const data = await res.json();
  localStorage.setItem('user', JSON.stringify(data));
  return data;
}

async function registerUser(employeeId, name, department, password) {
  const res = await fetch(`${API}/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ employee_id: employeeId, name, department, password }),
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'error'); }
  return await res.json();
}

function requireAuth() {
  const user = getCurrentUser();
  if (!user) { window.location.href = 'login.html'; return null; }
  return user;
}
