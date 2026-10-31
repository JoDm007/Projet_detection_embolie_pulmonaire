// frontend/js/main.js — démarrage et navigation entre l'écran de connexion et l'application
import { api } from './api.js';
import { initAnalyses, refresh, stopPolling, watch } from './analyses.js';
import { initLogin, initLogout } from './auth.js';
import { clearFindings, renderFindings } from './findings.js';
import { initSubmit, loadSubmitOptions } from './upload.js';
import { clearViewer, initViewer, showAnalysis } from './viewer.js';

const loginView = document.getElementById('view-login');
const appView = document.getElementById('view-app');

function showLogin() {
  stopPolling();
  clearFindings();
  clearViewer();
  appView.hidden = true;
  loginView.hidden = false;
  loginView.querySelector('input[name=username]').focus();
}

async function showApp(user) {
  document.getElementById('user-name').textContent = `${user.username} (${user.role})`;
  loginView.hidden = true;
  appView.hidden = false;
  try {
    await loadSubmitOptions();
  } catch (err) {
    const error = document.getElementById('submit-error');
    error.textContent = err.message;
    error.hidden = false;
  }
  refresh();
}

async function openAnalysis(analysis) {
  try {
    const data = await api.findings(analysis.id);
    renderFindings(analysis, data);
    await showAnalysis(analysis.id, data.has_groundtruth);
  } catch (err) {
    alert(err.message);
  }
}

function closeAnalysis() {
  clearFindings();
  clearViewer();
}

window.addEventListener('auth:expired', showLogin);

document.addEventListener('DOMContentLoaded', async () => {
  initLogin(showApp);
  initLogout(showLogin);
  initViewer();
  initAnalyses({ onSelect: openAnalysis, onDeselect: closeAnalysis });
  initSubmit({ onSubmitted: watch });

  try {
    showApp(await api.me());
  } catch {
    showLogin();
  }
});
