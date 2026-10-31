// frontend/js/upload.js — lancer une analyse : cas FUMPE ou archive DICOM importée
import { api } from './api.js';

const MAX_UPLOAD_MB = 1024;

export function initSubmit({ onSubmitted }) {
  const error = document.getElementById('submit-error');
  const engineSelect = document.getElementById('engine-select');
  const showError = (msg) => { error.textContent = msg; error.hidden = !msg; };

  // Onglets
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
      document.querySelectorAll('.tab-panel').forEach((p) => { p.hidden = p.dataset.panel !== tab.dataset.tab; });
      showError('');
    });
  });

  // Cas de démo
  const runBtn = document.getElementById('demo-run');
  runBtn.addEventListener('click', async () => {
    const caseId = document.getElementById('demo-select').value;
    if (!caseId) return;
    showError('');
    runBtn.disabled = true;
    try {
      const { analysis_id } = await api.submitDemo(caseId, engineSelect.value);
      onSubmitted(analysis_id);
    } catch (err) {
      showError(err.message);
    } finally {
      runBtn.disabled = false;
    }
  });

  // Import d'archive
  const dropzone = document.getElementById('dropzone');
  const input = document.getElementById('upload-input');

  async function send(file) {
    showError('');
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.zip')) return showError('Une archive .zip est attendue.');
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return showError(`Archive trop volumineuse (maximum ${MAX_UPLOAD_MB} Mo).`);
    dropzone.classList.add('dragover');
    try {
      const { analysis_id } = await api.submitUpload(file, engineSelect.value);
      onSubmitted(analysis_id);
    } catch (err) {
      showError(err.message);
    } finally {
      dropzone.classList.remove('dragover');
      input.value = '';
    }
  }

  input.addEventListener('change', () => send(input.files[0]));
  dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); });
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    send(e.dataTransfer.files[0]);
  });
}

export async function loadSubmitOptions() {
  const [health, cases] = await Promise.all([api.health(), api.demoCases()]);

  const engineSelect = document.getElementById('engine-select');
  engineSelect.replaceChildren(...health.engines.map((e) => {
    const opt = new Option(e.available ? e.label : `${e.label} — indisponible`, e.name);
    opt.disabled = !e.available;
    return opt;
  }));
  const firstAvailable = health.engines.find((e) => e.available);
  if (firstAvailable) engineSelect.value = firstAvailable.name;

  const status = document.getElementById('engine-status');
  status.replaceChildren(...health.engines.map((e) => {
    const pill = document.createElement('span');
    pill.className = `pill ${e.available ? 'ok' : 'off'}`;
    pill.textContent = `${e.name} : ${e.available ? 'prêt' : 'indisponible'}`;
    if (e.reason) pill.title = e.reason;
    return pill;
  }));

  const select = document.getElementById('demo-select');
  select.replaceChildren(...cases.map((c) => new Option(c.case_id, c.case_id)));
  document.getElementById('demo-run').disabled = cases.length === 0;
  if (cases.length === 0) select.replaceChildren(new Option('Aucun cas FUMPE dans data/raw', ''));
}
