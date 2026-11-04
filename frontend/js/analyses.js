// frontend/js/analyses.js — liste des analyses et suivi de leur progression (polling)
import { api } from './api.js';
import { el } from './dom.js';

const POLL_MS = 2000;
const STATUS_LABEL = { queued: 'En attente', running: 'En cours', done: 'Terminée', failed: 'Échec' };

let pollTimer = null;
let selectedId = null;
let handlers = {};

function formatDate(iso) {
  return new Date(iso).toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
}

function renderItem(a) {
  const li = el('li', 'analysis-item');
  li.classList.toggle('selected', a.id === selectedId);
  li.tabIndex = 0;

  const head = el('div', 'row');
  head.append(el('span', 'title', a.source === 'demo' ? `Cas ${a.case_id}` : 'Scanner importé'),
              el('span', `status status-${a.status}`, STATUS_LABEL[a.status]));
  li.append(head);

  const meta = el('div', 'row');
  const owner = a.owner ? ` · ${a.owner}` : '';
  meta.append(el('span', 'meta', `${formatDate(a.created_at)} · ${a.engine}${owner}`));
  if (a.status === 'done' || a.status === 'failed') {
    const del = el('button', 'btn btn-danger-text', 'Supprimer');
    del.type = 'button';
    del.title = 'Effacer immédiatement cette analyse et ses images';
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm('Supprimer définitivement cette analyse ?')) return;
      try {
        await api.deleteAnalysis(a.id);
        if (selectedId === a.id) { selectedId = null; handlers.onDeselect?.(); }
        refresh();
      } catch (err) { alert(err.message); }
    });
    meta.append(del);
  }
  li.append(meta);

  if (a.status === 'running' || a.status === 'queued') {
    li.append(el('div', 'step', a.step || ''));
    const bar = el('div', 'progress');
    const fill = el('div');
    fill.style.width = `${Math.round(a.progress * 100)}%`;
    bar.append(fill);
    li.append(bar);
  } else if (a.status === 'failed') {
    li.append(el('div', 'fail', a.error || 'Échec'));
  } else if (a.summary) {
    const n = a.summary.n_findings;
    li.append(el('div', 'step', `${n} zone${n > 1 ? 's' : ''} retenue${n > 1 ? 's' : ''} sur ${a.summary.n_candidates} candidats`));
  }

  const open = () => {
    if (a.status !== 'done') return;
    selectedId = a.id;
    document.querySelectorAll('.analysis-item').forEach((n) => n.classList.remove('selected'));
    li.classList.add('selected');
    handlers.onSelect?.(a);
  };
  li.addEventListener('click', open);
  li.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(); });
  return li;
}

export async function refresh() {
  clearTimeout(pollTimer);
  let list;
  try {
    list = await api.listAnalyses();
  } catch {
    pollTimer = setTimeout(refresh, POLL_MS * 2);
    return;
  }
  document.getElementById('analysis-list').replaceChildren(...list.map(renderItem));
  document.getElementById('analysis-empty').hidden = list.length > 0;

  // Ouvre automatiquement une analyse qu'on vient de lancer, dès qu'elle est terminée — sauf si
  // une autre est déjà affichée : charger un 2e volume (~1 Go en mémoire avec ses textures) sans
  // action de l'utilisateur peut saturer la RAM. Elle reste alors à un clic dans la liste.
  if (handlers.pendingOpen) {
    const target = list.find((a) => a.id === handlers.pendingOpen);
    if (target && target.status === 'done' && selectedId !== null) {
      handlers.pendingOpen = null;
    } else if (target && target.status === 'done') {
      handlers.pendingOpen = null;
      selectedId = target.id;
      handlers.onSelect?.(target);
      document.getElementById('analysis-list').replaceChildren(...list.map(renderItem));
    } else if (!target || target.status === 'failed') {
      handlers.pendingOpen = null;
    }
  }

  if (list.some((a) => a.status === 'queued' || a.status === 'running')) {
    pollTimer = setTimeout(refresh, POLL_MS);
  }
}

export function initAnalyses({ onSelect, onDeselect }) {
  handlers = { onSelect, onDeselect, pendingOpen: null };
}

export function watch(analysisId) {
  handlers.pendingOpen = analysisId;
  refresh();
}

export function stopPolling() {
  clearTimeout(pollTimer);
  selectedId = null;
}
