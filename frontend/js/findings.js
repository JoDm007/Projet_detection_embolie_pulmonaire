// frontend/js/findings.js — résumé de l'analyse et tableau des zones suspectes
import { el } from './dom.js';

function stat(value, label) {
  const box = el('div', 'stat');
  box.append(el('div', 'value', value), el('div', 'label', label));
  return box;
}

const pct = (x) => (x === undefined || x === null ? 'n/a' : `${Math.round(x * 100)} %`);
const volume = (mm3) => (mm3 >= 1000 ? `${(mm3 / 1000).toFixed(2)} cm³` : `${Math.round(mm3)} mm³`);

// onFindingClick(centroid_vox) : appelé au clic sur une ligne (ex. centrer le viewer)
export function renderFindings(analysis, data, { onFindingClick } = {}) {
  document.getElementById('results-card').hidden = false;
  document.getElementById('results-title').textContent =
    analysis.source === 'demo' ? `Résultats — cas ${analysis.case_id}` : 'Résultats — scanner importé';
  document.getElementById('results-model').textContent = data.model_used;

  const total = data.findings.reduce((s, f) => s + f.volume_mm3, 0);
  document.getElementById('results-stats').replaceChildren(
    stat(String(data.findings.length), 'zones retenues'),
    stat(String(data.n_candidates), 'candidats examinés (O2)'),
    stat(data.findings.length ? volume(total) : '—', 'volume cumulé des zones'),
    stat(`${data.shape_zyx[0]} coupes`, `voxel ${data.spacing_zyx[1]}×${data.spacing_zyx[2]}×${data.spacing_zyx[0]} mm`),
  );

  document.getElementById('results-warnings').replaceChildren(...data.warnings.map((w) => el('li', '', w)));

  const body = document.getElementById('findings-body');
  body.replaceChildren(...data.findings.map((f) => {
    const tr = el('tr');
    const score = el('td', '', f.score.toFixed(2));
    const bar = el('span', 'score-bar');
    const fill = el('i');
    fill.style.width = `${Math.round(Math.min(f.score, 1) * 100)}%`;
    bar.append(fill);
    score.append(bar);
    tr.append(
      el('td', '', String(f.id)),
      score,
      el('td', '', f.side),
      el('td', '', `${f.slice_range[0]}–${f.slice_range[1]}`),
      el('td', '', volume(f.volume_mm3)),
      el('td', 'muted', 'voir ›'),
    );
    tr.addEventListener('click', () => {
      body.querySelectorAll('tr').forEach((r) => r.classList.toggle('active', r === tr));
      onFindingClick?.(f.centroid_vox);
    });
    return tr;
  }));
  document.getElementById('findings-empty').hidden = data.findings.length > 0;

  const op = data.operating_point || {};
  document.getElementById('operating-point').textContent =
    `Point de fonctionnement : seuil ${op.threshold?.toFixed?.(3) ?? 'n/a'} (≈ ${op.fp_budget ?? '?'} fausses alertes ` +
    `par patient). Mesuré en validation croisée sur FUMPE : ${pct(op.sens_patients_globale)} des patients porteurs ` +
    `ont au moins une embolie retrouvée. Le côté est déduit de l'orientation axiale standard.`;
}

export function clearFindings() {
  document.getElementById('results-card').hidden = true;
}
