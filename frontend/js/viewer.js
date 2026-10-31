// frontend/js/viewer.js — intégration de NiiVue : CT + zones de l'IA + vérité terrain (démo)
import { api } from './api.js';

// Fenêtres d'affichage en HU : [min, max]
const WINDOWS = {
  vessels: [-250, 450],   // médiastin / vaisseaux opacifiés (lecture d'angioscanner)
  lung: [-1350, 150],     // parenchyme pulmonaire
};
const OPACITY_AI = 0.75;
const OPACITY_GT = 0.45;

let nv = null;
let current = { id: null, hasGt: false };
let windowName = 'vessels';

function el(id) { return document.getElementById(id); }

function sliceTypes() {
  const T = window.niivue.SLICE_TYPE;
  return { axial: T.AXIAL, coronal: T.CORONAL, sagittal: T.SAGITTAL, multi: T.MULTIPLANAR, render: T.RENDER };
}

async function ensureViewer() {
  if (nv) return nv;
  if (!window.niivue) throw new Error('NiiVue introuvable (frontend/vendor/niivue/niivue.umd.js).');
  nv = new window.niivue.Niivue({
    backColor: [0, 0, 0, 1],
    crosshairColor: [0.02, 0.71, 0.83, 0.9],
    show3Dcrosshair: true,
    isRadiologicalConvention: true,   // convention radiologique : droite du patient à gauche de l'écran
    textHeight: 0.03,
  });
  await nv.attachToCanvas(el('viewer-canvas'));
  nv.setSliceType(sliceTypes().multi);
  return nv;
}

export function initViewer() {
  el('layout-buttons').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-layout]');
    if (!btn || !nv) return;
    nv.setSliceType(sliceTypes()[btn.dataset.layout]);
    el('layout-buttons').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b === btn));
  });

  el('window-buttons').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-window]');
    if (!btn) return;
    windowName = btn.dataset.window;
    el('window-buttons').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b === btn));
    applyWindow();
  });

  el('toggle-overlay').addEventListener('change', applyOpacity);
  el('toggle-gt').addEventListener('change', applyOpacity);
}

function applyWindow() {
  if (!nv || !nv.volumes.length) return;
  const [min, max] = WINDOWS[windowName];
  nv.volumes[0].cal_min = min;
  nv.volumes[0].cal_max = max;
  nv.updateGLVolume();
}

function applyOpacity() {
  if (!nv || nv.volumes.length < 2) return;
  nv.setOpacity(1, el('toggle-overlay').checked ? OPACITY_AI : 0);
  if (current.hasGt && nv.volumes.length > 2) nv.setOpacity(2, el('toggle-gt').checked ? OPACITY_GT : 0);
}

export async function showAnalysis(analysisId, hasGroundtruth) {
  el('viewer-placeholder').hidden = true;
  el('viewer-loading').hidden = false;
  el('viewer-loading').textContent = 'Chargement du volume…';
  try {
    const viewer = await ensureViewer();
    const [min, max] = WINDOWS[windowName];
    const volumes = [
      { url: api.fileUrl(analysisId, 'volume.nii.gz'), name: 'volume.nii.gz', colormap: 'gray', cal_min: min, cal_max: max },
      { url: api.fileUrl(analysisId, 'overlay.nii.gz'), name: 'overlay.nii.gz', colormap: 'red', cal_min: 0.5, cal_max: 1, opacity: OPACITY_AI },
    ];
    if (hasGroundtruth) {
      volumes.push({ url: api.fileUrl(analysisId, 'groundtruth.nii.gz'), name: 'groundtruth.nii.gz', colormap: 'green', cal_min: 0.5, cal_max: 1, opacity: 0 });
    }
    await viewer.loadVolumes(volumes);
    current = { id: analysisId, hasGt: hasGroundtruth };
    el('toggle-gt').checked = false;
    el('toggle-gt-wrap').hidden = !hasGroundtruth;
    el('legend-gt').hidden = !hasGroundtruth;
    applyOpacity();
    el('viewer-loading').hidden = true;
  } catch (err) {
    el('viewer-loading').textContent = `Affichage impossible : ${err.message}`;
  }
}

export function goToVoxel([x, y, z]) {
  if (!nv || !nv.volumes.length) return;
  nv.scene.crosshairPos = nv.vox2frac([x, y, z]);
  nv.updateGLVolume();
}

export function clearViewer() {
  current = { id: null, hasGt: false };
  if (nv) { nv.volumes.slice().reverse().forEach((v) => nv.removeVolume(v)); nv.drawScene(); }
  el('viewer-placeholder').hidden = false;
  el('viewer-loading').hidden = true;
  el('toggle-gt-wrap').hidden = true;
  el('legend-gt').hidden = true;
}
