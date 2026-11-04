// frontend/js/viewer.js — intégration de NiiVue : CT + zones de l'IA + vérité terrain (démo)
//
// Performance : un volume CT fait ~200 Mo en mémoire. Chaque updateGLVolume() le renvoie au GPU,
// donc on ne l'appelle que quand les données affichées changent (fenêtre, opacité), jamais pour
// déplacer le curseur. La vérité terrain n'est chargée qu'au premier affichage.
import { api } from './api.js';

// Fenêtres d'affichage en HU : [min, max]
const WINDOWS = {
  vessels: [-250, 450],   // médiastin / vaisseaux opacifiés (lecture d'angioscanner)
  lung: [-1350, 150],     // parenchyme pulmonaire
};
const OPACITY_AI = 0.75;
const OPACITY_GT = 0.45;
// Ordre des volumes dans nv.volumes
const VOL = { ct: 0, ai: 1, gt: 2 };

let nv = null;
let current = { id: null, hasGt: false, gtLoaded: false };
let windowName = 'vessels';
// Les chargements NiiVue sont mis en file : deux loadVolumes concurrents mélangeraient leurs
// volumes. loadToken invalide un chargement devenu obsolète (autre analyse cliquée entre-temps).
let queue = Promise.resolve();
let loadToken = 0;

function el(id) { return document.getElementById(id); }

function enqueue(task) {
  queue = queue.then(task, task);
  return queue;
}

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
    // Pas de tuile 3D dans la vue « 3 plans » : le rendu volumique est recalculé à chaque
    // mouvement de souris et rend l'interface lente. Il reste disponible via le bouton « 3D ».
    multiplanarShowRender: window.niivue.SHOW_RENDER.NEVER,
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
  el('toggle-gt').addEventListener('change', onToggleGt);
}

function applyWindow() {
  if (!nv || !nv.volumes.length) return;
  const [min, max] = WINDOWS[windowName];
  nv.volumes[VOL.ct].cal_min = min;
  nv.volumes[VOL.ct].cal_max = max;
  nv.updateGLVolume();
}

// Opacités de tous les calques puis UN seul envoi au GPU (nv.setOpacity en ferait un par calque)
function applyOpacity() {
  if (!nv || nv.volumes.length <= VOL.ai) return;
  nv.volumes[VOL.ai].opacity = el('toggle-overlay').checked ? OPACITY_AI : 0;
  if (nv.volumes.length > VOL.gt) nv.volumes[VOL.gt].opacity = el('toggle-gt').checked ? OPACITY_GT : 0;
  nv.updateGLVolume();
}

function onToggleGt() {
  if (!current.hasGt || current.gtLoaded || !el('toggle-gt').checked) {
    applyOpacity();
    return;
  }
  const token = loadToken;
  const analysisId = current.id;
  el('toggle-gt').disabled = true;
  enqueue(async () => {
    if (token !== loadToken) return;
    try {
      await nv.addVolumeFromUrl({
        url: api.fileUrl(analysisId, 'groundtruth.nii.gz'), name: 'groundtruth.nii.gz',
        colormap: 'green', cal_min: 0.5, cal_max: 1, opacity: OPACITY_GT,
      });
      if (token !== loadToken) return;
      current.gtLoaded = true;
      applyOpacity();   // l'utilisateur a pu décocher pendant le chargement
    } catch (err) {
      el('toggle-gt').checked = false;
      alert(`Vérité terrain indisponible : ${err.message}`);
    } finally {
      el('toggle-gt').disabled = false;
    }
  });
}

export function showAnalysis(analysisId, hasGroundtruth) {
  const token = ++loadToken;
  el('viewer-placeholder').hidden = true;
  el('toggle-gt').checked = false;
  el('toggle-gt-wrap').hidden = !hasGroundtruth;
  el('legend-gt').hidden = !hasGroundtruth;

  return enqueue(async () => {
    if (token !== loadToken) return;
    // Même analyse déjà affichée : rien à recharger (les fichiers ne changent jamais)
    if (current.id === analysisId && nv && nv.volumes.length) {
      current.hasGt = hasGroundtruth;
      applyOpacity();
      return;
    }
    el('viewer-loading').hidden = false;
    el('viewer-loading').textContent = 'Chargement du volume…';
    try {
      const viewer = await ensureViewer();
      const [min, max] = WINDOWS[windowName];
      await viewer.loadVolumes([
        { url: api.fileUrl(analysisId, 'volume.nii'), name: 'volume.nii', colormap: 'gray', cal_min: min, cal_max: max },
        { url: api.fileUrl(analysisId, 'overlay.nii.gz'), name: 'overlay.nii.gz', colormap: 'red', cal_min: 0.5, cal_max: 1,
          opacity: el('toggle-overlay').checked ? OPACITY_AI : 0 },
      ]);
      if (token !== loadToken) return;
      current = { id: analysisId, hasGt: hasGroundtruth, gtLoaded: false };
      el('viewer-loading').hidden = true;
    } catch (err) {
      if (token !== loadToken) return;
      current = { id: null, hasGt: false, gtLoaded: false };
      el('viewer-loading').textContent = `Affichage impossible : ${err.message}`;
    }
  });
}

export function goToVoxel([x, y, z]) {
  if (!nv || !nv.volumes.length) return;
  nv.scene.crosshairPos = nv.vox2frac([x, y, z]);
  nv.createOnLocationChange();
  nv.drawScene();   // simple redessin : les textures GPU sont déjà à jour
}

export function clearViewer() {
  loadToken++;
  current = { id: null, hasGt: false, gtLoaded: false };
  enqueue(() => {
    if (nv) { nv.volumes.slice().reverse().forEach((v) => nv.removeVolume(v)); nv.drawScene(); }
  });
  el('viewer-placeholder').hidden = false;
  el('viewer-loading').hidden = true;
  el('toggle-gt-wrap').hidden = true;
  el('legend-gt').hidden = true;
}
