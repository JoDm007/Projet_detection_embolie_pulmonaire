# Frontend PE-Seg AI

Interface web statique : HTML + CSS + modules ES natifs, **sans étape de build et sans Internet**.
Servie par le backend à la même origine (`http://127.0.0.1:8000/`).

## Modules

| Fichier | Rôle |
|---|---|
| `index.html` | coquille : écran de connexion + application |
| `js/main.js` | démarrage, bascule connexion ↔ application |
| `js/api.js` | **seul** module qui appelle le backend (`fetch /api/...`, en-tête anti-CSRF) |
| `js/auth.js` | connexion / déconnexion |
| `js/upload.js` | lancer une analyse : cas FUMPE ou archive `.zip` DICOM ; état des moteurs |
| `js/analyses.js` | liste des analyses, suivi de la progression (polling 2 s), suppression |
| `js/viewer.js` | NiiVue : CT + zones IA (rouge) + vérité terrain FUMPE (vert, démo) |
| `js/findings.js` | résumé et tableau des zones ; clic → saut à la zone dans le viewer |
| `css/base.css` · `layout.css` · `viewer.css` | couleurs, grille des écrans, viewer |
| `vendor/niivue/` | NiiVue 0.69.0, copie locale (voir son README) |

## Règles

- Aucune ressource externe (CDN, polices en ligne) : la CSP du backend les bloquerait de toute façon.
- Les données affichées passent par `textContent`, jamais par `innerHTML` (pas d'injection HTML).
- Ajouter un écran = un nouveau module dans `js/` + ses appels dans `api.js`.
