# Scripts de diagnostic (O1/O2)

Outils **reproductibles** pour vérifier et justifier les choix du pipeline de détection classique.
Chaque script est autonome, lit les données du dépôt, et imprime un rapport lisible.
À lancer depuis la **racine du projet**.

> Prérequis : environnement Python du projet actif, `data/raw/` présent, et `data/processed/`
> généré (`python -m src.O1_preprocessing.run_preprocessing`).

---

## 1. `check_gt_alignment.py` — l'orientation du masque GT est-elle correcte ?

Vérifie, pour chaque patient porteur, la fraction de voxels de la vérité terrain (GT) qui tombent
dans **l'air** (HU < −400). Un caillot ne peut pas être dans l'air : un %air élevé = masque mal orienté.

```bash
python scripts/diagnostics/check_gt_alignment.py
```

**A permis de trouver** : le loader transposait le masque en `(2,1,0)` (Y/X inversés) au lieu de
`(2,0,1)`. Après correction + auto-orientation Z : 33/33 porteurs à < 1 % d'air (avant : jusqu'à 97 %).

## 2. `measure_gate.py` — les candidats O2 recouvrent-ils le GT ? (le « gate »)

Pour chaque patient : nombre de candidats, combien touchent le GT (VP), et le % du GT couvert.
C'est le critère qui dit si la détection O2 vise la bonne cible avant de passer au ML.

```bash
python scripts/diagnostics/measure_gate.py
```

**A permis de montrer** : l'O2 d'origine (seuil percentile + plafond de volume 5000) ne recouvrait
presque rien (58 % des porteurs, 0 % de couverture médiane) car le plafond jetait les grosses
composantes posées sur le GT.

## 3. `compare_detection_ab.py` — quelle stratégie de détection ? (A vs B)

Compare, sur un sous-ensemble de patients, deux approches, en mesurant le recouvrement du GT :
- **A** : défaut de remplissage *sombre* dans le lumen (définition clinique du caillot) ;
- **B** : composantes *brillantes* sans plafond de volume (alignées sur le GT FUMPE).

```bash
# Lent (~45 s/patient car il segmente les poumons) :
python scripts/diagnostics/compare_detection_ab.py --patients PAT002 PAT018 PAT005
```

**A permis de décider** : le GT FUMPE annotant le lumen *brillant*, c'est **B** qui recouvre le GT
(6/6 détectés, ~19 % de couverture) et non A (défaut sombre). D'où le mode `detection_mode='bright'`
du détecteur.

## 4. `check_lung_segmentation.py` — la segmentation des poumons est-elle correcte ?

Pour chaque patient, affiche le % du volume couvert par le masque pulmonaire (attendu : ~15-30 %).
Un %volume trop élevé = sur-segmentation (médiastin/cœur inclus), qui noie la détection de faux positifs.

```bash
python scripts/diagnostics/check_lung_segmentation.py --patients PAT018 PAT005 PAT002
```

**Contexte** : l'ancienne version (enveloppe convexe) sur-segmentait (PAT018 = 61 % du volume).
Corrigée en remplissant les trous internes au lieu de l'enveloppe convexe.

## 5. `sweep_vessel_threshold.py` — quel seuil `vessel_min_hu` ?

Segmente les poumons **une seule fois** par patient, puis teste plusieurs seuils (30/50/70/100 HU)
en mesurant détection, couverture GT et faux positifs. Évite de régénérer tous les candidats (~40 min)
à chaque essai de seuil.

```bash
python scripts/diagnostics/sweep_vessel_threshold.py
```

**Pourquoi** : 7 des 9 porteurs ratés ont un GT de médiane HU < 100 (le caillot sombre tire la
médiane vers le bas) ; un seuil plus bas devrait les récupérer, au prix de plus de faux positifs.
Ce script quantifie le compromis pour choisir la valeur avant la régénération finale.
