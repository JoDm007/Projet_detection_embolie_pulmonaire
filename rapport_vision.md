# Rapport d'Analyse : Module Vision (`ml-models-library/vision`)

## 1. État Actuel : Prêt pour la Phase 1 (P1-Ready)
Après un audit minutieux du code source, le module `vision` s'avère manifestement **prêt pour la P1**. La base de code est très propre, modulaire et alignée avec les standards de production que requiert une librairie de Machine Learning de haut niveau.

### Points Forts Constatés
- **Architecture de Classe Standardisée :** Héritage rigoureux de `BaseModel` (depuis `core`) respecté sans faille sur tout le spectre de la vision : `ImageClassifier`, `ObjectDetector`, `SemanticSegmenter`, et `VideoClassifier`. L'unification est très bien gérée.
- **Sécurité et Robustesse (Pilier Sécurité) :**
  - Validation des chemins (`_safe_path`, `_safe_load_path`) limitant les erreurs silencieuses ou liées à l'OS.
  - Sauvegarde et chargement hautement sécurisés grâce à l'implémentation de `weights_only=True` par défaut (mitigation des vulnérabilités de désérialisation de `torch.load`).
- **Optimisations orientées Production :**
  - Utilisation de l'Automatic Mixed Precision (AMP) via `torch.amp.autocast` intégrée nativement dans les flux `.predict()`.
  - Gestion très fine du sous-échantillonnage temporel (`temporal_stride`) pour les vidéos, empêchant la surcharge mémoire (OOM).
  - Gestion granulaire du gel des backbones (`freeze_backbone`, `unfreeze_backbone`) pour le transfert d'apprentissage.
  - Injection adaptative du dropout testée et gérée intelligemment en fonction du type de module terminal.

## 2. Inventaire des Modèles Validés
- **Classification d'Images (`classifier.py`) :** Prise en charge universelle (ResNet, EfficientNet, ViT). Bonne gestion en single et multi-label.
- **Détection d'Objets (`detector.py`) :** Unifie Faster R-CNN et RetinaNet. Substitution fine des têtes de prédiction (`FastRCNNPredictor`) pour la gestion sécurisée du nombre de classes (Background INCLUS).
- **Segmentation Sémantique (`segmenter.py`) :** Intégration de DeepLabV3, FCN et U-Net avec une attention particulière à la redéfinition du `aux_classifier` évitant un bug classique de redimensionnement de PyTorch.
- **Classification Vidéo (`video/classifier.py`) :** Backbones 3D robustes (r3d_18, mc3_18) avec forte structuration et validation 5D des batchs de clips temporels.
- **Factory API :** Point d'entrée simple et unifié via `create_vision_model()`.

---

## 3. Recommandations Perceptibles pour la Maintenabilité (P2 / Futur)

> [!TIP]
> Ces recommandations ciblent strictement la maintenabilité sur le long terme. Elles faciliteront le travail de l'équipe lors de l'intégration de nouveaux modèles, ou pendant le debug de production.

### A. Centralisation des valeurs statiques (Magic Numbers)
Actuellement, certaines valeurs codées en dur parsèment les instanciations de têtes de réseaux (ex. les canaux `256` et `512` dans le segmenter, ou les classes par défaut `1000` / `91` / `21`). 
**Actionable :** Isoler ces valeurs dans une instance `vision_constants.py` ou un dictionnaire de configurations de tête (`HEAD_CONFIGS`) pour faciliter la lecture et limiter les regressions.

### B. Couplage des Transforms
On observe la constante `SEGMENTATION_TRANSFORMS` laissée dans `segmenter.py`. L'audit suggère que ce modèle s'attende à une normalisation ImageNet.
**Actionable :** Assurer que les normalisations / pre-processings ne soient définis que dans le module `vision.transforms` et passés par contrat. La classe du modèle devrait se décharger de la connaissance intrinsèque de ces transforms.

### C. Injection de Configuration par `Dataclass`
Les constructeurs ont beaucoup d'arguments (`num_classes`, `architecture`, `dropout`, `freeze_backbone`, `box_score_thresh`, etc.).
**Actionable :** Le passage à une structure `Pydantic` ou `dataclass` (ex: `DetectorConfig`) rendra la création (particulièrement via la `factory`) beaucoup plus digeste et auto-validée côté typage.

### D. Pattern de Registry pour la Factory
Le pattern `create_vision_model` nécessite d'être complété manuellement (modification du dico `builders`).
**Actionable :** Implémenter un décorateur `@register_vision_model("task_name")` pour automatiser l'enregistrement au démarrage, respectant le principe Ouvert/Fermé (Open/Closed Principle de SOLID).

### E. Compatibilité de l'Exportation (ONNX / JIT)
**Actionable :** Penser à ajouter une série de tests automatisés qui vérifieront scrupuleusement si le modèle instancié résiste bien au `torch.jit.script` ou au `.onnx` export. Les comportements qui renvoient des dictionnaires (comme pour `SemanticSegmenter`) posent souvent problème lors des passages en C++ (libtorch). Procéder à un wrapping sous forme de tuple au besoin.
