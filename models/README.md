# `models/` — poids pré-entraînés

## Rôle

Ce dossier stocke les poids PyTorch nécessaires aux expériences profondes du projet. Les scripts `notebooks/Run_*.py` s'appuient sur ces fichiers pour initialiser les modèles avant ajustement et évaluation.

## Inventaire

| Fichier | Modèle associé | Chargement |
|---|---|---|
| `model_MLP_Ba.pth` | `src.models.MLPRegressor` | `src.utils_deep.load_weights` |
| `model_auto_Ba.pth` | auto-encodeur miroir / encodeur | non appelé explicitement dans `Run_*.py`, mais cohérent avec `MirrorAutoEncoder` |
| `model_trans_Ba.pth` | `src.models.NutritionTransformer` | `src.utils_deep.load_weights` |

## Dépendances

```text
models/*.pth
    ├──> notebooks/Run_10.py
    ├──> notebooks/Run_10_AE_safe.py
    └──> notebooks/Run_all.py
```

## Commandes associées

| Objectif | Exemple |
|---|---|
| Charger un MLP | `model = ud.load_weights("../models/model_MLP_Ba.pth", model, device)` |
| Charger un transformer | `model = ud.load_weights("../models/model_trans_Ba.pth", model, device)` |

## Points d'attention

- les architectures doivent être strictement compatibles avec les poids sauvegardés ;
- toute modification de `src/models.py` peut invalider la compatibilité des checkpoints ;
- les noms actuels ne portent pas toute la configuration expérimentale : documenter les hyperparamètres dans les rapports ou notebooks.
