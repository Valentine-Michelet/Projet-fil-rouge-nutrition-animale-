# `src/` — bibliothèque interne de modélisation et d'analyse

## Finalité

Le dossier `src/` contient les modules Python réutilisables du projet. Il constitue le coeur méthodologique du dépôt : préparation des données, définition des modèles, entraînement profond, expériences XGBoost, analyse de robustesse et visualisation.

## Arbre fonctionnel

```text
src/
├── data.py                  # lecture de fichiers et conversion numérique
├── import_data.py           # chargement du dataset fusionné et séparation X/y
├── features.py              # encodage de variables catégorielles
├── experiments.py           # expériences XGBoost et validation croisée
├── experiments_bruit_val.py # variantes avec bruit / robustesse
├── bruit_val.py             # injection de bruit gaussien
├── models.py                # MLP, auto-encodeur miroir, transformer tabulaire
├── utils_preprocessing.py   # scaling, splits, DataLoaders
├── utils_deep.py            # boucles d'entraînement et d'évaluation PyTorch
├── plot_deep.py             # tracés d'apprentissage profond
└── visu.py                  # visualisations PCA, corrélations, comparaisons
```

## Dépendances internes entre modules

```text
import_data.py ───────────────┐
features.py ──────────────────┤
utils_preprocessing.py ───────┼──> notebooks/Run_*.py
models.py ────────────────────┤
utils_deep.py ────────────────┤
plot_deep.py ─────────────────┤
experiments.py ───────────────┤
experiments_bruit_val.py ─────┘
```

## Description des modules

| Fichier | Rôle | Fonctions / classes principales |
|---|---|---|
| `data.py` | lecture simple de CSV/XLSX et conversion de colonnes | `load_data_excel`, `load_data_csv`, `convert_numeric_columns` |
| `import_data.py` | point d'entrée pour le dataset fusionné | `import_data_merged`, `extracion_X_y`, `sep_extr_X_y`, `import_moyennes` |
| `features.py` | encodage catégoriel | `encode_categorical_feature`, `onehot_encode_categorical_feature` |
| `experiments.py` | pipeline XGBoost classique et validation croisée | `train_xgboost_models`, `train_xgboost_models_cv`, `leave_one_product_out_cv` |
| `experiments_bruit_val.py` | scénarios de robustesse au bruit | fonctions analogues à `experiments.py` |
| `bruit_val.py` | génération de bruit gaussien | `add_gaussian_noise`, `add_noise_after_split` |
| `models.py` | modèles PyTorch | `MLPRegressor`, `MirrorAutoEncoder`, `FrozenEncoder`, `NutritionTransformer` |
| `utils_preprocessing.py` | normalisation et création de DataLoaders | `scale_from_train`, `split_2_DataLoader`, `tensor_2_DataLoader` |
| `utils_deep.py` | entraînement, early stopping, évaluation | `train_model`, `Calcul_evaluation`, `save_weights`, `load_weights` |
| `plot_deep.py` | visualisation d'apprentissage | `plot_loss_values_TV`, `plot_real_vs_pred` |
| `visu.py` | visualisations analytiques et comparatives | `plot_embeddings_3d_pca`, `plot_cv_results_*`, `plot_feature_importance` |

## Flux d'exécution typique

```text
1. import_data.py lit data/merge/data_merged.csv
2. utils_preprocessing.py normalise et sépare train/val/test
3. models.py définit l'architecture
4. utils_deep.py entraîne et évalue
5. visu.py / plot_deep.py synthétisent les résultats
6. reports/ et results/ stockent les sorties finales
```

## Exécution

`src/` n'est pas conçu comme un dossier de scripts autonomes ; il est importé par les notebooks et scripts du dépôt.

| Usage | Exemple |
|---|---|
| Import d'un module | `import src.import_data as id` |
| Chargement des données fusionnées | `df = id.import_data_merged("../data/merge/data_merged.csv")` |
| Chargement de poids PyTorch | `model = ud.load_weights("../models/model_MLP_Ba.pth", model, device)` |

## Dépendances externes

- `numpy`, `pandas`
- `scikit-learn`
- `xgboost`
- `torch`
- `matplotlib`, `seaborn`, `plotly`
- `tqdm`

## Points d'attention

- Certains modules supposent une organisation précise des colonnes dans les fichiers de données.
- Les scripts profonds utilisent des poids pré-entraînés dans `models/`.
- Plusieurs fonctions utilisent des chemins relatifs pensés pour une exécution depuis `notebooks/`.
