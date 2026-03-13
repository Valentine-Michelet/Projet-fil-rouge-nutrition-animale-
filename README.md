# Projet fil rouge — nutrition animale

## Objet

Ce dépôt regroupe un ensemble de scripts, notebooks, modèles entraînés, résultats expérimentaux et un démonstrateur web dédiés à la prédiction de variables nutritionnelles animales à partir de descripteurs physico-chimiques, de métadonnées produit et d'embeddings sémantiques.

Le projet s'organise autour de deux axes complémentaires :

1. **un socle expérimental et méthodologique** dans `src/`, `notebooks/`, `data/`, `models/`, `results/` et `reports/` ;
2. **un démonstrateur applicatif** dans `Demonstrateur_AFZ/` pour l'inférence interactive via FastAPI.

## Vue d'ensemble de l'architecture

```text
Projet-fil-rouge-nutrition-animale-/
├── Demonstrateur_AFZ/        # application FastAPI pour la prédiction
├── data/                     # données sources, fusionnées et embeddings
├── models/                   # poids pré-entraînés PyTorch
├── notebooks/                # scripts et notebooks d'expérimentation
├── reports/                  # rapports, figures et exports d'évaluation
├── results/                  # résultats sérialisés des expériences classiques
└── src/                      # modules Python réutilisables
```

## Relations entre les dossiers

```text
[data/sources] ──┐
                 ├──> [data/merge] ──> [src/import_data.py] ──┐
[data/embeddings]┘                                             │
                                                               ├──> [notebooks/*.ipynb, Run_*.py]
[src/models.py] <────────────── [models/*.pth] ────────────────┤
[src/utils_*.py] <──────────────────────────────────────────────┤
                                                               └──> [reports/, results/]

[Demonstrateur_AFZ/data] ──> [Demonstrateur_AFZ/models.py] ──> [Demonstrateur_AFZ/main.py]
[Demonstrateur_AFZ/templates] ────────────────────────────────┘
```

## Dossiers et rôle scientifique

| Dossier | Rôle principal | Point d'entrée typique |
|---|---|---|
| `src/` | bibliothèque interne de préparation, modélisation et visualisation | import Python depuis notebooks/scripts |
| `notebooks/` | expérimentation, validation, comparaison de scénarios | `python notebooks/Run_all.py` ou Jupyter |
| `data/` | stockage des jeux de données, embeddings et jeux fusionnés | lecture par `src/import_data.py` |
| `models/` | poids de réseaux neuronaux pré-entraînés | chargement via `src/utils_deep.py` |
| `results/` | objets picklés de validation croisée | lecture pour post-traitement |
| `reports/` | rapports PDF, figures et CSV de résultats | consultation / inclusion dans manuscrit |
| `Demonstrateur_AFZ/` | application web de prédiction | `uvicorn main:app --reload` |

## Exécution : scénarios usuels

| Objectif | Commande | Pré-requis |
|---|---|---|
| Lancer le démonstrateur web | `cd Demonstrateur_AFZ && conda env create -f environment_api.yml && conda activate api_ml_env && uvicorn main:app --reload` | données présentes dans `Demonstrateur_AFZ/data/` |
| Exécuter une campagne leave-one-name-out restreinte | `python notebooks/Run_10.py` | lancer depuis `notebooks/` ou adapter les chemins relatifs |
| Exécuter une campagne leave-one-name-out étendue | `python notebooks/Run_all.py` | GPU conseillé pour la partie auto-encodeur |
| Explorer le pipeline pas à pas | ouvrir les notebooks `.ipynb` dans Jupyter | environnement Python scientifique |

## Dépendances logicielles

Les dépendances observées dans le dépôt sont principalement :

- `python>=3.11`
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`
- `xgboost`
- `torch`
- `fastapi`, `uvicorn`, `jinja2`
- `pyyaml`, `tqdm`

Pour le démonstrateur, l'environnement Conda le plus explicite est fourni dans `Demonstrateur_AFZ/environment_api.yml`.

## Conseils d'utilisation

1. Conserver la hiérarchie relative des dossiers, car plusieurs scripts utilisent des chemins relatifs (`../data/...`, `data/...`).
2. Distinguer les **artefacts de recherche** (`src/`, `notebooks/`, `reports/`) du **pipeline d'inférence** (`Demonstrateur_AFZ/`).
3. Vérifier avant exécution que les fichiers volumineux (`.npy`, `.pth`, `.csv`, `.xlsx`) sont présents localement.
4. En cas de reproduction scientifique, consigner la configuration matérielle, la seed et la version des dépendances.
