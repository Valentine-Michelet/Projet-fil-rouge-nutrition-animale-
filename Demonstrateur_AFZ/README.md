# `Demonstrateur_AFZ/` — démonstrateur applicatif FastAPI

## Objet

`Demonstrateur_AFZ/` est une application web d'inférence destinée à prédire plusieurs variables nutritionnelles à partir d'un formulaire utilisateur. Le démonstrateur encapsule un pipeline d'encodage, d'imputation, d'entraînement XGBoost multi-sorties, de prédiction et d'export.

## Architecture locale

```text
Demonstrateur_AFZ/
├── main.py                 # routes FastAPI et orchestration applicative
├── models.py               # chargement des données, entraînement, prédiction
├── utils.py                # métriques, graphiques, IC, imputation
├── data/                   # données nécessaires au démonstrateur
├── templates/              # interface HTML Jinja2
├── environment_api.yml     # environnement Conda dédié
└── README.md
```

## Dépendances entre fichiers

```text
data/*.csv
   ├──> models.py
   │      ├──> import_model()
   │      └──> predict_from_input()
   ├──> utils.py (IC et imputation)
   └──> main.py via models.py / utils.py

templates/*.html ──> main.py ──> interface web FastAPI
```

## Description des composants

| Fichier | Rôle |
|---|---|
| `main.py` | initialise FastAPI, charge le modèle, expose les routes `/`, `/model_info`, `/predict`, `/diagnostics`, `/export_csv` |
| `models.py` | lit `Donnees_IA_2025.csv`, entraîne un `XGBRegressor`, prépare l'encodeur OneHot et exécute l'inférence |
| `utils.py` | calcule MAE/RMSE/R², génère des graphiques Plotly, gère l'imputation et la sélection des intervalles de confiance |
| `data/` | fournit les CSV nécessaires au démonstrateur |
| `templates/` | définit le rendu HTML de l'application |

## Pipeline applicatif

```text
1. Lecture des données historiques (`data/Donnees_IA_2025.csv`)
2. Encodage des variables catégorielles
3. Séparation train/test
4. Entraînement XGBoost multi-sorties au démarrage
5. Formulaire utilisateur -> nettoyage -> imputation éventuelle
6. Prédiction multi-cible + intervalles de confiance
7. Visualisation / export CSV
```

## Commandes d'installation et d'exécution

| Étape | Commande |
|---|---|
| Créer l'environnement | `cd Demonstrateur_AFZ && conda env create -f environment_api.yml` |
| Activer l'environnement | `conda activate api_ml_env` |
| Lancer l'application | `uvicorn main:app --reload` |
| Ouvrir l'application | navigateur sur `http://127.0.0.1:8000` |

## Routes principales

| Route | Méthode | Fonction |
|---|---|---|
| `/` | GET | page d'accueil |
| `/model_info` | GET | taille des jeux, métriques globales, paramètres du modèle |
| `/predict` | GET/POST | formulaire puis prédiction multi-sortie |
| `/diagnostics` | GET | visualisations des performances |
| `/export_csv` | GET | export de l'historique des prédictions |
| `/retrain` | POST | réentraînement du modèle |

## Points d'attention

- le modèle est réentraîné au chargement de l'application via `load_model()` ;
- les chemins de données sont relatifs au dossier `Demonstrateur_AFZ/` ;
- l'imputation dépend strictement de la présence du produit dans `Moyenne_Feedtables.csv`.
