# MiniML – Application de prédiction nutritionnelle

MiniML est une application web développée avec **FastAPI** permettant d'entraîner un modèle de Machine Learning et de prédire plusieurs indicateurs nutritionnels à partir de caractéristiques physico-chimiques de matières premières agricoles.

L’application intègre un pipeline complet de Data Science :  
prétraitement des données, encodage, modélisation multi-sorties, estimation d’intervalles de confiance, visualisation des performances et export des résultats.

---

## Fonctionnalités

- Modèle **XGBoost multi-sorties** (11 indicateurs nutritionnels prédits simultanément)
- Encodage automatique des variables catégorielles (OneHotEncoder avec gestion des catégories inconnues)
- Imputation automatique des valeurs manquantes via moyennes Feedtables
- Sélection dynamique des intervalles de confiance selon la complétude des données
- Interface web FastAPI + Bootstrap
- Historique intégré des prédictions (inputs + outputs)
- Indicateur d’imputation par prédiction
- Export CSV complet (inputs + outputs + indicateur d’imputation)
- Visualisation des performances du modèle (MAE, RMSE, R²)
- Réentraînement dynamique du modèle

---

## Pipeline Machine Learning

1. Nettoyage et normalisation des données
2. Encodage des variables catégorielles (OneHotEncoder)
3. Séparation train/test
4. Entraînement XGBoost multi-sorties
5. Évaluation (MAE, RMSE, R²)
6. Déploiement via FastAPI

---

## Indicateurs prédits

Le modèle prédit notamment :

- EB (kcal/kg brut)
- ED porc croissance (kcal/kg brut)
- EM porc croissance (kcal/kg brut)
- EN porc croissance (kcal/kg brut)
- EMAn coq (kcal/kg brut)
- EMAn poulet (kcal/kg brut)
- UFL 2018 par kg brut
- UFV 2018 par kg brut
- PDIA 2018 g/kg brut
- PDI 2018 g/kg brut
- BalProRu 2018 g/kg brut

---

## Structure du projet


├── main.py # Application FastAPI

├── models.py # Entraînement et prédiction

├── utils.py # Fonctions utilitaires (imputation, IC, métriques)

├── data/

│ ├── Donnees_IA_2025.csv

│ ├── Moyenne_Feedtables.csv

│ ├── IC_allfeatures.csv

│ └── IC_mspb.csv

├── templates/ # Templates HTML (Bootstrap)

├── environment_api.yml # Environnement Conda

└── README.md


---

## Installation

Créer un environnement :

```bash
conda env create -f environment_api.yml
```

## Lancement de l'application :

Il est important de noter qu'une fois l'environnement est installé sur votre machin, il n'est plus nécessaire de le recréer.
Il suffit juste de l'activer avec la commande : 

```bash
conda activate api_ml_env
```
 
Puis lancer l'application avec la commande : 
A chaque lancement, un nouvel model est entrainé ici.

```bash
uvicorn main:app --reload
```



