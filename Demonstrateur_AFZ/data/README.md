# `Demonstrateur_AFZ/data/` — données opérationnelles du démonstrateur

## Rôle

Ce sous-dossier contient les données directement consommées par l'application FastAPI. Contrairement à `data/` à la racine du dépôt, il s'agit ici d'un jeu autonome destiné à l'exécution de l'interface web.

## Inventaire

| Fichier | Usage dans le démonstrateur |
|---|---|
| `Donnees_IA_2025.csv` | base d'entraînement du modèle XGBoost |
| `Donnees_IA_2025.xlsx` | version tableur de la base principale |
| `Moyenne_Feedtables.csv` | imputation des variables manquantes par produit |
| `IC_allfeatures.csv` | intervalles de confiance si le formulaire est complet |
| `IC_mspb.csv` | intervalles de confiance en configuration réduite |

## Dépendances

```text
Donnees_IA_2025.csv ──> models.py::import_model()
Moyenne_Feedtables.csv ──> models.py::import_moyennes() et utils.py::impute_missing_with_product_means()
IC_allfeatures.csv / IC_mspb.csv ──> utils.py::load_confidence_intervals()
```

## Bonnes pratiques

- ne pas modifier les noms de colonnes sans répercuter le changement dans `models.py` et `main.py` ;
- vérifier les séparateurs (`;`) et l'encodage avant toute mise à jour ;
- maintenir la cohérence entre les colonnes de données et les champs exposés dans l'interface.
