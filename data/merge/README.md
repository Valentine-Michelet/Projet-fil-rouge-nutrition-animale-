# `data/merge/` — jeu de données fusionné

## Rôle

Ce sous-dossier contient le jeu de données tabulaire consolidé utilisé pour la majorité des expériences. Il sert de point d'entrée à `src/import_data.py`.

## Contenu

| Fichier | Description |
|---|---|
| `data_merged.csv` | table fusionnée incluant variables explicatives, variables cibles, métadonnées produit et informations OOD éventuelles |

## Dépendances

| Consommateur | Usage |
|---|---|
| `src/import_data.py` | chargement principal des données |
| `notebooks/Run_*.py` | séparation X/y et scénarios leave-one-name-out |
| notebooks analytiques | EDA, corrélations, robustesse |

## Chargement type

```python
import src.import_data as id

df = id.import_data_merged("../data/merge/data_merged.csv")
```

## Préconisations

- vérifier la présence et l'ordre attendu des colonnes avant toute expérimentation ;
- aligner strictement ce fichier avec `data/embeddings/embeddings_combined.npy` ;
- conserver un historique des versions si le schéma évolue.
