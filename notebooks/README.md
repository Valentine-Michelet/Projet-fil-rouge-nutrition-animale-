# `notebooks/` — expérimentations, analyses et campagnes de test

## Finalité

Le dossier `notebooks/` rassemble les notebooks exploratoires et les scripts d'orchestration des expériences. Il constitue l'interface principale pour reproduire les analyses, comparer les modèles et générer les sorties déposées dans `reports/` et `results/`.

## Contenu

```text
notebooks/
├── EDA.ipynb
├── Analyse_data_correlations.ipynb
├── data_integration_annotation.ipynb
├── embeddings_generation.ipynb
├── embedding_similarity.ipynb
├── id_regressor*.ipynb
├── ood_*.ipynb
├── xgboost_ood_test_by_cosine_similarity.ipynb
├── Run_10.py
├── Run_10_AE_safe.py
└── Run_all.py
```

## Rôle des scripts principaux

| Script | Objectif | Dépendances directes | Sortie principale |
|---|---|---|---|
| `Run_10.py` | campagne leave-one-name-out sur une liste restreinte de produits | `src.import_data`, `src.models`, `src.utils_deep`, `src.utils_preprocessing`, `models/*.pth`, `data/*` | `reports/results_leave_one_name_out.csv` |
| `Run_10_AE_safe.py` | variante où l'auto-encodeur est entraîné une seule fois puis gelé | mêmes dépendances que `Run_10.py` | `reports/results_leave_one_name_out_AE_froze.csv` |
| `Run_all.py` | campagne étendue sur une grande liste de produits OOD | mêmes dépendances | `reports/results_leave_one_name_out.csv` |

## Graphe de dépendances

```text
notebooks/Run_*.py
├── import src.import_data
├── import src.models
├── import src.utils_preprocessing
├── import src.utils_deep
├── import src.plot_deep
├── lit ../data/merge/data_merged.csv
├── lit ../data/embeddings/embeddings_combined.npy
├── charge ../models/model_MLP_Ba.pth
├── charge ../models/model_trans_Ba.pth
└── écrit ../reports/*.csv
```

## Commandes d'exécution

Les scripts `Run_*.py` utilisent `sys.path.insert(0, '../')` et des chemins relatifs du type `../data/...`. Il est donc recommandé de les lancer **depuis le dossier `notebooks/`**.

| Objectif | Commande |
|---|---|
| Exécuter la campagne courte | `cd notebooks && python Run_10.py` |
| Exécuter la campagne avec auto-encodeur gelé | `cd notebooks && python Run_10_AE_safe.py` |
| Exécuter la campagne complète | `cd notebooks && python Run_all.py` |
| Ouvrir un notebook | `jupyter lab` puis ouvrir le fichier `.ipynb` |

## Dépendances scientifiques

- embeddings sémantiques : `../data/embeddings/embeddings_combined.npy`
- données fusionnées : `../data/merge/data_merged.csv`
- poids profonds : `../models/model_MLP_Ba.pth`, `../models/model_trans_Ba.pth`
- fonctions internes : `src/`

## Recommandations méthodologiques

1. Lancer d'abord `EDA.ipynb` et `Analyse_data_correlations.ipynb` pour vérifier la cohérence des données.
2. Utiliser `Run_10_AE_safe.py` comme compromis reproductible lorsque le coût de réentraînement de l'auto-encodeur est limitant.
3. Conserver les sorties CSV produites dans `reports/` pour le traçage des expériences.
4. Pour des exécutions longues, journaliser la seed, la configuration GPU/CPU et la version des dépendances.
