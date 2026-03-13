# `results/` — résultats sérialisés

## Rôle

Le dossier `results/` stocke principalement des artefacts de validation croisée ou de comparaison de scénarios, sérialisés pour être relus ultérieurement sans relancer les expériences.

## Structure

```text
results/
└── pickle/
    ├── all_cv_results.pkl
    ├── model1_all_chemicals_cv.pkl
    ├── model2_chemicals_nom_cv.pkl
    ├── model3_nom_only_cv.pkl
    ├── model4_ms_pb_only_cv.pkl
    ├── model5_classe_only_cv.pkl
    └── model6_ms_pb_nom_cv.pkl
```

## Usage

| Fichier | Interprétation |
|---|---|
| `all_cv_results.pkl` | agrégat global de validation croisée |
| `model*_*.pkl` | résultats par scénario ou sous-ensemble de variables |

## Dépendances aval

| Consommateur probable | Usage |
|---|---|
| notebooks analytiques | comparaison de scénarios |
| `src.visu.py` | génération de figures de synthèse |
| `reports/` | inclusion de graphiques et tableaux finaux |

## Lecture type

```python
import pickle
with open("results/pickle/all_cv_results.pkl", "rb") as f:
    obj = pickle.load(f)
```

## Recommandations

- documenter le schéma exact des objets picklés pour améliorer la réutilisabilité ;
- éviter les modifications manuelles ;
- conserver la correspondance entre noms de fichiers, hypothèses expérimentales et figures produites.
