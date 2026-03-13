# `results/pickle/` — objets Python sérialisés

## Rôle

Ce sous-dossier contient les résultats intermédiaires ou finaux d'expériences sous forme picklée. Il s'agit d'un espace de stockage technique destiné au rechargement rapide de métriques, tableaux ou structures de validation croisée.

## Fichiers présents

| Fichier | Usage attendu |
|---|---|
| `all_cv_results.pkl` | synthèse multi-scénarios |
| `model1_all_chemicals_cv.pkl` | scénario « toutes variables chimiques » |
| `model2_chemicals_nom_cv.pkl` | scénario « chimiques + nom » |
| `model3_nom_only_cv.pkl` | scénario « nom uniquement » |
| `model4_ms_pb_only_cv.pkl` | scénario restreint « MS + PB » |
| `model5_classe_only_cv.pkl` | scénario « classe uniquement » |
| `model6_ms_pb_nom_cv.pkl` | scénario « MS + PB + nom » |

## Chargement

```python
import pickle as pkl
res = pkl.load(open("results/pickle/model1_all_chemicals_cv.pkl", "rb"))
```

## Précautions

- lire ces objets avec une version de Python compatible ;
- considérer le format pickle comme interne au projet, non comme format d'échange externe ;
- documenter dans les notebooks la structure exacte des objets chargés.
