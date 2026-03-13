# `reports/figures/` — figures prêtes à l'inclusion

## Rôle

Ce sous-dossier contient les visualisations exportées à partir des notebooks et fonctions de visualisation. Les figures sont destinées à l'analyse comparative, à l'interprétation des performances et à l'insertion dans les rapports finaux.

## Typologie des figures observées

| Préfixe / nom | Interprétation probable |
|---|---|
| `ACP_3D.png` | projection PCA / ACP en 3D |
| `AE.png`, `Embedding.png`, `Cosine.png` | analyses liées aux embeddings et à l'auto-encodeur |
| `MLP.png`, `transf.png`, `MLP_vs_trans.png` | comparaison de modèles profonds |
| `OOD_prot.png`, `IID_prot.png` | comparaison IID vs OOD |
| `comparison_*`, `global_r2_*`, `model*_cv_results.png` | synthèse quantitative des performances |

## Dépendances amont

- notebooks analytiques ;
- `src/visu.py` ;
- `src/plot_deep.py`.

## Recommandations

- associer chaque figure à un script/notebook producteur dans le rapport ;
- préciser l'unité, la cohorte et le scénario expérimental dans les légendes ;
- versionner séparément les figures destinées à publication.
