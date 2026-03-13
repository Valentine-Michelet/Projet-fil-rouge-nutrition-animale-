# `reports/` — rapports, figures et exports finaux

## Rôle

Le dossier `reports/` rassemble les livrables de synthèse du projet : rapports PDF, figures destinées à la rédaction scientifique et tableaux CSV issus des expériences leave-one-name-out.

## Structure

```text
reports/
├── figures/
├── Fourrages_Support_final.pdf
├── PFR_Fourrage_AFZ.pdf
├── PFR-Final_report.zip
├── results_leave_one_name_out.csv
└── results_leave_one_name_out_AE_froze.csv
```

## Dépendances

| Produit | Source amont |
|---|---|
| fichiers PDF | rédaction scientifique externe + figures du projet |
| `results_leave_one_name_out.csv` | `notebooks/Run_10.py` ou `Run_all.py` |
| `results_leave_one_name_out_AE_froze.csv` | `notebooks/Run_10_AE_safe.py` |
| figures PNG | notebooks et fonctions de `src/visu.py` / `src/plot_deep.py` |

## Commandes utiles

| Objectif | Commande |
|---|---|
| Régénérer le CSV leave-one-name-out | `cd notebooks && python Run_10.py` |
| Régénérer la variante AE gelé | `cd notebooks && python Run_10_AE_safe.py` |

## Usage scientifique

Ce dossier sert de couche de restitution finale. Il doit être cohérent avec les hypothèses, seeds et versions des expériences lancées dans `notebooks/`.

## Bonnes pratiques

- conserver le lien entre figures et scripts producteurs ;
- éviter les renommages ambigus de fichiers finaux ;
- archiver séparément les versions prêtes à diffusion.
