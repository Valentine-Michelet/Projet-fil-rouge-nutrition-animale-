# `data/sources/` — données sources et tables de référence

## Rôle

Ce sous-dossier regroupe les jeux de données amont utilisés pour construire le dataset de travail, annoter les correspondances et définir les règles d'imputation ou d'interprétation.

## Inventaire

| Fichier | Fonction probable dans le pipeline |
|---|---|
| `Donnees_IA_2025.xlsx` | table source principale du projet |
| `Moyenne_Feedtables.csv` | moyennes par produit pour imputation |
| `data_merged_with_ood_classification.xlsx` | version enrichie avec statut IID/OOD |
| `feedtables_definitions_formatted.xlsx` | dictionnaire ou table de définitions |
| `fuzzy_matches_annotated.xlsx` | résultats d'appariement annotés |
| `fuzzy_matches_for_annotation.xlsx` | base de travail avant annotation |

## Dépendances aval

```text
Moyenne_Feedtables.csv ──> src/import_data.py / Demonstrateur_AFZ/models.py
Donnees_IA_2025.xlsx ────> prétraitement et fusion
fuzzy_matches_*.xlsx ────> notebooks d'intégration / annotation
```

## Usage

Les fichiers de `sources/` servent principalement en amont. Ils ne sont généralement pas lus directement dans les scripts d'entraînement final, mais nourrissent la construction de `data/merge/data_merged.csv`.

## Recommandations

- conserver la traçabilité des annotations manuelles ;
- ne pas modifier les fichiers de référence sans documenter l'impact sur la fusion finale ;
- privilégier des exports intermédiaires versionnés plutôt qu'un écrasement silencieux.
