# `data/` — données du projet

## Finalité

Le dossier `data/` regroupe les entrées analytiques du projet : fichiers sources, tables fusionnées et embeddings sémantiques. Il alimente à la fois les scripts de recherche dans `notebooks/` et les modules utilitaires de `src/`.

## Structure

```text
data/
├── embeddings/
│   └── embeddings_combined.npy
├── merge/
│   └── data_merged.csv
└── sources/
    ├── Donnees_IA_2025.xlsx
    ├── Moyenne_Feedtables.csv
    ├── data_merged_with_ood_classification.xlsx
    ├── feedtables_definitions_formatted.xlsx
    ├── fuzzy_matches_annotated.xlsx
    └── fuzzy_matches_for_annotation.xlsx
```

## Dépendances vers le code

| Sous-dossier | Utilisé par | Usage |
|---|---|---|
| `embeddings/` | `notebooks/Run_*.py`, `src/import_data.py` | embeddings sémantiques concaténés |
| `merge/` | `src/import_data.py`, notebooks | dataset tabulaire fusionné pour apprentissage |
| `sources/` | prétraitement, annotation, intégration | données brutes et tables de référence |

## Flux de transformation

```text
sources/ ──> nettoyage / fusion / annotation ──> merge/data_merged.csv
sources/ ──> génération d'embeddings ──────────> embeddings/embeddings_combined.npy
```

## Exécution

Le dossier ne contient pas de script exécutable autonome. Il est consommé par les scripts du dépôt.

| Besoin | Fichier attendu |
|---|---|
| apprentissage tabulaire | `merge/data_merged.csv` |
| apprentissage hybride tabulaire + sémantique | `merge/data_merged.csv` + `embeddings/embeddings_combined.npy` |
| imputation par produit | `sources/Moyenne_Feedtables.csv` |

## Bonnes pratiques

- préserver les encodages et séparateurs d'origine (`;`, virgule décimale) lors de toute modification ;
- documenter toute transformation de `sources/` vers `merge/` ;
- éviter d'écraser les fichiers de référence sans versionnement.
