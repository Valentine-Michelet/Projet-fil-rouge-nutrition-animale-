# `data/embeddings/` — embeddings sémantiques

## Rôle

Ce sous-dossier contient les représentations vectorielles associées aux produits ou descriptions textuelles. Dans l'état actuel du dépôt, le fichier central est `embeddings_combined.npy`.

## Contenu

| Fichier | Description | Consommateurs |
|---|---|---|
| `embeddings_combined.npy` | matrice NumPy des embeddings concaténés | `notebooks/Run_10.py`, `Run_10_AE_safe.py`, `Run_all.py` |

## Usage dans le pipeline

```text
embeddings_combined.npy
    └──> auto-encodeur miroir (`src.models.MirrorAutoEncoder`)
            └──> encodeur gelé (`FrozenEncoder`)
                    └──> concaténation avec variables physico-chimiques
                            └──> régression MLP ou transformer
```

## Chargement

| Langage | Exemple |
|---|---|
| Python | `embedding = np.load("../data/embeddings/embeddings_combined.npy")` |

## Points d'attention

- le nombre de lignes doit rester aligné avec `data/merge/data_merged.csv` ;
- toute régénération des embeddings doit être tracée dans les notebooks correspondants.
