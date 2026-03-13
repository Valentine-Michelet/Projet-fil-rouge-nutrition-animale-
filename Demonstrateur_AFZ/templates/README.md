# `Demonstrateur_AFZ/templates/` — interface HTML Jinja2

## Rôle

Ce dossier contient les gabarits HTML utilisés par FastAPI via `Jinja2Templates`. Il constitue la couche de présentation de l'application.

## Fichiers

| Fichier | Rôle |
|---|---|
| `base.html` | squelette commun de navigation et de mise en page |
| `index.html` | page d'accueil |
| `predict.html` | formulaire de saisie et affichage des prédictions |
| `model_info.html` | synthèse descriptive du modèle entraîné |
| `diagnostics.html` | affichage des graphiques diagnostiques |

## Dépendances

| Template | Route FastAPI associée |
|---|---|
| `index.html` | `/` |
| `predict.html` | `/predict` |
| `model_info.html` | `/model_info` |
| `diagnostics.html` | `/diagnostics` |

## Flux de rendu

```text
main.py -> Jinja2Templates(directory="templates") -> TemplateResponse(...) -> navigateur
```

## Recommandations

- conserver la compatibilité entre les variables injectées par `main.py` et les placeholders Jinja2 ;
- limiter la logique métier dans les templates ;
- documenter tout ajout de champ formulaire dans `predict.html` et `main.py` simultanément.
