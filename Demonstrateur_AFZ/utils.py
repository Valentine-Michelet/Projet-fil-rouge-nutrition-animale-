import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _fig_to_div(fig, include_plotlyjs: bool) -> str:
    """
    Retourne un <div> HTML Plotly (sans page complète).
    - include_plotlyjs=True : inclut le JS Plotly (à faire une seule fois par page)
    - include_plotlyjs=False : réutilise le JS déjà chargé
    """
    return fig.to_html(
        full_html=False,
        include_plotlyjs=("cdn" if include_plotlyjs else False)
    )


def compute_metrics_per_target(y_true, y_pred, target_names):
    """
    y_true, y_pred: array-like (n_samples, n_targets)
    Retourne:
      - metrics_rows: list[dict] par cible
      - global_metrics: dict (moyenne sur cibles)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rows = []
    maes, rmses, r2s = [], [], []

    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mae_i = float(mean_absolute_error(yt, yp))
        rmse_i = float(np.sqrt(mean_squared_error(yt, yp)))
        r2_i = float(r2_score(yt, yp))

        rows.append({
            "target": str(name),
            "mae": mae_i,
            "rmse": rmse_i,
            "r2": r2_i
        })

        maes.append(mae_i)
        rmses.append(rmse_i)
        r2s.append(r2_i)

    global_metrics = {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "r2": float(np.mean(r2s))
    }

    return rows, global_metrics


def make_real_vs_pred_div(y_true, y_pred, target_names, ncols=4, include_js=False) -> str:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_targets = y_true.shape[1]
    nrows = int(np.ceil(n_targets / ncols))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[str(v)[:22] for v in target_names]
    )

    for i, _ in enumerate(target_names):
        row = i // ncols + 1
        col = i % ncols + 1

        x = y_true[:, i]
        y = y_pred[:, i]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=4, opacity=0.6),
                showlegend=False
            ),
            row=row,
            col=col
        )

        min_val = float(min(np.min(x), np.min(y)))
        max_val = float(max(np.max(x), np.max(y)))

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(dash="dash", color="red"),
                showlegend=False
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        height=320 * nrows,
        title="Predicted vs True (échantillon)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return _fig_to_div(fig, include_plotlyjs=include_js)


def make_metrics_bar_div(metrics_rows, include_js=False) -> str:
    targets = [r["target"][:22] for r in metrics_rows]
    mae = [r["mae"] for r in metrics_rows]
    rmse = [r["rmse"] for r in metrics_rows]
    r2 = [r["r2"] for r in metrics_rows]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["MAE (par cible)", "RMSE (par cible)", "R² (par cible)"]
    )

    fig.add_trace(go.Bar(x=targets, y=mae, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=targets, y=rmse, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=targets, y=r2, showlegend=False), row=1, col=3)

    fig.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=120)
    )
    fig.update_xaxes(tickangle=45)

    return _fig_to_div(fig, include_plotlyjs=include_js)


def make_residuals_hist_div(y_true, y_pred, include_js=False) -> str:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = (y_pred - y_true).ravel()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=60))
    fig.update_layout(
        title="Histogramme des résidus (toutes cibles concaténées)",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return _fig_to_div(fig, include_plotlyjs=include_js)


def determine_configuration(form_data: dict) -> str:
    PHYSICO_CHEM_COLUMNS = [
        "MS % brut", "PB % brut", "CB % brut", "MGR % brut", "MM % brut",
        "NDF % brut", "ADF % brut", "Lignine % brut", "Amidon % brut", "Sucres % brut"
    ]
    # Vérifie si MS et PB sont présents
    has_ms_pb = ("MS" in form_data) and ("PB" in form_data)

    # Vérifie si toutes les colonnes physico-chimiques sont présentes
    has_all_physico_chem = all(col in form_data for col in PHYSICO_CHEM_COLUMNS)

    if has_all_physico_chem:
        return "full"
    elif has_ms_pb:
        return "ms_pb"
    else:
        # Cas par défaut (ou lever une erreur si nécessaire)
        return "ms_pb"

def load_confidence_intervals(configuration: str) -> pd.DataFrame:
    if configuration == "full":
        return pd.read_csv("data/IC_allfeatures.csv")
    elif configuration == "ms_pb":
        return pd.read_csv("data/IC_mspb.csv")
    else:
        raise ValueError("Configuration non reconnue.")


# utils.py
def impute_missing_with_product_means(cleaned_data: dict, means_df, product_key: str = "Nom"):
    """
    Remplace les valeurs numeriques manquantes (None) par les moyennes du produit.
    cleaned_data: dict avec Produit + colonnes numeriques (floats ou None)
    means_df: DataFrame indexe par Produit, colonnes = variables numeriques
    """
    produit = cleaned_data.get(product_key)

    if produit is None:
        raise ValueError("Produit manquant dans le formulaire.")

    produit = str(produit).strip()

    if produit not in means_df.index:
        raise ValueError(f"Nom '{produit}' introuvable dans Moyenne_Feedtables.")

    means_row = means_df.loc[produit]

    for k, v in list(cleaned_data.items()):
        if k == product_key:
            continue
        if v is None:
            if k in means_row.index:
                cleaned_data[k] = float(means_row[k]) if means_row[k] == means_row[k] else None  # NaN check
            else:
                cleaned_data[k] = None

    return cleaned_data