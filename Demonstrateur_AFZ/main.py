from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import csv
import io

from models import import_model, predict_from_input, import_moyennes
import numpy as np
import pandas as pd

from utils import (
    compute_metrics_per_target,
    make_real_vs_pred_div,
    make_metrics_bar_div,
    make_residuals_hist_div,
    impute_missing_with_product_means
)



# ==========================
# Variables globales modèle
# ==========================
model = None
encoder = None
X_columns = None
X_train = None
X_test = None
y_train = None
y_test = None
mae = None
rmse = None
r2 = None

# ==============
# Target names
# ==============
TARGET_NAMES = [
    "EB (kcal) kcal/kg brut",
    "ED porc croissance (kcal) kcal/kg brut",
    "EM porc croissance (kcal) kcal/kg brut",
    "EN porc croissance (kcal) kcal/kg brut",
    "EMAn coq (kcal) kcal/kg brut",
    "EMAn poulet (kcal) kcal/kg brut",
    "UFL 2018 par kg brut",
    "UFV 2018 par kg brut",
    "PDIA 2018 g/kg brut",
    "PDI 2018 g/kg brut",
    "BalProRu 2018 g/kg brut"
]


# ==============
# Chargement du modèle
# ==============

def load_model():
    global model, encoder, X_columns, X_train, X_test, y_train, y_test, mae, rmse, r2
    model, encoder, X_columns, X_train, X_test, y_train, y_test, mae, rmse, r2 = import_model()

moyenne_df = import_moyennes()

# Chargement initial
load_model()


# ==========================
# FastAPI
# ==========================
app = FastAPI(title="Mini calculatrice FastAPI")
templates = Jinja2Templates(directory="templates")

prediction_history = []



# ==========================
# Routes
# ==========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "page": "home"
    })


@app.post("/calcul", response_class=HTMLResponse)
def calcul(
    request: Request,
    x: float = Form(...),
    y: float = Form(...),
    operation: str = Form(...)
):
    if operation == "add":
        result = x + y
        op_symbol = "+"
    elif operation == "mul":
        result = x * y
        op_symbol = "*"
    else:
        result = "Opération invalide"
        op_symbol = "?"

    history.append({
        "x": x,
        "y": y,
        "operation": op_symbol,
        "result": result
    })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "page": "home"
    })


@app.get("/model_info", response_class=HTMLResponse)
def model_info(request: Request):
    return templates.TemplateResponse("model_info.html", {
        "request": request,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "n_features": X_train.shape[1],
        "model_name": type(model).__name__,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "params": model.get_params(),
        "page": "model"
    })


@app.post("/retrain")
def retrain_model():
    load_model()
    return RedirectResponse(url="/model_info", status_code=303)


@app.get("/predict", response_class=HTMLResponse)
def predict_form(request: Request):

    categories = {col: encoder.categories_[i].tolist()
                  for i, col in enumerate(encoder.feature_names_in_)}

    numeric_cols = [c for c in X_columns if c not in encoder.get_feature_names_out()]

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "categories": categories,
        "numeric_cols": numeric_cols,
        "prediction_history": prediction_history, 
        "target_names": TARGET_NAMES,              
        "page": "predict"
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict_result(request: Request):
    form = await request.form()
    form_data = dict(form)

    numeric_cols = [c for c in X_columns if c not in encoder.get_feature_names_out()]

    cleaned_data = {}

    imputed_fields = []   ### AJOUT 1 — on prépare la liste

    print("FORM DATA KEYS:", form_data.keys())

    try:
        for col, val in form_data.items():

            # Normalisation string
            if isinstance(val, str):
                val = val.strip()

            if col in numeric_cols:
                # Champ vide => valeur manquante
                if val in ("", "N/A", "NA", "nan", "NaN", None):
                    cleaned_data[col] = None
                    imputed_fields.append(col)   ### AJOUT 2 — on marque
                else:
                    cleaned_data[col] = float(val)
            else:
                cleaned_data[col] = val

        # Imputation par produit
        cleaned_data = impute_missing_with_product_means(
            cleaned_data,
            moyenne_df,
            product_key="Nom"
        )

    except ValueError as e:
        categories = {col: encoder.categories_[i].tolist()
                      for i, col in enumerate(encoder.feature_names_in_)}

        return templates.TemplateResponse("predict.html", {
            "request": request,
            "categories": categories,
            "numeric_cols": numeric_cols,
            "error": f"Erreur formulaire: {str(e)}",
            "page": "predict"
        })

    ### AJOUT 3 — booléen global
    imputation_used = len(imputed_fields) > 0

    results = predict_from_input(
        model,
        encoder,
        X_columns,
        cleaned_data,
        TARGET_NAMES,
        imputation_used=imputation_used   ### AJOUT 4
    )

    prediction_history.append({
        "input": cleaned_data,
        "results": results,
        "imputation_used": imputation_used   ### AJOUT 5
    })

    categories = {col: encoder.categories_[i].tolist()
                  for i, col in enumerate(encoder.feature_names_in_)}

    print(prediction_history)

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "categories": categories,
        "numeric_cols": numeric_cols,
        "results": results,
        "prediction_history": prediction_history,
        "target_names": TARGET_NAMES,
        "imputation_used": imputation_used,   ### AJOUT 6
        "page": "predict"
    })

@app.get("/export_csv")
def export_csv():

    if not prediction_history:
        return {"error": "Aucune donnée à exporter"}

    rows = []

    for item in prediction_history:

        row_dict = {}

        # ===== INPUTS =====
        for key, value in item["input"].items():
            row_dict[key] = value

        # ===== OUTPUTS =====
        for output in item["results"]:
            row_dict[output["name"]] = output["value"]

        # ===== AJOUT COLONNE IMPUTATION =====
        row_dict["Imputation"] = "Oui" if item.get("imputation_used", False) else "Non"

        rows.append(row_dict)

    # Création DataFrame
    df_export = pd.DataFrame(rows)

    # ===== ORDONNER LES COLONNES =====
    input_cols = list(prediction_history[0]["input"].keys())
    output_cols = [r["name"] for r in prediction_history[0]["results"]]

    ordered_cols = input_cols + output_cols + ["Imputation"]

    df_export = df_export[ordered_cols]

    # ===== EXPORT CSV =====
    buffer = io.StringIO()
    df_export.to_csv(buffer, index=False, sep=";")

    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=historique_predictions.csv"
        }
    )

@app.get("/diagnostics", response_class=HTMLResponse)
def diagnostics(
    request: Request,
    split: str = "test",   # "train" ou "test"
    n: int = 600,          # nb de points
    seed: int = 42,
    ncols: int = 4,
    err_thresh: float = 0.15
):


    # --- choix du split
    split = (split or "test").lower().strip()
    if split not in ("train", "test"):
        split = "test"

    if split == "train":
        X_ref, y_ref = X_train, y_train
    else:
        X_ref, y_ref = X_test, y_test

    # --- sampling
    rng = np.random.default_rng(seed)
    n_total = int(X_ref.shape[0])
    n = max(50, min(int(n), n_total))  # garde-fou
    idx = rng.choice(n_total, size=n, replace=False)

    X_s = X_ref.iloc[idx] if hasattr(X_ref, "iloc") else X_ref[idx]
    y_s = y_ref.iloc[idx] if hasattr(y_ref, "iloc") else y_ref[idx]

    # --- predict
    y_pred = model.predict(X_s)

    # --- target names (on prend ceux du y pour être cohérent)
    target_names = list(getattr(y_ref, "columns", TARGET_NAMES))

    # --- métriques
    metrics_rows, global_metrics = compute_metrics_per_target(
        y_s, y_pred, target_names
        )

    # --- plots (Plotly JS inclus UNE fois, sur le 1er bloc)
    plot_scatter = make_real_vs_pred_div(
        y_s, y_pred, target_names, ncols=ncols, include_js=True
        )
    plot_metrics = make_metrics_bar_div(metrics_rows, include_js=False)
    plot_resid = make_residuals_hist_div(y_s, y_pred, include_js=False)
    # Création du tableau
    table_rows = []

    y_true_np = np.asarray(y_s)
    y_pred_np = np.asarray(y_pred)

    n_samples = y_true_np.shape[0]

    # conversion % → fraction
    error_threshold = float(err_thresh) / 100.0

    for i in range(n_samples):

        # ligne TRUE
        row_true = {
            "sample_id": i,
            "type": "true",
            "errors": {}
        }
        for j, name in enumerate(target_names):
            val = float(y_true_np[i, j])
            row_true[name] = val
            row_true["errors"][name] = False
        table_rows.append(row_true)

        # ligne PRED
        row_pred = {
            "sample_id": i,
            "type": "pred",
            "errors": {}
        }

        for j, name in enumerate(target_names):
            true_val = float(y_true_np[i, j])
            pred_val = float(y_pred_np[i, j])

            row_pred[name] = pred_val

            # erreur relative sécurisée
            if abs(true_val) > 1e-6:
                rel_err = abs(pred_val - true_val) / abs(true_val)
            else:
                rel_err = abs(pred_val - true_val)

            row_pred["errors"][name] = rel_err > error_threshold

        table_rows.append(row_pred)

    return templates.TemplateResponse("diagnostics.html", {
        "request": request,
        "page": "diagnostics",

        # controls (pour réafficher le formulaire)
        "split": split,
        "n": n,
        "seed": seed,
        "ncols": ncols,

        # metrics
        "metrics_rows": metrics_rows,
        "global_metrics": global_metrics,

        # plots html
        "plot_scatter": plot_scatter,
        "plot_metrics": plot_metrics,
        "plot_resid": plot_resid,
        "table_rows": table_rows,
        "target_names": target_names,
        "error_threshold": error_threshold
    })
