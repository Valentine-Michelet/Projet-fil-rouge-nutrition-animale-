import sys
sys.path.insert(0, '../')

import numpy as np
import src.plot_deep as plt_d
import src.models as md
import src.utils_deep as ud
import src.utils_preprocessing as up
import src.import_data as id
import pandas as pd
import tqdm as tqdm


import torch
import torch.nn.functional as F
import numpy
import matplotlib.pyplot as plt
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

df_merged = id.import_data_merged(file_name="../data/merge/data_merged.csv")
embedding = np.load("../data/embeddings/embeddings_combined.npy")
vars_expl = id.vars_expl
vars_cibles = id.vars_cibles

liste_nom_supp = [
    "Avoine",
    "Blé tendre",
    "Germes de maïs",
    "Manioc, amidon 70-74 %",
    "Farine de gousse de caroube",
    "Poudre de lait entier", 
    "Tourteau de palmiste, huile 5-20%", 
    "Graine de soja extrudée",
    "Drêches de maïs de distillerie avec solubles, déshydratées, huile > 6 %",
    "Triticale"
    ]
all_results = []

# =========================
# TEST PREALABLE DES NOMS
# =========================

# Nettoyage minimal (évite les espaces parasites)
df_merged["Nom"] = df_merged["Nom"].str.strip()
liste_nom_supp = [n.strip() for n in liste_nom_supp]

noms_df = set(df_merged["Nom"].dropna().unique())

noms_absents = [n for n in liste_nom_supp if n not in noms_df]

if len(noms_absents) == 0:
    print("✅ Tous les noms de liste_nom_supp sont présents dans df_merged.")
else:
    print("❌ Noms absents dans df_merged :")
    for n in noms_absents:
        print("   -", repr(n))
    
    raise ValueError("Corrige les noms avant de lancer la boucle.")

train_loader_emb, val_loader_emb, test_loader_emb, scaler_X_emb, scaler_y_emb = up.split_2_DataLoader(embedding, embedding, val_size=0.8)

model_auto_encoder = md.MirrorAutoEncoder().to(device)
optimizer_auto = torch.optim.AdamW(model_auto_encoder.parameters(), lr=1e-3, weight_decay=1e-4)
criterion_auto = F.smooth_l1_loss

model_auto_encoder, history_auto = ud.train_model(model_auto_encoder, train_loader_emb, val_loader_emb,optimizer_auto, criterion_auto, device)
frozen_encoder = md.FrozenEncoder(model_auto_encoder.encoder).to(device)

for p in frozen_encoder.parameters():
    p.requires_grad = False


for nom in tqdm.tqdm(liste_nom_supp):
    # Création des ensembles
    emb_avec, emb_sans, X_avec, X_sans, y_avec, y_sans = id.sep_extr_X_y(df_merged, embedding, vars_expl= vars_expl, vars_cibles=vars_cibles, Nom_a_separer=nom)

    # Transformation en tensor

    X_emb_sans = torch.FloatTensor(emb_sans).to(device)
    X_emb_avec = torch.FloatTensor(emb_avec).to(device)

    # Inférence et réduction de dimensions

    X_red_sans = frozen_encoder(X_emb_sans)
    X_red_avec = frozen_encoder(X_emb_avec)

    # Concaténation

    X_combined_red_sans = np.hstack([X_red_sans.to("cpu"), X_sans])
    X_combined_red_avec = np.hstack([X_red_avec.to("cpu"), X_avec])

    train_loader_MLP, val_loader_MLP, test_loader_MLP, scaler_X_MLP, scaler_y_MLP = up.split_2_DataLoader(
    X_combined_red_sans, y_sans, train_size=0.7, val_size=0.8
    )

    X_avec_scaled = scaler_X_MLP.transform(X_combined_red_avec)
    y_avec_scaled = scaler_y_MLP.transform(y_avec)

    avec_dataloader = up.tensor_2_DataLoader(X_avec_scaled, y_avec_scaled, shuffle_value=False)

    model_MLP = md.MLPRegressor(input_size=26, hidden_sizes=[128, 256, 64], output_size=11, dropout_rate=0.2).to(device)
    model_MLP = ud.load_weights("../models/model_MLP_Ba.pth",model_MLP, device)
    optimizer_MLP = torch.optim.Adam(model_MLP.parameters(), lr=0.001, weight_decay=1e-5)
    criterion_MLP = torch.nn.MSELoss()
    model_MLP, history_MLP = ud.train_model(
    model_MLP, train_loader_MLP, val_loader_MLP,optimizer_MLP, criterion_MLP, device)

    results_MLP = ud.Calcul_evaluation(
    model_MLP, train_loader_MLP, val_loader_MLP, avec_dataloader, scaler_y_MLP, device, vars_cibles
    )

    ud.print_regression_results(results_MLP, "MLP")

    model_trans = md.NutritionTransformer().to(device)
    model_trans = ud.load_weights("../models/model_trans_Ba.pth",model_trans, device)

    optimizer_trans = torch.optim.AdamW(
        model_trans.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    criterion_trans = torch.nn.MSELoss()

    model_trans, history_trans = ud.train_model(model_trans, train_loader_MLP, val_loader_MLP,optimizer_trans, criterion_trans, device, is_transformer=True)

    results_trans = ud.Calcul_evaluation(
    model_trans, train_loader_MLP, val_loader_MLP, avec_dataloader, scaler_y_MLP, device, vars_cibles, is_transformer = True
    )

    ud.print_regression_results(results_trans, "trans")

    # ===============================
    # Ajouter meta-info
    # ===============================

    results_MLP["model"] = "MLP"
    results_MLP["nom_test"] = nom

    results_trans["model"] = "Transformer"
    results_trans["nom_test"] = nom

    # Stockage global
    all_results.append(results_MLP)
    all_results.append(results_trans)


df_all_results = pd.concat(all_results, ignore_index=True)

print("\n=========================")
print("GLOBAL RESULTS DATAFRAME")
print("=========================")
print(df_all_results.head())

df_all_results.to_csv("../reports/results_leave_one_name_out_AE_froze.csv", index=False)
