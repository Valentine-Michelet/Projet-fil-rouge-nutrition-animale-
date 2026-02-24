import pandas as pd
import numpy as np

# Define column lists (same as pipeline)
# Liste des variables physico-chimiques

vars_expl = [
    "MS % brut", "PB % brut", "CB % brut", "MGR % brut", "MM % brut",
    "NDF % brut", "ADF % brut", "Lignine % brut", "Amidon % brut", "Sucres % brut"
]

# Liste des colonnes des variables cibles
vars_cibles = [
    "EB (kcal) kcal/kg brut", "ED porc croissance (kcal) kcal/kg brut", "EM porc croissance (kcal) kcal/kg brut",
    "EN porc croissance (kcal) kcal/kg brut", "EMAn coq (kcal) kcal/kg brut", "EMAn poulet (kcal) kcal/kg brut",
    "UFL 2018 par kg brut", "UFV 2018 par kg brut", "PDIA 2018 g/kg brut", "PDI 2018 g/kg brut", "BalProRu 2018 g/kg brut"
]

def import_data_merged(file_name):
    """
    Docstring pour import_data_merged
    
    :param file_name: Nom du fichier data_merged - ou .CSV
    Objectif : importer le fichier et retirer l'espace présent dans le dernier caractère de nom
    """
    df_merged = pd.read_csv(file_name)
    df_merged["Nom"] = df_merged["Nom"].str.strip()
    print(f"✓ Data loaded")
    print(f"  - df_merged shape: {df_merged.shape}")

    return df_merged


def separer_data_avec_sans_nom(df, Nom_a_separer = "Blé tendre", colonne = "Nom"):
    """
    Docstring pour separer_data_avec_sans_nom
    
    :param df: Dataframe issue de data_merge ou avec au moins la colonne Nom
    :param Nom_a_separer: Mot à supprimer en 2 datasets
    :param colonne : nom de la colonne où faire la recherche

    Output : df_avec, df_sans et mask
        dataframes filtrés (avec le nom et sans le nom), et le mask de donnée
    """

    df[colonne] = df[colonne].str.strip()
    mask = df[colonne] == Nom_a_separer

    df_avec = df[mask]
    df_sans = df[~mask]
    print("="*50)
    print(f"✓ Data séparée sur :", Nom_a_separer)
    print(f"  - df_sans shape: {df_sans.shape}")
    print(f"  - df_avec shape: {df_avec.shape}")
    print("="*50)

    return df_avec, df_sans, mask

def separer_embedding(embedding, mask):
    """
    Docstring pour separer_embedding
    
    :param embedding: Embedding de la solution
    :param mask: Mask calculé pour la séparation

    Output 
        - emb_avec embedding avec les valeurs masquées
        - emb_sans embeeding avec lesvaleurs non masquées
    """

    emb_avec = embedding[mask]
    emb_sans = embedding[~mask]
    print("Vérification - Avec :", emb_avec.shape,"- Sans :", emb_sans.shape)

    return emb_avec, emb_sans

def transformer_df_2_nparray(df, vars):
    """
    Docstring pour transformer_df_2_numpy__PC
    
    :param df: Dataframe avec les colonnes des vars
    :param vars_expl: Nom des colonnes à extraire

    Output : Numpy array des values du dataframe des colonnes sélectionnées
    """
    array = df[vars].fillna(0).values

    return array  


def extracion_X_y(df, vars_expl = vars_expl, vars_cibles = vars_cibles):

    X_vars = transformer_df_2_nparray(df, vars_expl)
    y = transformer_df_2_nparray(df, vars_cibles)
    print("Transformation faites - X_vars :", X_vars.shape, "- y :", y.shape )
    return X_vars, y

def sep_extr_X_y(df, embedding, Nom_a_separer = "Blé tendre", colonne = "Nom", vars_expl = vars_expl, vars_cibles = vars_cibles):
    """
    Docstring pour sep_extr_X_y
    
    :param df: Description
    :param embedding: Description
    :param Nom_a_separer: Description
    :param colonne: Description
    :param vars_expl: Description
    :param vars_cibles: Description
    Output
    - emb_avec
    - emb_sans
    - X_vars_avec
    - X_vars_sans
    - y_avec
    - y_sans
    """ 

    df_avec, df_sans, mask = separer_data_avec_sans_nom(df, Nom_a_separer, colonne)
    emb_avec, emb_sans = separer_embedding(embedding, mask)

    X_vars_avec, y_avec = extracion_X_y(df_avec, vars_expl, vars_cibles)
    X_vars_sans, y_sans = extracion_X_y(df_sans, vars_expl, vars_cibles)

    return emb_avec, emb_sans, X_vars_avec, X_vars_sans, y_avec, y_sans


def import_moyennes(data_dir: str = "data", 
                    path="data/Moyenne_Feedtables.csv") -> pd.DataFrame:
    
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        decimal=","
    )

    # Nettoyage
    df.columns = df.columns.str.strip()
    df["Nom"] = df["Nom"].astype(str).str.strip()

    # Index produit pour acces rapide
    df = df.set_index("Nom")

    # Conversion numerique (les trucs non numeriques -> NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Optionnel: nettoyer l index (espaces, etc.)
    df.index = df.index.astype(str).str.strip()

    return df
