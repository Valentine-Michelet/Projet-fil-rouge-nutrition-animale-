"""
src/features.py - Feature engineering functions (encoding, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict


def encode_categorical_feature(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    data: pd.DataFrame, 
    feature_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode a categorical feature using LabelEncoder.
    Learns encoding on entire dataset to ensure consistency.
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        data: Full original DataFrame (for fitting encoder)
        feature_name: Name of the categorical feature to encode
        
    Returns:
        Tuple of (X_train with encoded feature, X_test with encoded feature)
    """
    le = LabelEncoder()
    
    # Fit encoder on ALL classes in original dataset
    all_classes = data[feature_name].values
    le.fit(all_classes)
    
    # Encode train and test data (take from X_train/X_test if already there,
    # otherwise take from original data using indices)
    if feature_name in X_train.columns:
        encoded_train = le.transform(X_train[feature_name].values)
    else:
        encoded_train = le.transform(data.loc[X_train.index, feature_name].values)
    
    if feature_name in X_test.columns:
        encoded_test = le.transform(X_test[feature_name].values)
    else:
        encoded_test = le.transform(data.loc[X_test.index, feature_name].values)
    
    # Add encoded columns
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    
    X_train_out[f"{feature_name}_encoded"] = encoded_train
    X_test_out[f"{feature_name}_encoded"] = encoded_test
    
    return X_train_out, X_test_out


def onehot_encode_categorical_feature(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    data: pd.DataFrame, 
    feature_name: str, 
    indices_train: pd.Index, 
    indices_test: pd.Index,
    drop_first: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode a categorical feature.
    Learns categories from entire dataset to ensure consistency.
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        data: Full original DataFrame
        feature_name: Name of the categorical feature to encode
        indices_train: Indices for training set
        indices_test: Indices for test set
        drop_first: Whether to drop first category to avoid collinearity
        
    Returns:
        Tuple of (X_train with one-hot encoded features, X_test with one-hot encoded features)
    """
    # Get all unique categories from original data
    categories = sorted(data[feature_name].unique())
    
    # Create training one-hot encoding
    X_train_encoded = pd.DataFrame(0, 
                                   index=X_train.index, 
                                   columns=[f"{feature_name}_{cat}" for cat in categories])
    
    X_test_encoded = pd.DataFrame(0, 
                                  index=X_test.index, 
                                  columns=[f"{feature_name}_{cat}" for cat in categories])
    
    # Fill in the one-hot values
    for cat in categories:
        train_mask = data.loc[indices_train, feature_name] == cat
        test_mask = data.loc[indices_test, feature_name] == cat
        
        X_train_encoded.loc[train_mask.index[train_mask], f"{feature_name}_{cat}"] = 1
        X_test_encoded.loc[test_mask.index[test_mask], f"{feature_name}_{cat}"] = 1
    
    if drop_first:
        X_train_encoded = X_train_encoded.drop(columns=[X_train_encoded.columns[0]])
        X_test_encoded = X_test_encoded.drop(columns=[X_test_encoded.columns[0]])
    
    # Combine with original features
    X_train_out = pd.concat([X_train, X_train_encoded], axis=1)
    X_test_out = pd.concat([X_test, X_test_encoded], axis=1)
    
    return X_train_out, X_test_out


def get_simplified_names() -> Dict[str, str]:
    """
    Get mapping of full column names to simplified names for display.
    
    Returns:
        Dictionary mapping full names to simplified names
    """
    return {
        "MS % brut": "MS",
        "PB % brut": "PB",
        "CB % brut": "CB",
        "MGR % brut": "MGR",
        "MM % brut": "MM",
        "NDF % brut": "NDF",
        "ADF % brut": "ADF",
        "Lignine % brut": "Lignine",
        "Amidon % brut": "Amidon",
        "Sucres % brut": "Sucres",
        "EB (kcal) kcal/kg brut": "EB",
        "ED porc croissance (kcal) kcal/kg brut": "ED porc",
        "EM porc croissance (kcal) kcal/kg brut": "EM porc",
        "EN porc croissance (kcal) kcal/kg brut": "EN porc",
        "EMAn coq (kcal) kcal/kg brut": "EMAn coq",
        "EMAn poulet (kcal) kcal/kg brut": "EMAn poulet",
        "UFL 2018 par kg brut": "UFL",
        "UFV 2018 par kg brut": "UFV",
        "PDIA 2018 g/kg brut": "PDIA",
        "PDI 2018 g/kg brut": "PDI",
        "BalProRu 2018 g/kg brut": "BalProRu"
    }
