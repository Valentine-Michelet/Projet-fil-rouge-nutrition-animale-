import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch


from sklearn.preprocessing import StandardScaler


def scale_from_train(A_train, *arrays, return_scaler=True):
    """
    Fit un StandardScaler sur A_train,
    transforme A_train et tous les autres arrays passés.

    Paramètres
    ----------
    A_train : array-like
        Données d'entraînement
    *arrays : array-like
        Données à transformer avec le scaler du train (val, test, OOD...)
    return_scaler : bool
        Si True, renvoie aussi le scaler

    Retour
    -------
    scaled_arrays : tuple
        (A_train_scaled, A_1_scaled, A_2_scaled, ...)
    scaler : StandardScaler (optionnel)
    """

    scaler = StandardScaler()
    A_train_scaled = scaler.fit_transform(A_train)

    scaled_others = [scaler.transform(A) for A in arrays]

    if return_scaler:
        return (A_train_scaled, *scaled_others), scaler
    else:
        return (A_train_scaled, *scaled_others)


def tensor_2_DataLoader(X, y, shuffle_value=True): 

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
        )
    loader = DataLoader(dataset, batch_size=32, shuffle=shuffle_value)

    return loader


def split_2_DataLoader(X, y, train_size=0.6, val_size=0.25, scaler=True):
    """
    Docstring for split_2_DataLoader
    
    :param X: List/tensor/array des variables d'entrées
    :param y: List/tensor/array des variables des cibles
    :param train_size: taille du set d'entrainement
    :param val_size: taille du set de validation
    Rappel : train_size + val_size = 1 - test_size
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_size)

    if scaler==True :
        (X_train, X_val, X_test), scaler_X = scale_from_train(X_train, X_val, X_test)
        (y_train, y_val, y_test), scaler_y = scale_from_train(y_train, y_val, y_test)

    train_loader = tensor_2_DataLoader(X_train, y_train, shuffle_value=True)
    val_loader = tensor_2_DataLoader(X_val, y_val, shuffle_value=False)
    test_loader = tensor_2_DataLoader(X_test, y_test, shuffle_value=False)

    if scaler==True :
        return train_loader, val_loader, test_loader, scaler_X, scaler_y
    else : 
        return train_loader, val_loader, test_loader

def dataloader_to_tensors(loader):
    """
    Extrait X et y directement depuis un DataLoader basé sur TensorDataset.
    
    Parameters
    ----------
    loader : torch.utils.data.DataLoader
    
    Returns
    -------
    X : torch.Tensor
    y : torch.Tensor
    """
    dataset = loader.dataset
    
    if not hasattr(dataset, "tensors"):
        raise TypeError("Le DataLoader ne contient pas un TensorDataset.")
    
    X, y = dataset.tensors
    return X, y
