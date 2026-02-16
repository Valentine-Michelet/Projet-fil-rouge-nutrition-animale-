import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import src.utils_preprocessing as up

import copy
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

def random_phys_mask(batch_size: int, n_phys: int, keep_first_k: int = 2, p_drop: float = 0.3, device="cpu"):
    """
    Returns phys_present_mask: (B, n_phys) with 1=present, 0=missing
    Ensures first keep_first_k phys are always present.
    """
    mask = torch.ones(batch_size, n_phys, device=device)
    if n_phys > keep_first_k:
        drop = (torch.rand(batch_size, n_phys - keep_first_k, device=device) < p_drop).float()
        mask[:, keep_first_k:] = 1.0 - drop
    return mask

def make_phys_present_mask(
    X_batch,
    latent_dim=16,
    n_phys=10,
    keep_first_k=2,
    p_drop=0.3,
    mode="random"
):
    """
    Retourne un masque (batch, n_phys) :
    1 = variable présente
    0 = variable manquante
    """

    B = X_batch.size(0)
    device = X_batch.device

    if mode == "all":
        return torch.ones(B, n_phys, device=device)

    if mode == "random":
        mask = torch.ones(B, n_phys, device=device)

        if n_phys > keep_first_k:
            drop = (torch.rand(B, n_phys - keep_first_k, device=device) < p_drop).float()
            mask[:, keep_first_k:] = 1.0 - drop

        return mask

def train_epoch(
        model, train_loader, optimizer, criterion, device, latent_dim=16, n_phys=10, p_drop=0.8, is_transformer = False
):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if is_transformer is True:
            phys_mask = make_phys_present_mask(
                X_batch,
                latent_dim=latent_dim,
                n_phys=n_phys,
                p_drop=p_drop,
                mode="random"
            )

        # masque des variables physiques (tout présent pour l’instant)
        # phys_mask = torch.ones(X_batch.size(0), n_phys, device=device)

        optimizer.zero_grad()
        if is_transformer == True : 
            predictions = model(X_batch, phys_present_mask=phys_mask)
        else : 
            predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        if is_transformer == True : 
            # CLIP GRADIENT (important pour transformer tabulaire)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(train_loader.dataset)


def evaluate_model( model, loader, criterion, device, latent_dim=16, n_phys=10, is_transformer = False
):
    """Evaluate model on dataset (Transformer version)"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_transformer is True:
                # en validation : pas de variables manquantes
                phys_mask = make_phys_present_mask(
                    X_batch,
                    latent_dim=latent_dim,
                    n_phys=n_phys,
                    mode="all"
                )
                predictions = model(X_batch, phys_present_mask=phys_mask)         
            else: 
                predictions = model(X_batch)

            loss = criterion(predictions, y_batch)
            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


def get_predictions(model, loader, device, batch_size=32, latent_dim=16,
                    n_phys=10, keep_first_k=2, masking="none", p_drop=0.8,
                    is_transformer=False):
    """Get predictions for dataset"""
    model.eval()
    # dataset = TensorDataset(torch.FloatTensor(X))
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            if is_transformer is True:
                # # toutes les variables physiques présentes
                # phys_mask = torch.ones(X_batch.size(0), n_phys,device=device)
                
                B = X_batch.size(0)
                # Choix du type de masking
                if masking == "none":
                    phys_mask = torch.ones(B, n_phys, device=device)

                elif masking == "random":
                    phys_mask = random_phys_mask(
                        batch_size=B,
                        n_phys=n_phys,
                        keep_first_k=keep_first_k,
                        p_drop=p_drop,
                        device=device
                    )

                elif masking == "all":
                    phys_mask = torch.zeros(B, n_phys, device=device)
                    phys_mask[:, :keep_first_k] = 1.0  # garder les premières

                else:
                    raise ValueError("masking must be 'none', 'random', or 'all'")

                pred = model(X_batch, phys_present_mask=phys_mask)
            if is_transformer is False: 
                pred = model(X_batch)
            predictions.append(pred.cpu().numpy())

    return np.vstack(predictions)


def train_model(model,train_loader,val_loader, optimizer,loss_fn,device,
                num_epochs=500, patience=50, checkpoint_path=None, scheduler=None, 
                is_transformer=False, batch_size=32, latent_dim=16, n_phys=10, 
                keep_first_k=2, masking="none", p_drop=0.8,
                ):
    """
    Docstring for train_model
    
    :param model: Description
    :param train_loader: Description
    :param val_loader: Description
    :param optimizer: Description
    :param loss_fn: Description
    :param device: Description
    :param num_epochs: Description
    :param patience: Description
    :param checkpoint_path: Description
    :param scheduler: Description
    :param is_transformer: Description
    :param batch_size: Description
    :param latent_dim: Description
    :param n_phys: Description
    :param keep_first_k: Description
    :param masking: Description
    :param p_drop: Description
    """
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    history = {
        "train_loss": [],
        "val_loss": []
    }

    model.to(device)

    for epoch in tqdm(range(num_epochs)):

        # ---- TRAIN ----
        model.train()
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, is_transformer)

        # ---- VALIDATION ----
        model.eval()
        with torch.no_grad():
            val_loss = evaluate_model(model, val_loader, loss_fn,
                                      device, latent_dim, n_phys,
                                      is_transformer)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Scheduler optionnel
        if scheduler is not None:
            scheduler.step(val_loss)

        # Affichage propre
        if epoch % 10 ==0 :
            print(f"Epoch {epoch+1:3d} | Train \
                {train_loss:.4f} | Val {val_loss:.4f}")

        # ---- EARLY STOPPING ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())

            if checkpoint_path is not None:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "history": history
                }, checkpoint_path)

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Recharge le meilleur modèle
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    return model, history


def Calcul_evaluation(model, train_loader, val_loader, test_loader, scaler_y,
                      device, vars_cibles, is_transformer=False):
    
    train_loader_eval = DataLoader(
    train_loader.dataset,
    batch_size=train_loader.batch_size,
    shuffle=False
    )

    X_train, y_train_scaled = up.dataloader_to_tensors(train_loader_eval)
    X_val, y_val_scaled = up.dataloader_to_tensors(val_loader)
    X_test, y_test_scaled = up.dataloader_to_tensors(test_loader)
    
    # Get predictions in scaled space (pass arrays, not dataloaders)
    y_train_pred_scaled = get_predictions(model, train_loader_eval, device, is_transformer=is_transformer)
    y_val_pred_scaled = get_predictions(model, val_loader, device, is_transformer=is_transformer)
    y_test_pred_scaled = get_predictions(model, test_loader, device, is_transformer=is_transformer)

    # Inverse-transform predictions back to original scale
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

    y_train = scaler_y.inverse_transform(y_train_scaled.cpu().numpy())
    y_val = scaler_y.inverse_transform(y_val_scaled.cpu().numpy())
    y_test = scaler_y.inverse_transform(y_test_scaled.cpu().numpy())

    # Calculate metrics on original scale
    print("\nSCENARIO 2 Results:")
    print("\nTrain Set:")
    r2_train = r2_score(y_train, y_train_pred, multioutput='raw_values')
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred, multioutput='raw_values'))
    mae_train = mean_absolute_error(y_train, y_train_pred, multioutput='raw_values')

    print(f"  R²:   {r2_train.mean():.4f} (±{r2_train.std():.4f})")
    print(f"  RMSE: {rmse_train.mean():.4f} (±{rmse_train.std():.4f})")
    print(f"  MAE:  {mae_train.mean():.4f} (±{mae_train.std():.4f})")

    print("\nValidation Set:")
    r2_val = r2_score(y_val, y_val_pred, multioutput='raw_values')
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred, multioutput='raw_values'))
    mae_val = mean_absolute_error(y_val, y_val_pred, multioutput='raw_values')

    print(f"  R²:   {r2_val.mean():.4f} (±{r2_val.std():.4f})")
    print(f"  RMSE: {rmse_val.mean():.4f} (±{rmse_val.std():.4f})")
    print(f"  MAE:  {mae_val.mean():.4f} (±{mae_val.std():.4f})")

    print("\nTest Set (OOD - Feedtables):")
    r2_test = r2_score(y_test, y_test_pred, multioutput='raw_values')
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred, multioutput='raw_values'))
    mae_test = mean_absolute_error(y_test, y_test_pred, multioutput='raw_values')

    print(f"  R²:   {r2_test.mean():.4f} (±{r2_test.std():.4f})")
    print(f"  RMSE: {rmse_test.mean():.4f} (±{rmse_test.std():.4f})")
    print(f"  MAE:  {mae_test.mean():.4f} (±{mae_test.std():.4f})")

    results = []
    for i, target in enumerate(vars_cibles):
        r2_test = r2_score(y_test[:, i], y_test_pred[:, i])
        rmse_test = np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i]))
        mae_test = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
        
        results.append({
            'target': target,
            'R2_test': r2_test,
            'RMSE_test': rmse_test,
            'MAE_test': mae_test
        })

    df_results = pd.DataFrame(results)

    return df_results


def print_regression_results(df_results, scenario_name="Scenario"):
    """
    Affiche les résultats par variable cible et les statistiques globales.
    
    Parameters
    ----------
    df_results : pandas.DataFrame
        Doit contenir les colonnes :
        ['target', 'R2_test', 'RMSE_test', 'MAE_test']
    scenario_name : str
        Nom affiché pour le scénario
    """

    print(f"\n{scenario_name} - Performance per target variable:")
    print(f"{'Target':<35} {'R² (test)':<15} {'RMSE':<15} {'MAE':<15}")
    print("-" * 80)

    for _, row in df_results.iterrows():
        target = row["target"]
        r2 = row["R2_test"]
        rmse = row["RMSE_test"]
        mae = row["MAE_test"]

        print(f"{target[:33]:<35} {r2:>6.3f}        {rmse:>10.2f}       {mae:>10.2f}")

    print(f"\nOverall Statistics ({scenario_name}):")
    print(f"  - Mean R² (test):   {df_results['R2_test'].mean():.3f} ± {df_results['R2_test'].std():.3f}")
    print(f"  - Mean RMSE (test): {df_results['RMSE_test'].mean():.2f} ± {df_results['RMSE_test'].std():.2f}")
    print(f"  - Mean MAE (test):  {df_results['MAE_test'].mean():.2f} ± {df_results['MAE_test'].std():.2f}")


def save_weights(path, model):
    torch.save(model.state_dict(), path)
    print(f"Weights saved to {path}")


def load_weights(path, model, device):
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
