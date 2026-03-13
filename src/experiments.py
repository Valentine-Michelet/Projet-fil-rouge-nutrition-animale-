"""
src/experiments.py - Model training and experimentation functions
This is the core module for all XGBoost experiment scenarios.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, List, Optional


def prepare_data_random_split(
    data: pd.DataFrame,
    features: List[str],
    target_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data with random train/test split.
    
    Args:
        data: Full DataFrame
        features: List of feature columns
        target_cols: List of target columns
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = data[features].copy()
    y = data[target_cols].copy()
    
    # Stratification sur la colonne 'Nom' si elle existe dans les données
    stratify_col = None
    if 'Nom' in data.columns:
        stratify_col = data['Nom']
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )


def prepare_data_hide_one_per_class(
    data: pd.DataFrame,
    features: List[str],
    target_cols: List[str],
    class_col: str = 'Classe',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare data by hiding one product name per class from training.
    Tests generalization to unknown product names within known classes.
    
    Args:
        data: Full DataFrame
        features: List of feature columns
        target_cols: List of target columns
        class_col: Column name for class
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, classe_test, nom_test)
    """
    np.random.seed(random_state)
    indices_test_list = []
    
    # For each class, select 1 random product name to hide
    for classe in data[class_col].unique():
        classe_data = data[data[class_col] == classe]
        noms_uniques = classe_data['Nom'].unique()
        nom_selectionne = np.random.choice(noms_uniques, 1)[0]
        
        # Get ALL indices with this product name
        indices_nom = classe_data[classe_data['Nom'] == nom_selectionne].index
        indices_test_list.extend(indices_nom)
    
    indices_test = pd.Index(indices_test_list)
    indices_train = data.index.difference(indices_test)
    
    X_train = data.loc[indices_train, features].copy()
    y_train = data.loc[indices_train, target_cols].copy()
    
    X_test = data.loc[indices_test, features].copy()
    y_test = data.loc[indices_test, target_cols].copy()
    
    classe_test = data.loc[indices_test, class_col].values
    nom_test = data.loc[indices_test, "Nom"].values
    
    return X_train, X_test, y_train, y_test, classe_test, nom_test


def prepare_data_hide_class(
    data: pd.DataFrame,
    features: List[str],
    target_cols: List[str],
    class_col: str = 'Classe',
    class_to_hide: Optional[str] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Prepare data by hiding an entire class from training.
    Tests generalization to completely unknown classes.
    
    Args:
        data: Full DataFrame
        features: List of feature columns
        target_cols: List of target columns
        class_col: Column name for class
        class_to_hide: Specific class to hide (uses first if None)
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, classe_cachee)
    """
    np.random.seed(random_state)
    
    # Select class to hide
    if class_to_hide is None:
        classe_a_masquer = data[class_col].unique()[0]
    else:
        classe_a_masquer = class_to_hide
    
    # Split by class
    indices_train = data[data[class_col] != classe_a_masquer].index
    indices_test = data[data[class_col] == classe_a_masquer].index
    
    X_train = data.loc[indices_train, features].copy()
    y_train = data.loc[indices_train, target_cols].copy()
    
    X_test = data.loc[indices_test, features].copy()
    y_test = data.loc[indices_test, target_cols].copy()
    
    return X_train, X_test, y_train, y_test, classe_a_masquer


def prepare_data_drop_features(
    data: pd.DataFrame,
    features: List[str],
    target_cols: List[str],
    features_to_drop: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data with some features removed (robustness testing).
    
    Args:
        data: Full DataFrame
        features: List of all feature columns
        target_cols: List of target columns
        features_to_drop: List of features to exclude from training
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Filter out features to drop
    features_kept = [f for f in features if f not in features_to_drop]
    
    return prepare_data_random_split(data, features_kept, target_cols, test_size, random_state)


def train_xgboost_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    target_cols: List[str],
    simplified_names: Dict[str, str],
    random_state: int = 42,
    xgb_params: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, xgb.XGBRegressor]]:
    """
    Train XGBoost models for each target variable.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        target_cols: List of target column names
        simplified_names: Mapping of full names to simplified names
        random_state: Random seed
        xgb_params: Optional XGBoost parameters dict (defaults used if None)
        
    Returns:
        Tuple of (results_df, metrics_dict, models_dict)
    """
    if xgb_params is None:
        xgb_params = {}
    
    resultats = []
    models = {}
    y_pred_all = []
    
    for target in target_cols:
        # Train individual model
        model = xgb.XGBRegressor(random_state=random_state, **xgb_params)
        model.fit(X_train, y_train[target])
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test[target], y_pred)
        rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
        r2 = r2_score(y_test[target], y_pred)
        
        resultats.append({
            'Variable cible': simplified_names.get(target, target),
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4)
        })
        
        models[target] = model
        y_pred_all.append(y_pred)
    
    # Calculate global metrics
    y_pred_all = np.column_stack(y_pred_all)
    y_test_all = y_test[target_cols].values
    
    r2_uniform = r2_score(y_test_all, y_pred_all, multioutput='uniform_average')
    r2_weighted = r2_score(y_test_all, y_pred_all, multioutput='variance_weighted')
    
    df_resultats = pd.DataFrame(resultats)
    
    metrics_globales = {
        "R2_uniform": round(r2_uniform, 4),
        "R2_variance_weighted": round(r2_weighted, 4),
        "R2_min": round(df_resultats['R2'].min(), 4),
        "R2_max": round(df_resultats['R2'].max(), 4),
        "R2_mean": round(df_resultats['R2'].mean(), 4),
    }
    
    return df_resultats, metrics_globales, models


def train_xgboost_models_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    target_cols: List[str],
    simplified_names: Dict[str, str],
    n_splits: int = 5,
    random_state: int = 42,
    xgb_params: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, List[float]], List[Dict], List[float]]:
    """
    Train XGBoost models using K-Fold Cross-Validation with metrics tracking.
    
    Args:
        X: Full feature DataFrame
        y: Full targets DataFrame
        target_cols: List of target column names
        simplified_names: Mapping of full names to simplified names
        n_splits: Number of CV folds (default: 5)
        random_state: Random seed
        xgb_params: Optional XGBoost parameters dict
        
    Returns:
        Tuple of (summary_df, global_metrics_dict, fold_metrics_dict, all_fold_results, r2_global_folds)
        - summary_df: DataFrame with mean and std for each target
        - global_metrics_dict: Overall R² metrics across all folds
        - fold_metrics_dict: Per-target metrics for each fold
        - all_fold_results: List of detailed results per fold
        - r2_global_folds: List of global R² scores (one per fold)
    """
    if xgb_params is None:
        xgb_params = {}
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Storage for metrics across folds
    fold_scores = {target: {'R2': [], 'MAE': [], 'RMSE': []} for target in target_cols}
    all_fold_results = []
    r2_global_folds = []
    
    fold_idx = 0
    for train_idx, test_idx in kf.split(X):
        fold_idx += 1
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]
        
        fold_results = []
        y_pred_all = []
        
        for target in target_cols:
            # Train model
            model = xgb.XGBRegressor(random_state=random_state, **xgb_params)
            model.fit(X_train_fold, y_train_fold[target])
            
            # Predict
            y_pred = model.predict(X_test_fold)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_fold[target], y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_fold[target], y_pred))
            r2 = r2_score(y_test_fold[target], y_pred)
            
            fold_scores[target]['R2'].append(r2)
            fold_scores[target]['MAE'].append(mae)
            fold_scores[target]['RMSE'].append(rmse)
            
            fold_results.append({
                'Fold': fold_idx,
                'Variable cible': simplified_names.get(target, target),
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })
            
            y_pred_all.append(y_pred)
        
        # Calculate global metrics for this fold
        y_pred_all = np.column_stack(y_pred_all)
        y_test_all = y_test_fold[target_cols].values
        
        r2_weighted = r2_score(y_test_all, y_pred_all, multioutput='variance_weighted')
        r2_global_folds.append(r2_weighted)
        
        all_fold_results.extend(fold_results)
    
    # Create summary DataFrame
    summary_data = []
    for target in target_cols:
        summary_data.append({
            'Variable cible': simplified_names.get(target, target),
            'R2_mean': np.mean(fold_scores[target]['R2']),
            'R2_std': np.std(fold_scores[target]['R2'], ddof=1),  # Use ddof=1 for better estimation
            'MAE_mean': np.mean(fold_scores[target]['MAE']),
            'MAE_std': np.std(fold_scores[target]['MAE'], ddof=1),  # Use ddof=1 for better estimation
            'RMSE_mean': np.mean(fold_scores[target]['RMSE']),
            'RMSE_std': np.std(fold_scores[target]['RMSE'], ddof=1)  # Use ddof=1 for better estimation
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Global metrics
    global_metrics = {
        'R2_variance_weighted_mean': np.mean(r2_global_folds),
        'R2_variance_weighted_std': np.std(r2_global_folds),
        'R2_mean': summary_df['R2_mean'].mean(),
        'R2_std': summary_df['R2_mean'].std()
    }
    
    return summary_df, global_metrics, fold_scores, all_fold_results, r2_global_folds


def get_hidden_products_info(classe_test: np.ndarray, nom_test: np.ndarray) -> pd.DataFrame:
    """
    Create a summary of hidden products used for testing.
    
    Args:
        classe_test: Array of test classes
        nom_test: Array of test product names
        
    Returns:
        DataFrame with unique class-name combinations
    """
    test_products = pd.DataFrame({
        'Classe': classe_test,
        'Nom': nom_test
    })
    
    return test_products.drop_duplicates().reset_index(drop=True)


def compare_scenarios(
    scenario_results: Dict[str, Dict[str, float]],
    metric: str = "R2_variance_weighted"
) -> pd.DataFrame:
    """
    Compare metrics across multiple experiment scenarios.
    
    Args:
        scenario_results: Dict mapping scenario names to their metric dicts
        metric: Which metric to extract
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for scenario_name, metrics_dict in scenario_results.items():
        if metric in metrics_dict:
            comparison_data.append({
                'Scenario': scenario_name,
                'Metric': metric,
                'Value': metrics_dict[metric]
            })
    
    return pd.DataFrame(comparison_data).sort_values('Value', ascending=False)


def robustness_test_features_removed(
    data: pd.DataFrame,
    all_features: List[str],
    target_cols: List[str],
    simplified_names: Dict[str, str],
    features_to_test: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
    """
    Run robustness tests by removing features one at a time.
    
    Args:
        data: Full DataFrame
        all_features: List of all available features
        target_cols: List of target columns
        simplified_names: Mapping of names
        features_to_test: Features to test removal of
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        Dictionary mapping feature_name to (results_df, metrics_dict)
    """
    results = {}
    
    for feature_to_drop in features_to_test:
        X_train, X_test, y_train, y_test = prepare_data_drop_features(
            data, all_features, target_cols, [feature_to_drop], test_size, random_state
        )
        
        results_df, metrics, _ = train_xgboost_models(
            X_train, X_test, y_train, y_test, 
            target_cols, simplified_names, random_state
        )
        
        results[f"Without {feature_to_drop}"] = (results_df, metrics)
    
    return results


def leave_one_product_out_cv(
    data: pd.DataFrame,
    features: List[str],
    target_cols: List[str],
    simplified_names: Dict[str, str],
    product_col: str = 'Nom',
    random_state: int = 42,
    xgb_params: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, List[float]], List[Dict], List[float]]:
    """
    Leave One Product Out Cross-Validation for OOD evaluation.
    For each unique product, train on all OTHER products and test on that product.
    
    Args:
        data: Full DataFrame
        features: List of feature columns
        target_cols: List of target column names
        simplified_names: Mapping of full names to simplified names
        product_col: Column name for product identifier (default: 'Nom')
        random_state: Random seed
        xgb_params: Optional XGBoost parameters dict
        
    Returns:
        Tuple of (summary_df, global_metrics_dict, fold_metrics_dict, all_fold_results, r2_global_folds)
        - summary_df: DataFrame with mean and std for each target variable
        - global_metrics_dict: Overall R² metrics (mean, std, min, max)
        - fold_metrics_dict: Per-target metrics for each fold (product left out)
        - all_fold_results: List of detailed results per fold
        - r2_global_folds: List of global R² scores (one per product/fold)
    """
    if xgb_params is None:
        xgb_params = {}
    
    # Get unique products
    unique_products = data[product_col].unique()
    
    # Storage for metrics
    fold_scores = {target: {'R2': [], 'MAE': [], 'RMSE': []} for target in target_cols}
    all_fold_results = []
    r2_global_folds = []
    product_sizes = []
    
    fold_idx = 0
    for product_to_hide in unique_products:
        fold_idx += 1
        
        # Split: train on all other products, test on this product
        train_indices = data[data[product_col] != product_to_hide].index
        test_indices = data[data[product_col] == product_to_hide].index
        
        X_train = data.loc[train_indices, features].reset_index(drop=True)
        X_test = data.loc[test_indices, features].reset_index(drop=True)
        y_train = data.loc[train_indices, target_cols].reset_index(drop=True)
        y_test = data.loc[test_indices, target_cols].reset_index(drop=True)
        
        # Record product info
        product_size = len(X_test)
        product_sizes.append({
            'Product': product_to_hide,
            'Size': product_size
        })
        
        fold_results = []
        y_pred_all = []
        
        # Train model for each target
        for target in target_cols:
            # Train individual model
            model = xgb.XGBRegressor(random_state=random_state, **xgb_params)
            model.fit(X_train, y_train[target], verbose=False)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test[target], y_pred)
            rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
            r2 = r2_score(y_test[target], y_pred)
            
            fold_scores[target]['R2'].append(r2)
            fold_scores[target]['MAE'].append(mae)
            fold_scores[target]['RMSE'].append(rmse)
            
            fold_results.append({
                'Fold': fold_idx,
                'Product': product_to_hide,
                'Product_Size': product_size,
                'Variable cible': simplified_names.get(target, target),
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })
            
            y_pred_all.append(y_pred)
        
        # Calculate global metrics for this fold
        y_pred_all = np.column_stack(y_pred_all)
        y_test_all = y_test[target_cols].values
        
        r2_weighted = r2_score(y_test_all, y_pred_all, multioutput='variance_weighted')
        r2_global_folds.append(r2_weighted)
        
        all_fold_results.extend(fold_results)
    
    # Create summary DataFrame
    summary_data = []
    for target in target_cols:
        summary_data.append({
            'Variable cible': simplified_names.get(target, target),
            'R2_mean': np.mean(fold_scores[target]['R2']),
            'R2_std': np.std(fold_scores[target]['R2'], ddof=1),
            'MAE_mean': np.mean(fold_scores[target]['MAE']),
            'MAE_std': np.std(fold_scores[target]['MAE'], ddof=1),
            'RMSE_mean': np.mean(fold_scores[target]['RMSE']),
            'RMSE_std': np.std(fold_scores[target]['RMSE'], ddof=1)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Global metrics
    global_metrics = {
        'R2_variance_weighted_mean': np.mean(r2_global_folds),
        'R2_variance_weighted_std': np.std(r2_global_folds),
        'R2_mean': summary_df['R2_mean'].mean(),
        'R2_std': summary_df['R2_mean'].std(),
        'R2_min': summary_df['R2_mean'].min(),
        'R2_max': summary_df['R2_mean'].max()
    }
    
    return summary_df, global_metrics, fold_scores, all_fold_results, r2_global_folds
