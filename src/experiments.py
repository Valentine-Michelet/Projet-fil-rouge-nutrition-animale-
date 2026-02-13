"""
src/experiments.py - Model training and experimentation functions
This is the core module for all XGBoost experiment scenarios.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
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
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


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
