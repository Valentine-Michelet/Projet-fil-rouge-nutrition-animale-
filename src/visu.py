"""
src/visu.py - Visualization functions (matplotlib & plotly)
"""

from pyexpat import features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union
import math

def plot_embeddings_3d_pca(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    hue_col: str = 'OOD',
    title: str = "3D PCA - Embeddings",
    color_map: Optional[Dict] = None,
) -> Tuple[go.Figure, PCA]:
    """
    Plot 3D PCA visualization of embeddings with coloring by specified column.
    
    Args:
        embeddings: 2D numpy array of embeddings (n_samples, n_features)
        df: DataFrame containing the hue_col for coloring
        hue_col: Column name for coloring points (default: 'OOD')
                 - If 'OOD': uses default colors {0: blue, 1: red}
                 - Otherwise: generates automatic colors for unique values
        title: Plot title
        color_map: Optional dictionary mapping values to hex colors
                   Example: {0: '#1f77b4', 1: '#d62728'}
    
    Returns:
        Tuple of (plotly figure, fitted PCA object)
    """
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Fit PCA 3D
    pca = PCA(n_components=3)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    # Create DataFrame with PCA components
    pca_df = pd.DataFrame({
        'PC1': embeddings_pca[:, 0],
        'PC2': embeddings_pca[:, 1],
        'PC3': embeddings_pca[:, 2],
        'HueCol': df[hue_col].values
    })
    
    # Default color map for OOD: blue (0) / red (1)
    if color_map is None:
        if hue_col == 'OOD':
            unique_vals = sorted(pca_df['HueCol'].unique())
            if set(unique_vals) == {0, 1}:
                color_map = {0: '#1f77b4', 1: '#d62728'}
            else:
                # Fallback: auto colors
                colors = px.colors.qualitative.Set1
                color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_vals)}
        else:
            # Auto color for non-OOD columns
            unique_vals = sorted(pca_df['HueCol'].unique())
            colors = px.colors.qualitative.Set1
            color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_vals)}
    
    # Convert hue values to string labels for legend
    pca_df['HueLabel'] = pca_df['HueCol'].astype(str)
    if hue_col == 'OOD':
        pca_df['HueLabel'] = pca_df['HueCol'].map({0: 'IID', 1: 'OOD'})
    
    # Create discrete color map for plotly
    discrete_color_map = {}
    for val in pca_df['HueCol'].unique():
        label = pca_df[pca_df['HueCol'] == val]['HueLabel'].iloc[0]
        discrete_color_map[label] = color_map.get(val, '#808080')
    
    # Plot 3D scatter
    fig = px.scatter_3d(
        pca_df,
        x='PC1', y='PC2', z='PC3',
        color='HueLabel',
        color_discrete_map=discrete_color_map,
        title=f'{title}<br>PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}, PC3: {pca.explained_variance_ratio_[2]:.1%}',
        height=800,
        labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3', 'HueLabel': hue_col}
    )
    
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=60))
    
    return fig, pca

def plot_acp_3d(
    df: pd.DataFrame, 
    features: List[str], 
    hue_col: Optional[str] = None,
    n_components: int = 3,
    title: str = "3D PCA Visualization",
) -> Tuple[go.Figure, PCA]:
    
    # Préparation des données
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # --- GESTION DU HUE (COULEUR & BINAIRE) ---
    symbol_col = None
    color_discrete_map = None
    
    if hue_col is not None and hue_col in df.columns:
        pca_df['Color'] = df[hue_col].values
        color_col = 'Color'
        
        # Vérification si c'est binaire (2 valeurs uniques)
        if df[hue_col].nunique() == 2:
            # On ajoute des symboles différents pour le binaire
            symbol_col = 'Color'
            # Optionnel : forcer des couleurs très distinctes (ex: Rouge/Bleu)
            color_discrete_map = {
                df[hue_col].unique()[0]: "#1f77b4", 
                df[hue_col].unique()[1]: "#d62728"
            }
    else:
        color_col = None

    # Création du plot
    fig = px.scatter_3d(
        pca_df,
        x='PC1', y='PC2', z='PC3',
        color=color_col,
        symbol=symbol_col, # Ajout dynamique des symboles
        color_discrete_map=color_discrete_map, # Couleurs forcées si binaire
        title=f'{title}<br>Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}, PC3={pca.explained_variance_ratio_[2]:.1%}',
        height=800,
        opacity=0.7 # Un peu de transparence aide à voir les amas en 3D
    )
    
    # Amélioration du style des points
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))
    
    return fig, pca

def plot_variance_explained(
    df: pd.DataFrame,
    features: List[str],
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot cumulative variance explained by PCA components.
    
    Args:
        pca: Fitted PCA object
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained_var = np.cumsum(pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(range(1, len(explained_var) + 1), 
           pca.explained_variance_ratio_, 
           alpha=0.7, 
           label='Individual')
    ax.plot(range(1, len(explained_var) + 1), 
            explained_var, 
            'ro-', 
            linewidth=2, 
            label='Cumulative')
    
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('PCA Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_circle(
    df: pd.DataFrame,
    features: List[str],
    simplified_names: dict = None,
    pc1: int = 0,
    pc2: int = 1,
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Plot PCA correlation circle (cercle de corrélation).
    
    Args:
        df: Input DataFrame
        features: List of feature columns
        pca: Fitted PCA object
        pc1: Index of first principal component
        pc2: Index of second principal component
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Standardize data
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    # Get loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    
    # Plot arrows
    for i, feature in enumerate(features):
        display_name = simplified_names.get(feature, feature)
        ax.arrow(
            0, 0,
            loadings[i, pc1],
            loadings[i, pc2],
            head_width=0.05,
            head_length=0.05,
            fc='blue',
            ec='blue'
        )
        
        ax.text(
            loadings[i, pc1] * 1.15,
            loadings[i, pc2] * 1.15,
            display_name,
            fontsize=10,
            ha='center',
            va='center'
        )
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(f'PC{pc1+1} ({pca.explained_variance_ratio_[pc1]:.1%})')
    ax.set_ylabel(f'PC{pc2+1} ({pca.explained_variance_ratio_[pc2]:.1%})')
    ax.set_title('PCA Correlation Circle')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig



def plot_correlation_matrix(
    df_features: List[str], 
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = True,
    cmap: str = 'coolwarm',
    title: str = "Correlation Matrix"
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df_features: List of feature columns
        figsize: Figure size
        annot: Whether to annotate cells with values
        cmap: Colormap name
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Calculate correlation
    corr = df_features.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr,
        annot=annot,
        fmt='.2f',
        cmap=cmap,
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )
    
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig



def plot_predictions_vs_true(
    models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    target_vars: List[str],
    simplified_names: Dict[str, str],
    title: str = "Predictions vs True Values",
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Plot predictions vs true values for all targets.
    
    Args:
        models: Dictionary of fitted models {target: model}
        X_test: Test features
        y_test: True test targets
        target_vars: List of target variable names
        simplified_names: Mapping of names to simplified versions
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_targets = len(target_vars)
    n_cols = 3
    n_rows = math.ceil(n_targets / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, target in enumerate(target_vars):
        ax = axes[idx]
        
        y_true = y_test[target]
        y_pred = models[target].predict(X_test)
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(simplified_names.get(target, target))
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(target_vars), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
    return fig


def plot_feature_importance(
    models: Dict,
    X_test: pd.DataFrame,
    target_vars: List[str],
    simplified_names: Dict[str, str],
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot feature importance for XGBoost models.
    
    Args:
        models: Dictionary of fitted XGBoost models
        X_test: Test features DataFrame (used for feature names)
        target_vars: List of target variable names
        simplified_names: Mapping of names to simplified versions
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_targets = len(target_vars)
    n_cols = 2
    n_rows = math.ceil(n_targets / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    feature_names = X_test.columns.tolist()
    
    for idx, target in enumerate(target_vars):
        ax = axes[idx]
        
        importances = models[target].feature_importances_
        feature_importance_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        ax.barh(feature_importance_df.index, feature_importance_df.values, alpha=0.7)
        ax.set_xlabel('Importance')
        ax.set_title(simplified_names.get(target, target))
        ax.invert_yaxis()
    
    # Hide unused subplots
    for idx in range(len(target_vars), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
    return fig


def plot_distribution(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 10),
    title: str = "Feature Distributions"
) -> plt.Figure:
    """
    Plot distributions of numerical features.
    
    Args:
        df: Input DataFrame
        features: List of columns to plot (if None, uses all columns in df)
        figsize: Figure size
        title: Plot title
    Returns:
        Matplotlib figure
    """
    # If features not provided, use all columns from df
    if features is None:
        features = df.columns.tolist()
    
    n_features = len(features)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        data = df[feature].dropna()
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(feature)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(features), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig



def plot_ml_results(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    x_col: str = 'Variable cible',
    y_col: str = 'R2',
    title: str = "Performance Comparison",
    baseline_val: Optional[float] = None,
    figsize: Optional[Tuple[int, int]] = None,
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    Fonction unique pour tracer soit une grille de résultats (si dict) 
    soit un graphique unique (si DataFrame).
    """
    classic_colors = colors or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # --- CAS 1 : GRILLE DE GRAPHIQUES (Dictionnaire) ---
    if isinstance(data, dict):
        n_scenarios = len(data)
        if n_scenarios == 0: return None
        
        n_cols = 2
        n_rows = (n_scenarios + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (16, 5 * n_rows))
        axes = axes.flatten() if n_scenarios > 1 else [axes]

        for i, (name, df) in enumerate(data.items()):
            ax = axes[i]
            df_sorted = df.sort_values(by=y_col, ascending=False).reset_index(drop=True)
            x_pos = np.arange(len(df_sorted))
            
            bars = ax.bar(x_pos, df_sorted[y_col], edgecolor='black', alpha=0.8,
                          color=[classic_colors[j % len(classic_colors)] for j in range(len(df_sorted))])
            
            ax.set_title(f"Scénario : {name}", fontweight='bold', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_sorted[x_col], rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        for j in range(i + 1, len(axes)): axes[j].axis('off')

    # --- CAS 2 : GRAPHIQUE UNIQUE (DataFrame) ---
    else:
        df_sorted = data.sort_values(by=y_col, ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=figsize or (12, 6))
        x_pos = np.arange(len(df_sorted))
        
        bars = ax.bar(x_pos, df_sorted[y_col], edgecolor='black', alpha=0.8,
                      color=[classic_colors[j % len(classic_colors)] for j in range(len(df_sorted))])
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_ylabel(y_col, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_sorted[x_col], rotation=15, ha='right')
        
        if baseline_val is not None:
            ax.axhline(y=baseline_val, color='forestgreen', linestyle='--', alpha=0.6)
            
        ax.set_ylim(0, max(df_sorted[y_col].max() * 1.15, 1.0))
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.4f}', 
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig