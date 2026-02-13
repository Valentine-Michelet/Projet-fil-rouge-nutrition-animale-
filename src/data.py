"""
src/data.py - Data loading functions (simple, no classes)
"""

import pandas as pd
from typing import Tuple


def load_data_excel(file_path: str) -> pd.DataFrame:
    """
    Load data from Excel file and clean column names.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        DataFrame with cleaned columns
    """
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df['Classe'] = df['Classe'].str.strip()
    df['Nom'] = df['Nom'].str.strip()
    return df


def load_data_csv(file_path: str, sep: str = ',') -> pd.DataFrame:
    """
    Load data from CSV file and clean column names.
    
    Args:
        file_path: Path to CSV file
        sep: CSV separator
        
    Returns:
        DataFrame with cleaned columns
    """
    df = pd.read_csv(file_path, sep=sep)
    df.columns = df.columns.str.strip()
    if 'Classe' in df.columns:
        df['Classe'] = df['Classe'].str.strip()
    if 'Nom' in df.columns:
        df['Nom'] = df['Nom'].str.strip()
    return df


def convert_numeric_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Convert specified columns to numeric type, handling errors gracefully.
    
    Args:
        df: Input DataFrame
        columns: List of columns to convert
        
    Returns:
        DataFrame with converted columns
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    return df_copy
