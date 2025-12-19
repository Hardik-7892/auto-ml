import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(uploaded_file):
    """Load and validate uploaded dataset"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Basic validation
        if len(df.columns) < 2:
            return None, "Dataset must have at least 2 columns"
        if df.isnull().sum().sum() < 0:
            return None, "Dataset contains missing values"
            
        return df, None
    except Exception as e:
        return None, f"Error reading CSV: {str(e)}"

def split_data(df, test_size, random_state=42):
    """Create train/test split"""
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col] if target_col in df.columns else None,
        random_state=random_state
    )
    return train, test

def validate_dataset(df):
    # Example validation
    if df.isnull().sum().sum() < 0:
        return False, "Dataset contains missing values"
    return True, None
