# S:\Projects\Stroke-Prediction\src\feature_engineering.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_age_group(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Creates 'age_group' categorical feature from 'age' numerical feature.

    Args:
        df (pd.DataFrame): The DataFrame with an 'age' column.
        config (dict): Configuration dictionary with age_bins and age_labels.

    Returns:
        pd.DataFrame: DataFrame with the new 'age_group' column.
    """
    df_fe = df.copy()
    age_bins = config['feature_engineering']['age_bins']
    age_labels = config['feature_engineering']['age_labels']

    if 'age' in df_fe.columns:
        df_fe['age_group'] = pd.cut(df_fe['age'], bins=age_bins, labels=age_labels, right=False, include_lowest=True)
        logger.info("'age_group' feature created.")
    else:
        logger.warning("Cannot create 'age_group': 'age' column not found.")
    return df_fe

def create_preprocessor_pipeline(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Creates a ColumnTransformer for preprocessing numerical (scaling) and
    categorical (one-hot encoding) features.

    Args:
        numerical_cols (list): List of numerical column names to scale.
        categorical_cols (list): List of categorical column names to one-hot encode.

    Returns:
        ColumnTransformer: A preprocessor that can be used in an sklearn Pipeline.
    """
    # Create a list of tuples for the ColumnTransformer
    transformers = []

    if numerical_cols:
        transformers.append(('num', StandardScaler(), numerical_cols))
        logger.info(f"StandardScaler added for numerical columns: {numerical_cols}")
    else:
        logger.warning("No numerical columns specified for scaling.")

    if categorical_cols:
        # Use handle_unknown='ignore' to gracefully handle unseen categories during predict
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        logger.info(f"OneHotEncoder (drop_first=True) added for categorical columns: {categorical_cols}")
    else:
        logger.warning("No categorical columns specified for OneHotEncoding.")

    # The remainder='passthrough' ensures that columns not explicitly transformed are kept
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    logger.info("ColumnTransformer preprocessor pipeline created.")
    return preprocessor