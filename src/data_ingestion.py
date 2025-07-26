# S:\Projects\Stroke-Prediction\src\data_ingestion.py

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The absolute path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        Exception: For other reading errors.
    """
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"The file {file_path} is empty.")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def clean_initial_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Performs initial data cleaning steps:
    - Drops specified columns (e.g., 'id').
    - Imputes missing 'bmi' values.
    - Handles inconsistent 'gender' value 'Other'.
    - Converts 'gender' to numerical.
    - One-hot encodes initial categorical columns.

    Args:
        df (pd.DataFrame): The raw DataFrame.
        config (dict): Configuration dictionary containing cleaning parameters.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy() # Work on a copy to avoid modifying original df

    # Drop specified columns
    cols_to_drop = config['data_cleaning'].get('drop_columns', [])
    for col in cols_to_drop:
        if col in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(col, axis=1)
            logger.info(f"Dropped column: {col}")
        else:
            logger.warning(f"Column '{col}' specified for dropping not found in DataFrame.")

    # Handle missing 'bmi' values
    bmi_strategy = config['data_cleaning'].get('impute_bmi_strategy', 'mean')
    if 'bmi' in df_cleaned.columns and df_cleaned['bmi'].isnull().any():
        if bmi_strategy == 'mean':
            df_cleaned['bmi'] = df_cleaned['bmi'].fillna(df_cleaned['bmi'].mean())
            logger.info("Missing 'bmi' values imputed with mean.")
        elif bmi_strategy == 'median':
            df_cleaned['bmi'] = df_cleaned['bmi'].fillna(df_cleaned['bmi'].median())
            logger.info("Missing 'bmi' values imputed with median.")
        else:
            logger.warning(f"Unknown BMI imputation strategy: '{bmi_strategy}'. No imputation performed.")
    elif 'bmi' not in df_cleaned.columns:
        logger.warning("BMI column not found for imputation.")
    else:
        logger.info("No missing 'bmi' values found, no imputation needed.")


    # Handle inconsistent 'gender' value 'Other'
    gender_to_remove = config['data_cleaning'].get('gender_to_remove')
    if 'gender' in df_cleaned.columns and gender_to_remove in df_cleaned['gender'].unique():
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned[df_cleaned['gender'] != gender_to_remove]
        logger.info(f"Removed '{gender_to_remove}' from 'gender' column. {initial_rows - len(df_cleaned)} rows removed.")
    elif 'gender' not in df_cleaned.columns:
        logger.warning("Gender column not found for 'Other' removal.")
    else:
        logger.info(f"'{gender_to_remove}' not found in 'gender' column, no rows removed.")


    # Convert 'gender' to numerical (binary)
    gender_mapping = config['data_cleaning'].get('gender_mapping')
    if 'gender' in df_cleaned.columns and gender_mapping:
        if set(df_cleaned['gender'].unique()) <= set(gender_mapping.keys()):
            df_cleaned['gender'] = df_cleaned['gender'].map(gender_mapping)
            logger.info("Gender column converted to numerical.")
        else:
            logger.warning("Gender mapping incomplete for existing gender values. Skipping direct mapping.")
            # Fallback to one-hot if not perfectly mappable after removing "Other"
            df_cleaned = pd.get_dummies(df_cleaned, columns=['gender'], drop_first=True, dtype=int)
            logger.warning("Gender column had unmapped values, so it was One-Hot Encoded instead of direct mapped.")
    elif 'gender' in df_cleaned.columns:
        logger.warning("No gender mapping provided in config. Skipping gender conversion.")


    # Columns for One-Hot Encoding (initial set, before age_group)
    # These were hardcoded in previous step 4, let's identify them based on object dtypes
    # We will exclude gender if it was directly mapped.
    initial_ohe_cols = [col for col in df_cleaned.select_dtypes(include='object').columns if col != 'gender']

    if initial_ohe_cols:
        df_cleaned = pd.get_dummies(df_cleaned, columns=initial_ohe_cols, drop_first=True, dtype=int)
        logger.info(f"Initial categorical columns {initial_ohe_cols} One-Hot Encoded.")
    else:
        logger.info("No initial categorical columns found for One-Hot Encoding.")

    logger.info(f"Data cleaning complete. New shape: {df_cleaned.shape}")
    return df_cleaned

def save_data(df: pd.DataFrame, file_path: str, index: bool = False):
    """
    Saves a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The absolute path to save the CSV.
        index (bool): Whether to write the DataFrame index as a column.
    """
    try:
        df.to_csv(file_path, index=index)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise