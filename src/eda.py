# S:\Projects\Stroke-Prediction\src\eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def plot_target_distribution(df: pd.DataFrame, plot_path: str):
    """
    Plots the distribution of the target variable 'stroke'.

    Args:
        df (pd.DataFrame): The DataFrame.
        plot_path (str): Absolute path to save the plot.
    """
    if 'stroke' not in df.columns:
        logger.warning("Cannot plot target distribution: 'stroke' column not found.")
        return

    plt.figure(figsize=(6, 4))
    sns.countplot(x='stroke', data=df)
    plt.title('Distribution of Stroke Cases')
    plt.xlabel('Stroke (0: No, 1: Yes)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No Stroke', 'Stroke'])
    plt.savefig(plot_path)
    plt.close() # Close plot to free memory
    logger.info(f"Target distribution plot saved to '{plot_path}'")

    stroke_counts = df['stroke'].value_counts()
    logger.info(f"Stroke counts:\n{stroke_counts}")
    if len(df) > 0:
        logger.info(f"Percentage of stroke cases: {stroke_counts.get(1, 0) / len(df) * 100:.2f}%")
    logger.info("Observation: The dataset is highly imbalanced.")


def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: list, plot_path: str):
    """
    Plots histograms for numerical feature distributions.

    Args:
        df (pd.DataFrame): The DataFrame.
        numerical_cols (list): List of numerical column names to plot.
        plot_path (str): Absolute path to save the plot.
    """
    present_cols = [col for col in numerical_cols if col in df.columns]
    if not present_cols:
        logger.warning("No specified numerical columns found for distribution plots.")
        return

    plt.figure(figsize=(5 * len(present_cols), 5))
    for i, col in enumerate(present_cols):
        plt.subplot(1, len(present_cols), i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Numerical feature distributions plot saved to '{plot_path}'")

def plot_correlation_heatmap(df: pd.DataFrame, target_col: str, plot_path: str):
    """
    Plots a correlation heatmap of features with the target variable.

    Args:
        df (pd.DataFrame): The DataFrame.
        target_col (str): The name of the target column.
        plot_path (str): Absolute path to save the plot.
    """
    if target_col not in df.columns:
        logger.warning(f"Cannot plot correlation heatmap: Target column '{target_col}' not found.")
        return

    correlations = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    logger.info(f"Correlation with '{target_col}':\n{correlations}")

    # Exclude the target itself from the top correlated features, take top 10
    top_correlated_features = correlations.drop(target_col, errors='ignore').index[:10].tolist()
    
    # Ensure all selected features are actually present in df.columns
    features_for_heatmap = [col for col in top_correlated_features + [target_col] if col in df.columns]
    
    if len(features_for_heatmap) < 2: # Need at least 2 columns for a meaningful correlation matrix
        logger.warning("Not enough numeric features to plot correlation heatmap.")
        return

    corr_matrix_top = df[features_for_heatmap].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_top, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Correlation Matrix of Top Features with {target_col}')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Correlation heatmap saved to '{plot_path}'")

def plot_categorical_vs_target(df: pd.DataFrame, target_col: str, plot_path: str):
    """
    Plots countplots for categorical features against the target variable.
    Handles one-hot encoded columns by grouping them by original prefix.

    Args:
        df (pd.DataFrame): The DataFrame.
        target_col (str): The name of the target column.
        plot_path (str): Absolute path to save the plot.
    """
    if target_col not in df.columns:
        logger.warning(f"Cannot plot categorical vs target: Target column '{target_col}' not found.")
        return

    # Identify original categorical columns and one-hot encoded counterparts
    # This requires knowing original column names (e.g., from config or initial analysis)
    # For now, let's assume common patterns for one-hot encoded columns
    # A more robust solution might pass the original categorical column names.
    
    # Identify binary flag columns directly if they are not OHE
    binary_flag_cols = ['hypertension', 'heart_disease']
    
    # Attempt to group OHE columns by their original feature if present
    ohe_prefixes = ['ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_group'] # age_group from FE
    
    # Filter columns to only include those relevant and present
    plot_cols = []
    
    # Add direct binary flags
    for col in binary_flag_cols:
        if col in df.columns:
            plot_cols.append(col)

    # Add gender if it's still categorical or converted to 0/1 (binary)
    if 'gender' in df.columns and df['gender'].dtype != 'object': # Assuming it's already numerical 0/1
        plot_cols.append('gender')
    
    # Group OHE columns
    for prefix in ohe_prefixes:
        # Find all columns starting with this prefix AND ending with 0 or 1 etc. (assuming boolean like)
        # This part requires careful handling if you don't drop_first or have many categories
        # A simpler approach for just plotting OHE output: plot the _1 column for each if drop_first=True
        relevant_ohe_cols = [col for col in df.columns if col.startswith(prefix + '_') and col != f'{prefix}_Other']
        if relevant_ohe_cols:
            plot_cols.extend(relevant_ohe_cols)
        elif prefix in df.columns and df[prefix].dtype == 'object': # Original column not yet OHE
             plot_cols.append(prefix)


    plot_cols = list(set([col for col in plot_cols if col in df.columns and col != target_col])) # Ensure unique and existing

    if not plot_cols:
        logger.warning("No relevant categorical columns found for plotting 'vs. Target'.")
        return

    num_plots = len(plot_cols)
    if num_plots == 0:
        logger.warning("No categorical columns to plot against target.")
        return

    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols # Ceiling division

    plt.figure(figsize=(num_cols * 6, num_rows * 5)) # Dynamic figure size
    for i, col in enumerate(plot_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.countplot(x=col, hue=target_col, data=df)
        plt.title(f'{col} vs. {target_col}')
        plt.ylabel('Count')
        plt.xlabel('')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Categorical features vs. target plots saved to '{plot_path}'")

def plot_numerical_vs_target_boxplots(df: pd.DataFrame, numerical_cols: list, target_col: str, plot_path: str):
    """
    Plots box plots for numerical features against the target variable.

    Args:
        df (pd.DataFrame): The DataFrame.
        numerical_cols (list): List of numerical column names to plot.
        target_col (str): The name of the target column.
        plot_path (str): Absolute path to save the plot.
    """
    if target_col not in df.columns:
        logger.warning(f"Cannot plot numerical vs target: Target column '{target_col}' not found.")
        return
    present_cols = [col for col in numerical_cols if col in df.columns]
    if not present_cols:
        logger.warning("No specified numerical columns found for numerical vs. target plots.")
        return

    plt.figure(figsize=(5 * len(present_cols), 5))
    for i, col in enumerate(present_cols):
        plt.subplot(1, len(present_cols), i + 1)
        sns.boxplot(x=target_col, y=col, data=df)
        plt.title(f'{col} vs. {target_col}')
        plt.xticks([0, 1], ['No Stroke', 'Stroke'])
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Numerical features vs. target box plots saved to '{plot_path}'")