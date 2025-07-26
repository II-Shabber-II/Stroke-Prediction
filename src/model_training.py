# S:\Projects\Stroke-Prediction\src\model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.pipeline import Pipeline as ImbPipeline # Use imblearn's pipeline for SMOTE
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from typing import Dict, Any, Tuple
from collections import Counter # <-- Added this import

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_test_split_data(df: pd.DataFrame, target_col: str, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets, stratified by the target column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the randomness of the splitting.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
    """
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame for splitting.")
        raise ValueError(f"Target column '{target_col}' not found.")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")
    logger.info(f"Original target distribution: {Counter(y)}")
    logger.info(f"Training target distribution: {Counter(y_train)}")
    logger.info(f"Test target distribution: {Counter(y_test)}")
    return X_train, X_test, y_train, y_test

def build_model_pipeline(preprocessor: Pipeline, model_type: str, random_state: int) -> ImbPipeline:
    """
    Builds a full machine learning pipeline including preprocessing, SMOTE, and the model.

    Args:
        preprocessor (Pipeline): The ColumnTransformer/Pipeline for preprocessing.
        model_type (str): Type of model to use ('Logistic Regression', 'Random Forest', 'Gradient Boosting').
        random_state (int): Random state for reproducible results.

    Returns:
        ImbPipeline: The scikit-learn pipeline.
    """
    steps = [
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=random_state)), # SMOTE applied after preprocessing (within pipeline)
    ]

    if model_type == 'Logistic Regression':
        steps.append(('classifier', LogisticRegression(random_state=random_state, class_weight='balanced', solver='liblinear')))
    elif model_type == 'Random Forest':
        steps.append(('classifier', RandomForestClassifier(random_state=random_state, class_weight='balanced')))
    elif model_type == 'Gradient Boosting':
        steps.append(('classifier', GradientBoostingClassifier(random_state=random_state)))
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")
    
    pipeline = ImbPipeline(steps)
    logger.info(f"Pipeline built for {model_type}.")
    return pipeline

def train_and_tune_model(pipeline: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series, param_grid: Dict, model_name: str) -> GridSearchCV:
    """
    Trains and tunes a model using GridSearchCV.

    Args:
        pipeline (ImbPipeline): The scikit-learn pipeline to tune.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        param_grid (Dict): Dictionary of parameters to search over.
        model_name (str): Name of the model for logging.

    Returns:
        GridSearchCV: The fitted GridSearchCV object.
    """
    logger.info(f"Starting GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5, # Using 5-fold cross-validation
        scoring='roc_auc', # Optimize for ROC AUC due to imbalanced data
        n_jobs=-1, # Use all available cores
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"GridSearchCV for {model_name} completed.")
    logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
    logger.info(f"Best ROC AUC score for {model_name}: {grid_search.best_score_:.4f}")
    return grid_search

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, plots_dir: str):
    """
    Evaluates a trained model and logs/plots various metrics.

    Args:
        model (Any): The trained model (or GridSearchCV object's best_estimator_).
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        model_name (str): Name of the model for logging and plot titles.
        plots_dir (str): Directory to save plots.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    logger.info(f"\n--- {model_name} Evaluation Metrics ---")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  Confusion Matrix:\n{conf_matrix}")
    logger.info(f"  Classification Report:\n{class_report}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend()
    plt.grid(True)
    roc_curve_path = os.path.join(plots_dir, f'roc_curve_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(roc_curve_path)
    plt.close()
    logger.info(f"ROC Curve plot saved to '{roc_curve_path}'")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No Stroke', 'Predicted Stroke'],
                yticklabels=['Actual No Stroke', 'Actual Stroke'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_plot_path = os.path.join(plots_dir, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(cm_plot_path)
    plt.close()
    logger.info(f"Confusion Matrix plot saved to '{cm_plot_path}'")

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'roc_auc': roc_auc}

def plot_feature_importance(model: Pipeline, model_name: str, plots_dir: str): # <--- Removed X_train_cols
    """
    Plots feature importance for tree-based models (if available).

    Args:
        model (Pipeline): The trained pipeline (best_estimator_ from GridSearchCV).
        model_name (str): Name of the model.
        plots_dir (str): Directory to save plots.
    """
    # Access the classifier from the pipeline
    classifier = model.named_steps['classifier']

    if hasattr(classifier, 'feature_importances_'):
        logger.info(f"Generating feature importance plot for {model_name}...")

        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        feature_names_out = preprocessor.get_feature_names_out()
        
        feature_importances = pd.Series(classifier.feature_importances_, index=feature_names_out).sort_values(ascending=False)

        plt.figure(figsize=(10, 7))
        # Ensure there are features to plot before calling barplot
        if not feature_importances.empty:
            sns.barplot(x=feature_importances.head(15), y=feature_importances.head(15).index) # Plot top 15
            plt.title(f'Top 15 Feature Importances for {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            importance_plot_path = os.path.join(plots_dir, f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
            plt.savefig(importance_plot_path)
            plt.close()
            logger.info(f"Feature importance plot saved to '{importance_plot_path}'")
        else:
            logger.warning(f"No feature importances to plot for {model_name} after preprocessing.")
    else:
        logger.info(f"Model '{model_name}' does not have 'feature_importances_' attribute.")

def save_model(model: Any, file_path: str):
    """
    Saves a trained model using joblib.

    Args:
        model (Any): The model object to save.
        file_path (str): The absolute path to save the model.
    """
    try:
        joblib.dump(model, file_path)
        logger.info(f"Model saved successfully to '{file_path}'")
    except Exception as e:
        logger.error(f"Error saving model to '{file_path}': {e}")
        raise

def load_model(file_path: str) -> Any:
    """
    Loads a trained model using joblib.

    Args:
        file_path (str): The absolute path to the saved model.

    Returns:
        Any: The loaded model object.
    """
    if not os.path.exists(file_path):
        logger.error(f"Model file not found: {file_path}")
        raise FileNotFoundError(f"Model file not found: {file_path}")
    try:
        model = joblib.load(file_path)
        logger.info(f"Model loaded successfully from '{file_path}'")
        return model
    except Exception as e:
        logger.error(f"Error loading model from '{file_path}': {e}")
        raise