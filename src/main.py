# S:\Projects\Stroke-Prediction\src\main.py

import logging
import os
import sys
import pandas as pd

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules from src
from src.config_manager import load_config, setup_paths
from src.data_ingestion import load_data, clean_initial_data, save_data
from src.feature_engineering import create_age_group, create_preprocessor_pipeline
from src.eda import (
    plot_target_distribution, plot_numerical_distributions,
    plot_correlation_heatmap, plot_categorical_vs_target,
    plot_numerical_vs_target_boxplots
)
from src.model_training import (
    train_test_split_data, build_model_pipeline,
    train_and_tune_model, evaluate_model, plot_feature_importance, save_model
)

# --- Set up basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the entire stroke prediction pipeline.
    """
    logger.info("--- Starting Stroke Prediction Pipeline ---")

    # 1. Load Configuration and Setup Paths
    try:
        config_file_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        config = load_config(config_file_path)
        config = setup_paths(config) # This will also create necessary directories
        paths = config['paths']
        model_params = config['model_params']
        logger.info("Configuration and paths loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load configuration or set up paths. Exiting.")
        sys.exit(1)

    # 2. Data Ingestion and Initial Cleaning
    df = None
    try:
        if os.path.exists(paths['final_processed_data_path']):
            df = load_data(paths['final_processed_data_path'])
            logger.info("Loaded final processed data. Skipping initial cleaning.")
        elif os.path.exists(paths['preprocessed_data_path']):
            df = load_data(paths['preprocessed_data_path'])
            logger.info("Loaded preprocessed data. Proceeding to feature engineering.")
        else:
            logger.info("No processed data found. Starting from raw data.")
            raw_df = load_data(paths['raw_data_path'])
            df = clean_initial_data(raw_df, config)
            save_data(df, paths['preprocessed_data_path'])
            logger.info(f"Initial preprocessing complete and data saved to {paths['preprocessed_data_path']}")

    except Exception as e:
        logger.exception("Error during data ingestion or initial cleaning. Exiting.")
        sys.exit(1)

    if df is None:
        logger.error("DataFrame could not be loaded or processed. Exiting.")
        sys.exit(1)

    # 3. Exploratory Data Analysis (EDA)
    logger.info("\n--- Performing EDA ---")
    try:
        plot_target_distribution(df, os.path.join(paths['plots_dir'], 'eda_stroke_distribution.png'))
        plot_numerical_distributions(df, model_params['numerical_features'], os.path.join(paths['plots_dir'], 'eda_numerical_distributions.png'))
        plot_correlation_heatmap(df, model_params['target_column'], os.path.join(paths['plots_dir'], 'eda_correlation_heatmap.png'))
        
        # Determine which categorical columns to plot against target for EDA
        # These are original categorical columns before any OHE, plus 'gender' and 'age_group'
        eda_categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'gender'] # original object columns
        if 'gender' in df.columns and df['gender'].dtype == 'object': # if gender wasn't mapped to 0/1 yet
             eda_categorical_cols.append('gender')
        if 'age_group' in df.columns: # If age_group exists after FE (it will)
             eda_categorical_cols.append('age_group')

        plot_categorical_vs_target(df, model_params['target_column'], os.path.join(paths['plots_dir'], 'eda_categorical_vs_target.png'))
        plot_numerical_vs_target_boxplots(df, model_params['numerical_features'], model_params['target_column'], os.path.join(paths['plots_dir'], 'eda_numerical_vs_target_boxplots.png'))
        logger.info("EDA plots generated and saved.")
    except Exception as e:
        logger.exception("Error during EDA. Continuing pipeline, but investigate.")

    # 4. Feature Engineering
    logger.info("\n--- Performing Feature Engineering ---")
    try:
        df_fe = create_age_group(df, config)
        save_data(df_fe, paths['final_processed_data_path'])
        logger.info(f"Feature engineering complete and data saved to {paths['final_processed_data_path']}")
    except Exception as e:
        logger.exception("Error during feature engineering. Exiting.")
        sys.exit(1)

    # 5. Model Training and Evaluation
    logger.info("\n--- Starting Model Training and Evaluation ---")
    try:
        X_train, X_test, y_train, y_test = train_test_split_data(
            df_fe, model_params['target_column'], model_params['test_size'], model_params['random_state']
        )

        # Determine which categorical columns to OHE *within the pipeline*
        # These should be columns that are *still* of 'object' or 'category' dtype
        # AFTER initial cleaning (which already OHEs some) and feature engineering (age_group).
        categorical_features_for_pipeline_ohe = [
            col for col in X_train.columns
            if X_train[col].dtype in ['object', 'category'] # Look for un-encoded categorical types
        ]
        
        preprocessor = create_preprocessor_pipeline(
            numerical_cols=model_params['numerical_features'],
            categorical_cols=categorical_features_for_pipeline_ohe
        )

        models_to_train_keys = [k for k in config['model_params'].keys() if k not in ['random_state', 'test_size', 'numerical_features', 'target_column']]
        best_overall_roc_auc = 0
        best_overall_model = None
        best_model_name_for_save = "None" # Initialize with a default value

        for model_name_key in models_to_train_keys:
            model_config = config['model_params'][model_name_key]
            model_type_str = " ".join(word.capitalize() for word in model_name_key.split('_')) # e.g., 'logistic_regression' -> 'Logistic Regression'

            pipeline = build_model_pipeline(preprocessor, model_type_str, model_params['random_state'])
            
            # Extract param grid for the classifier part of the pipeline
            param_grid = {f'classifier__{k}': v for k, v in model_config.items()}

            tuned_model_gs = train_and_tune_model(pipeline, X_train, y_train, param_grid, model_type_str)
            
            current_model_best_estimator = tuned_model_gs.best_estimator_
            metrics = evaluate_model(current_model_best_estimator, X_test, y_test, model_type_str, paths['plots_dir'])

            if metrics['roc_auc'] > best_overall_roc_auc:
                best_overall_roc_auc = metrics['roc_auc']
                best_overall_model = current_model_best_estimator
                best_model_name_for_save = model_type_str
            
            # Pass only the necessary arguments to plot_feature_importance
            plot_feature_importance(current_model_best_estimator, model_type_str, paths['plots_dir'])


        if best_overall_model:
            logger.info(f"\nOverall Best Model: {best_model_name_for_save} with ROC AUC: {best_overall_roc_auc:.4f}")
            save_model(best_overall_model, paths['best_pipeline_path'])
            logger.info("Best model pipeline saved.")
        else:
            logger.warning("No best model found or saved.")

    except Exception as e:
        logger.exception("Error during model training or evaluation. Exiting.")
        sys.exit(1)

    logger.info("--- Stroke Prediction Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()