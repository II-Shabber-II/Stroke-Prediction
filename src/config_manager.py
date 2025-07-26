# S:\Projects\Stroke-Prediction\src\config_manager.py

import yaml
import os
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_config(config_path: str) -> dict:
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): The path to the config.yaml file.

    Returns:
        dict: The loaded configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config file: {e}")
            raise

def setup_paths(config: dict) -> dict:
    """
    Sets up absolute paths for data, models, and plots directories.

    Args:
        config (dict): The loaded configuration dictionary.

    Returns:
        dict: Updated configuration with absolute paths.
    """
    base_dir = config['project_paths']['base_dir']
    
    config['paths'] = {
        'data_dir': os.path.join(base_dir, config['project_paths']['data_dir_name']),
        'models_dir': os.path.join(base_dir, config['project_paths']['models_dir_name']),
        'plots_dir': os.path.join(base_dir, config['project_paths']['plots_dir_name']),
        'raw_data_path': os.path.join(
            os.path.join(base_dir, config['project_paths']['data_dir_name']),
            config['data_files']['raw_data_name']
        ),
        'preprocessed_data_path': os.path.join(
            os.path.join(base_dir, config['project_paths']['data_dir_name']),
            config['data_files']['preprocessed_data_name']
        ),
        'final_processed_data_path': os.path.join(
            os.path.join(base_dir, config['project_paths']['data_dir_name']),
            config['data_files']['final_processed_data_name']
        ),
        'scaler_path': os.path.join(
            os.path.join(base_dir, config['project_paths']['models_dir_name']),
            config['output_model_names']['scaler'] # Assuming scaler is saved separately, though better if part of pipeline
        ),
        'best_pipeline_path': os.path.join(
            os.path.join(base_dir, config['project_paths']['models_dir_name']),
            config['output_model_names']['best_pipeline']
        )
    }

    # Create directories if they don't exist
    for dir_key in ['data_dir', 'models_dir', 'plots_dir']:
        path = config['paths'][dir_key]
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory: {path}")
    
    logger.info("Project paths set up and directories ensured.")
    return config