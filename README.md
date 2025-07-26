# Stroke Prediction ML Pipeline

## Project Overview

This project implements a comprehensive machine learning pipeline to predict the likelihood of a patient having a stroke based on various health and lifestyle factors. It addresses common challenges in real-world healthcare datasets, such as data imbalance and the need for robust preprocessing and model evaluation.

The pipeline covers data ingestion, extensive exploratory data analysis (EDA), feature engineering, model training with hyperparameter tuning, and detailed evaluation, saving the best-performing model for future use.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Model Performance](#results--model-performance)
- [Visualizations](#visualizations)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contact](#contact)

## Features

* **Robust Data Ingestion & Cleaning:** Handles missing values and performs initial categorical data encoding.
* **Comprehensive EDA:** Generates various plots to understand data distributions, correlations, and class imbalance.
* **Feature Engineering:** Creates new features (e.g., `age_group`) to potentially improve model performance.
* **Automated Preprocessing Pipeline:** Uses `ColumnTransformer` for consistent scaling of numerical features and one-hot encoding of categorical features.
* **Imbalance Handling:** Integrates `SMOTE` (Synthetic Minority Over-sampling Technique) within the ML pipeline to address the skewed class distribution.
* **Multiple Model Training:** Trains and evaluates Logistic Regression, Random Forest, and Gradient Boosting Classifiers.
* **Hyperparameter Tuning:** Employs `GridSearchCV` with 5-fold cross-validation to find optimal model parameters, optimizing for `ROC AUC`.
* **Detailed Model Evaluation:** Reports Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrix, and Classification Reports.
* **Feature Importance Analysis:** Visualizes key features for tree-based models.
* **Best Model Persistence:** Saves the entire trained pipeline for easy deployment.

## Dataset

The project utilizes the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle.
It contains 11 features and a target variable (`stroke`) indicating whether the patient had a stroke or not.

**Key Columns:**
* `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke` (target).

## Project Structure
```
Stroke-Prediction/
├── config/
│   └── config.yaml             # Configuration file for paths, model parameters
├── data/
│   ├── raw_data.csv            # Original dataset (e.g., stroke.csv)
│   ├── preprocessed_data.csv   # Intermediate cleaned data
│   └── final_processed_data.csv# Data after feature engineering
├── models/
│   └── best_pipeline.joblib    # Saved best-performing ML pipeline
├── plots/
│   ├── eda_.png               # Exploratory Data Analysis plots
│   ├── roc_curve_.png         # ROC curves for each model
│   ├── confusion_matrix_.png  # Confusion matrices for each model
│   └── feature_importance_.png# Feature importance plots
└── src/
├── init.py             # Makes 'src' a Python package (corrected name)
├── main.py                 # Main execution script for the entire pipeline
├── config_manager.py       # Handles loading configuration and setting up paths
├── data_ingestion.py       # Functions for loading and initial data cleaning
├── eda.py                  # Functions for Exploratory Data Analysis
├── feature_engineering.py  # Functions for creating/transforming features
└── model_training.py       # Functions for model building, tuning, evaluation, saving
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Stroke-Prediction-ML-Pipeline.git](https://github.com/YOUR_USERNAME/Stroke-Prediction-ML-Pipeline.git)
    cd Stroke-Prediction-ML-Pipeline
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install pandas scikit-learn imbalanced-learn matplotlib seaborn joblib pyyaml
    ```
    *(Note: `joblib` and `pyyaml` are usually included if you install scikit-learn and others, but listing them explicitly is good.)*

4.  **Place the dataset:**
    Download the `stroke.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and place it in the `data/` directory as `raw_data.csv`.

## Usage

To run the entire stroke prediction pipeline:

```bash
python src/main.py
```

The script will perform all steps from data ingestion to model training and evaluation, saving plots and the best model to the respective plots/ and models/ directories.

## Results & Model Performance:
The pipeline evaluates multiple models (Logistic Regression, Random Forest, Gradient Boosting) based on ROC AUC score, which is a robust metric for imbalanced datasets. The best-performing model (based on cross-validation and test set performance) is saved.

[After running the script, you can come back here and add brief summary of your best model's performance, e.g., "The Gradient Boosting Classifier achieved the highest ROC AUC of ~0.82 on the test set, demonstrating strong discriminative power even with the imbalanced data."]

## Visualizations:
All generated plots (EDA, ROC Curves, Confusion Matrices, Feature Importances) are saved as .png files in the plots/ directory. Open these files to visualize the insights and model performance.

## Future Enhancements:
Experiment with more advanced imputation techniques for missing BMI values.

Explore other feature engineering strategies.

Test additional classification algorithms (e.g., LightGBM, XGBoost, SVMs).

Implement more sophisticated hyperparameter optimization techniques (e.g., Randomized Search, Bayesian Optimization).

Develop a small Flask/Streamlit application to deploy the saved model for interactive predictions.

Add more extensive logging and error handling.

Implement MLOps practices like DVC for data versioning.

## License:
This project is licensed under the MIT License - see the LICENSE.md file for details.
(Optional: If you plan to add a LICENSE.md file. If not, remove this line or specify "MIT License" directly.)

## Contact:
If you have any questions or suggestions, feel free to reach out:

Your Name: Shabber Zaidi

GitHub: https://github.com/II-Shabber-II

LinkedIn: https://www.linkedin.com/in/shabber-zaidi/
