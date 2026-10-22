# Assignment 3: KNN Classification and Regression

This assignment contains two business problems implemented in separate
Jupyter notebooks using the supplied datasets.

## Project Contents

-   `assignment3_partA_sanjeevani.ipynb` --- Sanjeevani Diagnostics:
    diabetes screening using K-Nearest Neighbours (KNN) classification.
-   `assignment3_partB_dashmart.ipynb` --- DashMart: delivery-time
    estimation using K-Nearest Neighbours (KNN) regression.
-   `sanjeevani_diabetes_screening_raw.csv` --- dataset for Part A.
-   `dashmart_delivery_times_raw.csv` --- dataset for Part B.

Keep each CSV in the same folder as its corresponding notebook, or
update the notebook's file path if you store the files elsewhere.

## Requirements

-   Python 3
-   Jupyter Notebook or JupyterLab
-   pandas
-   NumPy
-   Matplotlib
-   scikit-learn

## Part A --- Sanjeevani Diagnostics

**Objective:** Build and evaluate a KNN classification model for
diabetes screening.

The notebook covers data cleaning and preparation, missing values and
duplicates, feature engineering, manual KNN calculations, model
training, scaling comparisons, cross-validation, hyperparameter tuning,
evaluation metrics, threshold-based decisions, and a recommendation for
the screening team.

Key assignment settings: - Stratified train/test split: 75% training and
25% testing. - `random_state=42`. - Start with the seven specified
numeric features. - Use five-fold stratified cross-validation. - Tune
KNN parameters including `k`, `weights`, and distance parameter `p`. -
Evaluate a screening budget that selects the top 250 visitors from 750
test visitors, and compare against the random-selection benchmark.

## Part B --- DashMart Delivery Times

**Objective:** Estimate delivery duration using KNN regression and
assess whether predictions can support delivery promises.

The notebook covers data cleaning, time and distance parsing,
exploratory plots, a median baseline, raw and scaled KNN models,
regression metrics, cross-validation, hour-of-day encoding,
hyperparameter tuning, prediction-error analysis, delivery safety
buffers, example predictions, nearest-neighbour inspection, a rain
counterfactual, coupon-savings analysis, model comparisons, and
recommendations.

Key assignment settings: - Train/test split: 80% training and 20%
testing. - `random_state=42`. - Start with the eight specified
features. - Compare raw features, `StandardScaler`, and
`MinMaxScaler`. - Use five-fold cross-validation and tune KNN parameters
including `k` and `weights`. - Evaluate safety buffers of 0, 2, 3, 4,
and 5 minutes. - Consider the stated goals of MAE below 2.5 minutes and
no more than 10% of orders arriving later than the promised time.

## Reproducibility and Good Practice

-   Run the notebooks from top to bottom so that cells execute in the
    intended order.
-   Keep the original CSV files unchanged; perform cleaning and feature
    engineering in the notebooks.
-   Use the specified `random_state=42` wherever the assignment calls
    for reproducible splits.
-   Fit preprocessing steps on training data only. Use pipelines for
    scaling and modelling to help avoid data leakage.
-   Review the output and the short interpretation accompanying each
    question before submission.

