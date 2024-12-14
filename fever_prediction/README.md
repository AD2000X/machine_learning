# Introduction: Infrared Thermography Temperature Prediction using Machine Learning

## Overview
This project demonstrates the use of sensor-collected data to predict oral temperatures and identify fever using machine learning techniques. The dataset used for this project comes from the [Infrared Thermography Temperature Dataset](https://archive.ics.uci.edu/dataset/925/infrared+thermography+temperature+dataset), which contains temperature readings from various infrared imaging locations, along with environmental and demographic data.

### Background
Fever is a critical symptom of many infectious diseases, including Severe Acute Respiratory Syndrome (SARS), Influenza A (H1N1), Ebola Virus Disease (EVD), and Coronavirus (COVID-19). While fever screening alone cannot halt epidemics, it is a valuable part of broader risk management strategies ([1](https://doi.org/10.3390/s22010215)).

Infrared thermography allows for non-contact temperature measurements, such as using a forehead scan, and imputing more meaningful oral temperatures. This project leverages these readings to predict:

1. **Oral temperature (in both fast mode and monitor mode).**
2. **Fever status**, defined as oral temperature ≥ 37.5°C.

### Dataset Details
The dataset contains:
- **33 features**, including:
  - Demographic attributes: gender, age, ethnicity.
  - Environmental conditions: ambient temperature, humidity, distance.
  - Thermal image readings from various body sites.
- **Target variables**:
  - `aveOralF`: Oral temperature in fast mode (regression target).
  - `aveOralM`: Oral temperature in monitor mode (regression target).
  - Fever status for `aveOralF` and `aveOralM` (classification targets).

## Project Goals
1. Predict oral temperatures (`aveOralF` and `aveOralM`) using machine learning regression techniques.
2. Classify fever status based on oral temperature readings:
   - Fever with respect to `aveOralF`.
   - Fever with respect to `aveOralM`.

## References
1. Wang, Q., Zhou, Y., Ghassemi, P., McBride, D., Casamento, J., & Pfefer, T. (2022). *Infrared Thermography for Measuring Elevated Body Temperature: Clinical Accuracy, Calibration, and Evaluation*. Sensors. [DOI: 10.3390/s22010215](https://doi.org/10.3390/s22010215)
2. [UCI Machine Learning Repository: Infrared Thermography Temperature Dataset](https://archive.ics.uci.edu/dataset/925/infrared+thermography+temperature+dataset)


## Table of Contents
- [Machine Learning Models](#machine-learning-models)
  - [Linear Regression](#1-linear-regression)
  - [Stochastic Gradient Descent (SGD) Regressor](#2-stochastic-gradient-descent-sgd-regressor)
  - [Polynomial Regression](#3-polynomial-regression)
  - [Decision Tree](#4-decision-tree)
  - [K-Nearest Neighbors (KNN)](#5-k-nearest-neighbors-knn)
  - [Random Forest](#6-random-forest)
  - [Gradient Boosting](#7-gradient-boosting)
  - [Adaptive Boosting (AdaBoost)](#8-adaptive-boosting-adaboost)
  - [Support Vector Machine (SVM)](#9-support-vector-machine-svm)
  - [Logistic Regression](#10-logistic-regression)
- [Deep Learning](#deep-learning)
  - [Multi-Layer Perceptron (MLP)](#1-multi-layer-perceptron-mlp)
- [Shared Techniques Across Models](#shared-techniques-across-models)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)

---

## Machine Learning Models

### 1. Linear Regression
- **Library**: `sklearn.linear_model.LinearRegression`
- **Purpose**: Models the linear relationship between features and target for regression problems.
- **Usage**: Predicting continuous variables (e.g., body temperature).

### 2. Stochastic Gradient Descent (SGD) Regressor
- **Library**: `sklearn.linear_model.SGDRegressor`
- **Purpose**: Optimizes linear models using stochastic gradient descent, suitable for large datasets.

### 3. Polynomial Regression
- **Library**: `sklearn.preprocessing.PolynomialFeatures` combined with `LinearRegression`
- **Purpose**: Captures non-linear relationships by expanding features to higher-degree terms.
- **Implementation**: Used within a pipeline (`make_pipeline`).

### 4. Decision Tree
- **Libraries**:
  - `sklearn.tree.DecisionTreeRegressor` (for regression)
  - `sklearn.tree.DecisionTreeClassifier` (for classification)
- **Purpose**: Constructs tree structures to make predictions for regression and classification tasks.

### 5. K-Nearest Neighbors (KNN)
- **Library**: `sklearn.neighbors.KNeighborsRegressor`
- **Purpose**: Predicts outcomes based on the nearest data points in the feature space.
- **Customization**: Experimented with different values of `k` (e.g., `k=1` and `k=14`).

### 6. Random Forest
- **Libraries**:
  - `sklearn.ensemble.RandomForestRegressor` (for regression)
  - `sklearn.ensemble.RandomForestClassifier` (for classification)
- **Purpose**: Combines predictions from multiple decision trees to improve generalization.

### 7. Gradient Boosting
- **Library**: `sklearn.ensemble.GradientBoostingClassifier`
- **Purpose**: Sequentially builds decision trees to minimize prediction errors iteratively.

### 8. Adaptive Boosting (AdaBoost)
- **Library**: `sklearn.ensemble.AdaBoostClassifier`
- **Purpose**: Focuses on difficult samples by adjusting weights in each iteration.

### 9. Support Vector Machine (SVM)
- **Library**: `sklearn.svm.SVC`
- **Purpose**: Finds the optimal hyperplane for classification, effective in high-dimensional spaces.

### 10. Logistic Regression
- **Library**: `sklearn.linear_model.LogisticRegression`
- **Purpose**: Solves binary classification problems using a probabilistic approach with sigmoid functions.

---

## Deep Learning

### 1. Multi-Layer Perceptron (MLP)
- **Library**: `tensorflow.keras`
- **Implementation**:
  - **Architecture**: Sequential model with input, hidden, and output layers.
  - **Regularization**: L1 and L2 penalties to avoid overfitting.
  - **Early stopping**: Improves generalization.
- **Purpose**: Non-linear regression and classification tasks using neural networks.

---

## Shared Techniques Across Models

### Model Training and Evaluation
- **Techniques**:
  - Models are trained, validated, and evaluated using performance metrics such as RMSE and F1 score.
  - Cross-validation techniques are applied to assess generalization.

### Hyperparameter Tuning
- **Methods**:
  - Grid Search (`GridSearchCV`) and Randomized Search (`RandomizedSearchCV`) for optimizing model parameters.
