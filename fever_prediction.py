import pandas as pd

results = {
    'Model': [
        'aveOralF_Regression',
        'aveOralM_Regression',
        'aveOralF_Classification',
        'aveOralM_Classification'
    ],
    'Search Method': [
        'Grid Search: XGBoost',
        'Grid Search: XGBoost',
        'Random Forest Classifier',
        'Random Forest Classifier'
    ],
    'Metric': [
        'RMSE',
        'RMSE',
        'F1',
        'F1'
    ],
    'Value on Test set': [
        0.2233,
        0.2246,
        0.8333,
        0.8260
    ]
}

# Convert dictionary to DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame as a table
print(results_df)



"""# Data Acquisition"""

!pip install ucimlrepo

"""#### We can find this dataset contains 2 different targets that are continuous data type and 33 features with 3 cateorical types and 30 continuous types, which mean we may have to perform one-hot encoding or label encoding."""

from ucimlrepo import fetch_ucirepo

# fetch dataset
infrared_thermography_temperature = fetch_ucirepo(id=925)

# data (as pandas dataframes)
X = infrared_thermography_temperature.data.features
y = infrared_thermography_temperature.data.targets

# metadata
print(infrared_thermography_temperature.metadata)

# variable information
print(infrared_thermography_temperature.variables)

"""# Import necessary library"""

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥1 is required
import sklearn
assert sklearn.__version__ >= "1.0"

# Common imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Train Model
import statsmodels.api as sm
import xgboost as xgb
from scipy.stats import randint, uniform
from seaborn import regplot
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, precision_recall_curve, roc_curve, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from xgboost import XGBRegressor

"""## The gender, age and ethnicity has beem divide to some sub-categrories, so that mean if we decide to perform with one-hot encoding, there will be a a lot of 0 and 1 distributed in sparse vectors, it could lead an increase in the complexity of the model and also the loading of computing."""

X.head(10).T

y.head(10).T

"""# Data Preprocess
## Therefore, we decide to use label encoding, convert categorical features to numeric types so that the machine can process, further improve the prediction accuracy.
"""

combined_df = pd.concat([X, y], axis=1)

print("first 3 rows:")
print(combined_df.head(3))

print("\nlast 3 rows:")
print(combined_df.tail(3))

combined_df

"""## Surprisingly, seems there are 2 missing value in the "Distance", but the dataset description from the authors claims there's no missing value."""

combined_df.info()

"""## There are 2 missing value indeed. Let's take mre examine about the dataset."""

print(combined_df.isnull().sum())  # check null in features

combined_df.duplicated().sum()

"""## Apparently, we can find seems there's a max number "79.0" in the distance feature. Consider this is a thermometer, it is unreasonable no matter the measurement is millimeter, centimeter, or meter, so this value should be handled.
## Second, the standard deviation of humidity is quite high, but consider the raining or wet weather, so we put it away for now. All other statistical looks normal.
"""

combined_df.describe().T

"""## There are many ways to dealt with missing value and outlier, for example, we can simplely use sklearn imputer to replace missing value, but we decide take a close look and find the lier as well. And we can find the outlier is in row 97 and missing vaalue in row 902 and 903."""

# Set option to display more rows, if necessary
pd.set_option('display.max_rows', None)

# Print the 'Distance' column of the DataFrame
print(combined_df['Distance'])

"""## We decide to use median to replace missing value and outliers since there are only three values."""

# cauculate median
median_value = combined_df['Distance'].median()

# replce outlier with median
combined_df.loc[97, 'Distance'] = median_value

# replce missing value with median
combined_df.loc[902, 'Distance'] = median_value
combined_df.loc[903, 'Distance'] = median_value

# check results
print("outlier value now：", combined_df.loc[97, 'Distance'])
print("missing valaue now：", combined_df.loc[[902, 903], 'Distance'])

# check again
print("check missing value again：", combined_df['Distance'].isnull().sum())

combined_df.describe().T

"""# EDA

## Most of features are normal except humidity. Let's check later.
"""

combined_df.plot(kind='box', subplots=True, layout=(6, 6), figsize=(20, 15))
plt.show()

"""## Actually, the outlier of humidity seems not single value, they are clutch, proving our opinion is acceptale, meaning it could be raining day or wet weather, so we won't process it."""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
combined_df.hist(bins=50, figsize=(20,15))
plt.show()

"""## As we mentioned before, we convert Gender, Age, and Ethnicity to lable."""

# Create a copy of combined_df to avoid changing the original DataFrame
combined_df_encoded = combined_df.copy()

# Initialize label encoders
label_encoders = {}

# For each categorical column, fit a label encoder, transform the data, and store the encoder
for column in ['Gender', 'Ethnicity', 'Age']:
    label_encoders[column] = LabelEncoder()
    combined_df_encoded[column] = label_encoders[column].fit_transform(combined_df_encoded[column])

# Combined_df_encoded contains the label encoded categorical features
print(combined_df_encoded.head())

"""## Now the data type is int64, numerical type."""

combined_df_encoded.info()

"""## Since our job is to predict if people is fever or not, we should set our target for regression and classification tasks. Temperature over than 37.5 is be regarded as fever in common awareness. So we set our target append a new column into our dataframe."""

# Add binary fever index to combined_df_encoded
combined_df_encoded['Fever_aveOralF'] = (combined_df_encoded['aveOralF'] > 37.5).astype(int)
combined_df_encoded['Fever_aveOralM'] = (combined_df_encoded['aveOralM'] > 37.5).astype(int)

"""## We can see 0 is normal, 1 is fever."""

combined_df_encoded['Fever_aveOralF']

"""## Further examine the balance of dataset. Almost 95% data point is normal, so this is an imbalanced dataset in our targets, aveOralM and aveOralF. We should perform stratified sampling for our classification tasks."""

# piechart based on 'aveOralF'
plt.figure(figsize=(8, 11))
plt.subplot(2, 1, 1)
combined_df_encoded['Fever_aveOralF'].value_counts().plot.pie(
    labels=['Normal', 'Fever'],
    autopct='%1.1f%%',
    startangle=140,
    colors=['Purple', 'Cyan'],
    legend=True
)
plt.title('Fever based on aveOralF')

# piechart based on 'aveOralM'
plt.subplot(2, 1, 2)
combined_df_encoded['Fever_aveOralM'].value_counts().plot.pie(
    labels=['Normal', 'Fever'],
    autopct='%1.1f%%',
    startangle=140,
    colors=['Purple', 'Cyan'],
    legend=True
)
plt.title('Fever based on aveOralM')

plt.tight_layout()
plt.show()

"""## We examine the heatmap map and try to select some obvious feature for our training. Unfortunatly, there are too many variables so that it is hard to identify by our eyes."""

# Compute the correlation matrix for the combined DataFrame with encoded categorical variables
corr_matrix_encoded = combined_df_encoded.corr()

# heatmap for the new correlation matrix
plt.figure(figsize=(40, 20))
sns.heatmap(corr_matrix_encoded, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap with Encoded Categorical Variables')
plt.show()

"""# Feature Selection
## Therefore, we perform feature selection for certain models, especailly for MLP (<MultiLayer Perceptron). We further drop our target from dataframe because it could affect our model performance by too high-correlations, and it could lead to overfit as well. So we set our target aveOralF for one of four tasks.
## We can see the selected features is 'Ethnicity', 'T_atm', 'T_offset1', 'Max1R13_1', 'aveAllL13_1', 'T_RC1', 'T_FHCC1', 'T_FHRC1', 'T_FHBC1', 'T_FH_Max1', 'T_Max1', 'T_OR1'].
"""

# Copy the preprocessed DataFrame for separate analyses
reg_f = combined_df_encoded.copy()
reg_m = combined_df_encoded.copy()
class_f = combined_df_encoded.copy()
class_m = combined_df_encoded.copy()

# Separate the features from the target
X = reg_f.drop(['aveOralF', 'aveOralM', 'Fever_aveOralF', 'Fever_aveOralM'], axis=1)
y = reg_f['aveOralF']  # Assuming 'aveOralF' is the continuous target for regression

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the full model with all features
model = sm.OLS(y, X).fit()

# Backward elimination process
p_value_threshold = 0.05
variables = X.columns.tolist()
while len(variables) > 0:
    # Fit the model and get the p-values
    model = sm.OLS(y, X[variables]).fit()
    p_values = model.pvalues

    # Get the max p-value and its corresponding variable
    max_p_value_var = p_values.idxmax()
    max_p_value = p_values.max()

    # Check if the max p-value is greater than the threshold
    if max_p_value > p_value_threshold:
        # Remove the variable with the highest p-value
        variables.remove(max_p_value_var)
    else:
        # Exit the loop if no variable has p-value above the threshold
        break

# Final variables are now selected
final_features = variables
print('Selected features for aveOralF Regression:', final_features)

"""## We exclude the columns should be exclude, including our target related features, including binary indicators and label-encoded categorical columns. Then assign the specific columns need to be standarized. Next, standardized the dataframe. Lastly, all related columns has been standarized."""

# List of columns to exclude from standardization, including binary indicators and label-encoded categorical columns
columns_to_exclude = ['aveOralF', 'aveOralM', 'Fever_aveOralF', 'Fever_aveOralM', 'Age', 'Gender', 'Ethnicity']

# Selecting the columns that need to be standardized, excluding the label-encoded categorical columns and binary indicators
columns_to_scale = [col for col in combined_df_encoded.columns if col not in columns_to_exclude]

# Instantiating the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the data for the specified columns
combined_df_encoded[columns_to_scale] = scaler.fit_transform(combined_df_encoded[columns_to_scale])

# Combined_df_encoded' has the standardized columns except the ones we excluded
print(combined_df_encoded.head())

"""# aveOralF_Regression

# Random Sampling for Regression Model
## We use random sampling for regression task. Split 80% for traning and 20% for test. Of course, drop the feature sgold not included, such as our target, and assign aveOralF as our target in y_train.
"""

# Separating the target variable 'aveOralF' and dropping it from the features set
X = combined_df_encoded.drop(['aveOralF', 'Fever_aveOralF', 'aveOralM', 'Fever_aveOralM'], axis=1)
y = combined_df_encoded['aveOralF']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,  # Features
    y,  # Target
    test_size=0.2,
    random_state=42
)

print(X_train.head())
print(y_train.head())

"""# Liner Regression
## The residuals are near the 0 line, which means the model's prediction have negative an positive deviations. Residual doesn't increase or decrease with the predicted values, indicating that the model's variance is fixed stable. There are some outliers, meaning the actual and predication exsist errors.
"""

# Instantiate the Linear Regression model
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

# Predictions on the training and testing datasets
y_train_pred = linear_model.predict(X_train)

# Evaluate the model with Root Mean Squared Error (RMSE)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("Training RMSE:", train_rmse)

# Calculate residuals
residuals = y_train - y_train_pred

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, residuals, alpha=0.5)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.axhline(y=0, color='k', linestyle='--')  # Zero line for reference
plt.show()

# Comparing actual and predicted values
comparison_df = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
print(comparison_df.head())

"""## RMSE is 0.247, and the standard deviation is 0.033, which means the model performs well, indicating that the model's performance is stable at different cross-validation folds."""

# Instantiate the Linear Regression model (again for clarity)
linear_model_cv = LinearRegression()

# Perform cross-validation
cv_scores = cross_val_score(linear_model_cv, X, y, n_jobs=-1, cv=10, scoring='neg_mean_squared_error')

# Calculate RMSE from cross-validation scores
cv_rmse = np.sqrt(-cv_scores)

print("Cross-validation RMSE scores:", cv_rmse)
print("Mean RMSE:", cv_rmse.mean())
print("Standard deviation of RMSE:", cv_rmse.std())

"""# SGD Linear Regression"""

# SGD is sensitive to feature scaling, therefore scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

sgd_regressor.fit(X_train_scaled, y_train)

y_train_pred_sgd = sgd_regressor.predict(X_train_scaled)

train_rmse_sgd = np.sqrt(mean_squared_error(y_train, y_train_pred_sgd))

print("Training RMSE (SGD):", train_rmse_sgd)

# Comparing actual and predicted values using SGDRegressor predictions
comparison_df_sgd = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred_sgd})
print(comparison_df_sgd.head())

"""## As same result as Liner Regression, it could be the reason that we use the same parameters in CV or the feature correlation is relative low."""

# Instantiate the Linear Regression model
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

cv_scores = cross_val_score(linear_model_cv, X, y, n_jobs=-1, cv=10, scoring='neg_mean_squared_error')

cv_rmse = np.sqrt(-cv_scores)

print("Cross-validation RMSE scores:", cv_rmse)
print("Mean RMSE:", cv_rmse.mean())
print("Standard deviation of RMSE:", cv_rmse.std())

"""# Polynomial Regression"""

features_select_fR = X_train[['Ethnicity', 'T_atm', 'T_offset1', 'Max1R13_1', 'aveAllL13_1', 'T_RC1', 'T_FHCC1', 'T_FHRC1', 'T_FHBC1', 'T_FH_Max1', 'T_Max1', 'T_OR1']]

features_select_fR

selected_features = ['Ethnicity', 'T_atm', 'T_offset1', 'Max1R13_1', 'aveAllL13_1', 'T_RC1', 'T_FHCC1', 'T_FHRC1', 'T_FHBC1', 'T_FH_Max1', 'T_Max1', 'T_OR1']

# Apply the feature selection to X_train and X_test
X_train_selected = X_train[selected_features]

# Initialize the PolynomialFeatures transformer and Linear Regression model within a pipeline
poly_regression_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

# Fit the pipeline to the selected features from the training data
poly_regression_pipeline.fit(X_train_selected, y_train)

y_train_pred = poly_regression_pipeline.predict(X_train_selected)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Training RMSE:", train_rmse)

comparison_df_poly = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})

print(comparison_df_poly.head())

cv_scores = cross_val_score(poly_regression_pipeline, X_train_selected, y_train, n_jobs=-1, cv=10, scoring='neg_mean_squared_error')

# Convert scores to positive as 'cross_val_score' returns negative MSE scores for maximization
cv_scores_positive = -cv_scores

cv_rmse_scores = np.sqrt(cv_scores_positive)

print("Cross-validation RMSE scores:", cv_rmse_scores)
print("Mean RMSE:", cv_rmse_scores.mean())
print("Standard Deviation of RMSE:", cv_rmse_scores.std())

"""# Decision Tree"""

dtr_regressor = DecisionTreeRegressor(random_state=42)

dtr_regressor.fit(X_train, y_train)

y_train_pred = dtr_regressor.predict(X_train)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("Training RMSE:", train_rmse)

# Tree's quantity
print("Decision Tree depth:", dtr_regressor.get_depth())
print("Number of leaves:", dtr_regressor.get_n_leaves())

dtr_regressor2 = DecisionTreeRegressor(max_depth=5, random_state=42)
dtr_regressor2.fit(X_train, y_train)
y_train_pred = dtr_regressor2.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Training RMSE:", train_rmse)
print("Decision Tree depth:", dtr_regressor2.get_depth())
print("Number of leaves:", dtr_regressor2.get_n_leaves())

cv_scores = cross_val_score(poly_regression_pipeline, X_train_selected, y_train, n_jobs=-1, cv=10, scoring='neg_mean_squared_error')

cv_scores_positive = -cv_scores

cv_rmse_scores = np.sqrt(cv_scores_positive)

print("Cross-validation RMSE scores:", cv_rmse_scores)
print("Mean RMSE:", cv_rmse_scores.mean())
print("Standard Deviation of RMSE:", cv_rmse_scores.std())

"""# KNN"""

knn_regressor = KNeighborsRegressor(n_neighbors=1)
knn_regressor.fit(X_train, y_train)

# overfitting because mse is 0
F_prediction_KNN = knn_regressor.predict(X_train)
F_KNN_mse = mean_squared_error(y_train, F_prediction_KNN)
F_KNN_rmse = np.sqrt(F_KNN_mse)
F_KNN_rmse

knn_regressor2 = KNeighborsRegressor(n_neighbors=14)
knn_regressor2.fit(X_train, y_train)
F_prediction_KNN2 = knn_regressor2.predict(X_train)
F_KNN_mse2 = mean_squared_error(y_train, F_prediction_KNN2)
F_KNN_rmse2 = np.sqrt(F_KNN_mse2)
F_KNN_rmse2

# KNeighborsRegressor, n_neighbors=1
knn_regressor = KNeighborsRegressor(n_neighbors=1)

cv_scores_knn = cross_val_score(knn_regressor, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

cv_rmse_scores_knn = np.sqrt(-cv_scores_knn)

print("Cross-validation RMSE scores for KNN (n_neighbors=1):", np.round(cv_rmse_scores_knn, 2))
print("Mean RMSE:", round(cv_rmse_scores_knn.mean(), 2))
print("Standard deviation of RMSE:", round(cv_rmse_scores_knn.std(), 2))

# KNeighborsRegressor, n_neighbors=14
knn_regressor2 = KNeighborsRegressor(n_neighbors=14)

# 10-fold
cv_scores_knn2 = cross_val_score(knn_regressor2, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

cv_rmse_scores_knn2 = np.sqrt(-cv_scores_knn2)

print("\nCross-validation RMSE scores for KNN (n_neighbors=14):", np.round(cv_rmse_scores_knn2, 2))
print("Mean RMSE:", round(cv_rmse_scores_knn2.mean(), 2))
print("Standard deviation of RMSE:", round(cv_rmse_scores_knn2.std(), 2))

"""# Random Forest"""

F_RF_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
F_RF_regressor.fit(X_train, y_train)

F_RF_pred = F_RF_regressor.predict(X_train)
F_RF_mse = mean_squared_error(y_train, F_RF_pred)
F_RF_rmse = np.sqrt(F_RF_mse)
F_RF_rmse

F_RF_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

cv_scores = cross_val_score(F_RF_regressor, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

cv_rmse_scores = np.sqrt(-cv_scores)

print("Cross-validation RMSE scores:", np.round(cv_rmse_scores, 2))
print("Mean RMSE:", round(cv_rmse_scores.mean(), 2))
print("Standard deviation of RMSE:", round(cv_rmse_scores.std(), 2))

"""# XGBoost"""

!pip install xgboost

xgb_regressor = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_regressor.fit(X_train, y_train)

xgb_pred = xgb_regressor.predict(X_train)
xgb_mse = mean_squared_error(y_train, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_rmse

xgb_regressor2 = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
xgb_regressor2.fit(X_train, y_train)
xgb_predictions2 = xgb_regressor2.predict(X_train)
xgb_mse2 = mean_squared_error(y_train, xgb_predictions2)
xgb_rmse2 = np.sqrt(xgb_mse2)
xgb_rmse2

# XGBoost Regressor Model 1
xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42)

cv_scores_xgb = cross_val_score(xgb_reg, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

cv_rmse_scores_xgb = np.sqrt(-cv_scores_xgb)

print("Cross-validation RMSE scores for XGBoost Model 1:", np.round(cv_rmse_scores_xgb, 2))
print("Mean RMSE for XGBoost Model 1:", round(cv_rmse_scores_xgb.mean(), 2))
print("Standard deviation of RMSE for XGBoost Model 1:", round(cv_rmse_scores_xgb.std(), 2))

# XGBoost Regressor Model 2 (with max_depth=3)
xgb_reg2 = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)

cv_scores_xgb2 = cross_val_score(xgb_reg2, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

cv_rmse_scores_xgb2 = np.sqrt(-cv_scores_xgb2)

# Print cross-validation results for XGBoost Model 2
print("\nCross-validation RMSE scores for XGBoost Model 2:", np.round(cv_rmse_scores_xgb2, 2))
print("Mean RMSE for XGBoost Model 2:", round(cv_rmse_scores_xgb2.mean(), 2))
print("Standard deviation of RMSE for XGBoost Model 2:", round(cv_rmse_scores_xgb2.std(), 2))

"""# MLP
## Our MLP + Selected Features with Early Stopping and L1+L2 performance is weaker than the origianl MLP, which is quite surprised. The reason could be too strong regularization interactetion. L1 and L2 are used for prevent overfitting. L1 regularization (sparsity) makes the model learn weights that are actually zero, so it enhanced feature engineering. L2 regularization makes the model learn more normal distributed weights.

## MLP
"""

# Scale features for neural network models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model architecture
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))  # Input layer
model.add(Dense(32, activation='relu'))  # Hidden layer
model.add(Dense(1))  # Output layer

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Fit the model
history = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on the training data
train_mse = model.evaluate(X_train_scaled, y_train, verbose=0)
print(f"Training Mean Squared Error: {train_mse}")

# Predictions on train dataset
y_train_pred = model.predict(X_train_scaled)

model.summary()

# Convert the history.history dict to a pandas DataFrame and plot the loss
history_df = pd.DataFrame(history.history)
history_df.plot(figsize=(8, 5))
plt.grid(True)
# Adjust the y-axis limits to the range of loss values for regression task
plt.gca().set_ylim(0, history_df['loss'].max())  # Use the maximum value of training loss to set the upper limit
plt.title("Model Loss During Training")
plt.ylabel("Loss (Mean Squared Error)")
plt.xlabel("Epoch")
# plt.savefig("keras_learning_curves_plot.png")
plt.show()

"""## MLP + Selected Features"""

# List of selected features
selected_features = [
    'Ethnicity', 'T_atm', 'T_offset1', 'Max1R13_1', 'aveAllL13_1',
    'T_RC1', 'T_FHCC1', 'T_FHRC1', 'T_FHBC1', 'T_FH_Max1', 'T_Max1', 'T_OR1'
]

# Select the specified features from the training data
X_train_selected = X_train[selected_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)

# Model structure
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=1)

train_mse = model.evaluate(X_train_scaled, y_train, verbose=0)
print(f"Training Mean Squared Error: {train_mse}")

y_train_pred = model.predict(X_train_scaled)

history_df = pd.DataFrame(history.history)
history_df.plot(figsize=(8, 5))
plt.grid(True)

plt.gca().set_ylim(0, history_df['loss'].max())
plt.title("Model Loss During Training")
plt.ylabel("Loss (Mean Squared Error)")
plt.xlabel("Epoch")

plt.show()

"""## MLP + Selected Features with Early Stopping and L1+L2"""

selected_features = [
    'Ethnicity', 'T_atm', 'T_offset1', 'Max1R13_1', 'aveAllL13_1',
    'T_RC1', 'T_FHCC1', 'T_FHRC1', 'T_FHBC1', 'T_FH_Max1', 'T_Max1', 'T_OR1'
]

X_train_selected = X_train[selected_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)

# Model structure with L1 and L2 regularization
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu',
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # L1 and L2 penalties on the layer's weights
model.add(Dense(32, activation='relu',
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Initialize early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model with early stopping
history = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])

# Evaluate the model on the training data
train_mse = model.evaluate(X_train_scaled, y_train, verbose=0)
print(f"Training Mean Squared Error: {train_mse}")

# Predictions
y_train_pred = model.predict(X_train_scaled)

model.summary()

"""# Best Model Evaluation"""

data = {
    "Fast Mode Model": [
        "Linear Regression", "SGD Linear Regression", "Polynomial Regression",
        "Decision Tree", "KNN", "Random Forest",
        "XGBoost", "MLP", "MLP + Selected Features", "MLP + Selected Features with Early Stopping and L1+L2"
    ],
    "Fast Mode RMSE": [
        0.200, 0.214, 0.204, 0.166, 0.235, 0.081,
        0.075, 1.063, 0.368, 4.418
    ]
}

df = pd.DataFrame(data)

df_sorted = df.sort_values("Fast Mode RMSE")

df_sorted

"""# Grid Search: XGBoost (best model)"""

# Setup the parameters from param_grid
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.005, 0.01, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1]
}

xgb = XGBRegressor(random_state=42)

# Instantiate the GridSearchCV object as grid_search
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=1,
                           n_jobs=-1)

# Perform Grid Search
grid_search.fit(X_train, y_train)

# Retrieve the best parameters and the corresponding score
best_param = grid_search.best_params_
best_score = np.sqrt(-grid_search.best_score_)

# Prediction on the test set with the best model and calculate RMSE
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("Best parameters found: ", best_param)
print("Lowest RMSE found: ", best_score)
print("RMSE on training set: ", rmse)

grid_search.best_estimator_

"""# Random Search: XGBoost"""

# # Setup the parameters from distribs
param_distribs = {
    'max_depth': randint(low=3, high=10),
    'learning_rate': uniform(0.005, 0.015),
    'n_estimators': randint(100, 500),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.4, 0.6)
}

# XGBRegressor model
xgb = XGBRegressor(random_state=42)

# RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_distribs,
                                   n_iter=10, cv=5, scoring='neg_mean_squared_error',
                                   verbose=1, n_jobs=-1, random_state=42)

# Perform Randomized Search
random_search.fit(X_train, y_train)

# Retrieve the best parameters and the corresponding score
best_param = random_search.best_params_
best_score = np.sqrt(-random_search.best_score_)

# Prediction on the test set with the best model and calculate RMSE
best_model = random_search.best_estimator_
y_train_pred = best_model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("Best parameters found: ", best_param)
print("Lowest RMSE found: ", best_score)
print("RMSE on test set: ", rmse)

random_search.best_estimator_

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

print(X_train.columns)

# the best predictor and the important feature
best_estimator = grid_search.best_estimator_

feature_importance = best_estimator.feature_importances_
features = X_train.columns
importance = zip(features, feature_importance)
sorted_importance = sorted(importance, key=lambda x: x[1], reverse=True)

for feature, importance in sorted_importance:
    print(f"{feature}: {importance}")

"""# Run on test set"""

# Evaluate the best model on the test set by Grid Search model

final_best_model = grid_search.best_estimator_

# Predict on test set
test_predictions = final_best_model.predict(X_test)

# RMSE
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)

print(test_rmse)

"""# aveOralM_Regression

# Feature Selection
"""

reg_f = combined_df_encoded.copy()
reg_m = combined_df_encoded.copy()
class_f = combined_df_encoded.copy()
class_m = combined_df_encoded.copy()

X = reg_f.drop(['aveOralF', 'aveOralM', 'Fever_aveOralF', 'Fever_aveOralM'], axis=1)
y = reg_f['aveOralM']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

p_value_threshold = 0.05
variables = X.columns.tolist()
while len(variables) > 0:
        model = sm.OLS(y, X[variables]).fit()
    p_values = model.pvalues
    max_p_value_var = p_values.idxmax()
    max_p_value = p_values.max()

    if max_p_value > p_value_threshold:
        variables.remove(max_p_value_var)

    else:
        break

final_features = variables
print('Selected features for aveOralF Regression:', final_features)

from sklearn.preprocessing import StandardScaler

# List of columns to exclude from standardization, including binary indicators and label-encoded categorical columns
columns_to_exclude = ['aveOralF', 'aveOralM', 'Fever_aveOralF', 'Fever_aveOralM', 'Age', 'Gender', 'Ethnicity']

# Selecting the columns that need to be standardized, excluding the label-encoded categorical columns and binary indicators
columns_to_scale = [col for col in combined_df_encoded.columns if col not in columns_to_exclude]

# Instantiating the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the data for the specified columns
combined_df_encoded[columns_to_scale] = scaler.fit_transform(combined_df_encoded[columns_to_scale])

# Now 'combined_df_encoded' has the standardized columns except the ones we excluded
print(combined_df_encoded.head())  # Display the first few rows to verify

"""# Random Sampling for Regression Model"""

X = combined_df_encoded.drop(['aveOralM', 'Fever_aveOralM', 'aveOralF', 'Fever_aveOralF'], axis=1)
y = combined_df_encoded['aveOralM']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.head())
print(y_train.head())

"""# Liner Regression"""

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_train_pred = linear_model.predict(X_train)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("Training RMSE:", train_rmse)

residuals = y_train - y_train_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, residuals, alpha=0.5)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.axhline(y=0, color='k', linestyle='--')
plt.show()

comparison_df = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
print(comparison_df.head())

linear_model_cv = LinearRegression()

cv_scores = cross_val_score(linear_model_cv, X, y, n_jobs=-1, cv=10, scoring='neg_mean_squared_error')

cv_rmse = np.sqrt(-cv_scores)

print("Cross-validation RMSE scores:", cv_rmse)
print("Mean RMSE:", cv_rmse.mean())
print("Standard deviation of RMSE:", cv_rmse.std())

"""# SGD Linear Regression"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

sgd_regressor.fit(X_train_scaled, y_train)

y_train_pred_sgd = sgd_regressor.predict(X_train_scaled)

train_rmse_sgd = np.sqrt(mean_squared_error(y_train, y_train_pred_sgd))

print("Training RMSE (SGD):", train_rmse_sgd)

comparison_df_sgd = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred_sgd})
print(comparison_df_sgd.head())

sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

cv_scores = cross_val_score(linear_model_cv, X, y, n_jobs=-1, cv=10, scoring='neg_mean_squared_error')

cv_rmse = np.sqrt(-cv_scores)

print("Cross-validation RMSE scores:", cv_rmse)
print("Mean RMSE:", cv_rmse.mean())
print("Standard deviation of RMSE:", cv_rmse.std())

"""# Polynomial Regression"""

features_select_fR = X_train[['Gender', 'Ethnicity', 'T_atm', 'T_offset1', 'Max1R13_1', 'aveAllL13_1', 'T_RC_Max1', 'LCC1', 'T_FHCC1', 'T_FHLC1', 'T_FHBC1', 'T_FH_Max1', 'T_Max1', 'T_OR1']]

features_select_fR

selected_features = ['Gender', 'Ethnicity', 'T_atm', 'T_offset1', 'Max1R13_1', 'aveAllL13_1', 'T_RC_Max1', 'LCC1', 'T_FHCC1', 'T_FHLC1', 'T_FHBC1', 'T_FH_Max1', 'T_Max1', 'T_OR1']

X_train_selected = X_train[selected_features]

poly_regression_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

poly_regression_pipeline.fit(X_train_selected, y_train)

y_train_pred = poly_regression_pipeline.predict(X_train_selected)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Training RMSE:", train_rmse)

comparison_df_poly = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})

print(comparison_df_poly.head())

cv_scores = cross_val_score(poly_regression_pipeline, X_train_selected, y_train, n_jobs=-1, cv=10, scoring='neg_mean_squared_error')

cv_scores_positive = -cv_scores

cv_rmse_scores = np.sqrt(cv_scores_positive)

print("Cross-validation RMSE scores:", cv_rmse_scores)
print("Mean RMSE:", cv_rmse_scores.mean())
print("Standard Deviation of RMSE:", cv_rmse_scores.std())

"""# Decision Tree"""

dtr_regressor = DecisionTreeRegressor(random_state=42)

dtr_regressor.fit(X_train, y_train)

y_train_pred = dtr_regressor.predict(X_train)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("Training RMSE:", train_rmse)

print("Decision Tree depth:", dtr_regressor.get_depth())
print("Number of leaves:", dtr_regressor.get_n_leaves())

dtr_regressor2 = DecisionTreeRegressor(max_depth=5, random_state=42)
dtr_regressor2.fit(X_train, y_train)
y_train_pred = dtr_regressor2.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Training RMSE:", train_rmse)
print("Decision Tree depth:", dtr_regressor2.get_depth())
print("Number of leaves:", dtr_regressor2.get_n_leaves())

cv_scores = cross_val_score(poly_regression_pipeline, X_train_selected, y_train, n_jobs=-1, cv=10, scoring='neg_mean_squared_error')

cv_scores_positive = -cv_scores

cv_rmse_scores = np.sqrt(cv_scores_positive)

print("Cross-validation RMSE scores:", cv_rmse_scores)
print("Mean RMSE:", cv_rmse_scores.mean())
print("Standard Deviation of RMSE:", cv_rmse_scores.std())

"""# KNN"""

knn_regressor = KNeighborsRegressor(n_neighbors=1)
knn_regressor.fit(X_train, y_train)

F_prediction_KNN = knn_regressor.predict(X_train)
F_KNN_mse = mean_squared_error(y_train, F_prediction_KNN)
F_KNN_rmse = np.sqrt(F_KNN_mse)
F_KNN_rmse

knn_regressor2 = KNeighborsRegressor(n_neighbors=14)
knn_regressor2.fit(X_train, y_train)
F_prediction_KNN2 = knn_regressor2.predict(X_train)
F_KNN_mse2 = mean_squared_error(y_train, F_prediction_KNN2)
F_KNN_rmse2 = np.sqrt(F_KNN_mse2)
F_KNN_rmse2

knn_regressor = KNeighborsRegressor(n_neighbors=1)

cv_scores_knn = cross_val_score(knn_regressor, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

cv_rmse_scores_knn = np.sqrt(-cv_scores_knn)

print("Cross-validation RMSE scores for KNN (n_neighbors=1):", np.round(cv_rmse_scores_knn, 2))
print("Mean RMSE:", round(cv_rmse_scores_knn.mean(), 2))
print("Standard deviation of RMSE:", round(cv_rmse_scores_knn.std(), 2))

knn_regressor2 = KNeighborsRegressor(n_neighbors=14)

cv_scores_knn2 = cross_val_score(knn_regressor2, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

cv_rmse_scores_knn2 = np.sqrt(-cv_scores_knn2)

print("\nCross-validation RMSE scores for KNN (n_neighbors=14):", np.round(cv_rmse_scores_knn2, 2))
print("Mean RMSE:", round(cv_rmse_scores_knn2.mean(), 2))
print("Standard deviation of RMSE:", round(cv_rmse_scores_knn2.std(), 2))

"""# Random Forest"""

F_RF_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
F_RF_regressor.fit(X_train, y_train)

F_RF_pred = F_RF_regressor.predict(X_train)
F_RF_mse = mean_squared_error(y_train, F_RF_pred)
F_RF_rmse = np.sqrt(F_RF_mse)
F_RF_rmse

F_RF_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

cv_scores = cross_val_score(F_RF_regressor, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

cv_rmse_scores = np.sqrt(-cv_scores)

print("Cross-validation RMSE scores:", np.round(cv_rmse_scores, 2))
print("Mean RMSE:", round(cv_rmse_scores.mean(), 2))
print("Standard deviation of RMSE:", round(cv_rmse_scores.std(), 2))

"""# XGBoost"""

xgb_regressor = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_regressor.fit(X_train, y_train)

xgb_pred = xgb_regressor.predict(X_train)
xgb_mse = mean_squared_error(y_train, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_rmse

xgb_regressor2 = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
xgb_regressor2.fit(X_train, y_train)
xgb_predictions2 = xgb_regressor2.predict(X_train)
xgb_mse2 = mean_squared_error(y_train, xgb_predictions2)
xgb_rmse2 = np.sqrt(xgb_mse2)
xgb_rmse2

xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42)

cv_scores_xgb = cross_val_score(xgb_reg, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

cv_rmse_scores_xgb = np.sqrt(-cv_scores_xgb)

print("Cross-validation RMSE scores for XGBoost Model 1:", np.round(cv_rmse_scores_xgb, 2))
print("Mean RMSE for XGBoost Model 1:", round(cv_rmse_scores_xgb.mean(), 2))
print("Standard deviation of RMSE for XGBoost Model 1:", round(cv_rmse_scores_xgb.std(), 2))

xgb_reg2 = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)

cv_scores_xgb2 = cross_val_score(xgb_reg2, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

cv_rmse_scores_xgb2 = np.sqrt(-cv_scores_xgb2)

print("\nCross-validation RMSE scores for XGBoost Model 2:", np.round(cv_rmse_scores_xgb2, 2))
print("Mean RMSE for XGBoost Model 2:", round(cv_rmse_scores_xgb2.mean(), 2))
print("Standard deviation of RMSE for XGBoost Model 2:", round(cv_rmse_scores_xgb2.std(), 2))

"""# MLP

## MLP
"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

istory = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=1)

train_mse = model.evaluate(X_train_scaled, y_train, verbose=0)
print(f"Training Mean Squared Error: {train_mse}")

y_train_pred = model.predict(X_train_scaled)

model.summary()

history_df = pd.DataFrame(history.history)
history_df.plot(figsize=(8, 5))
plt.grid(True)

plt.gca().set_ylim(0, history_df['loss'].max())
plt.title("Model Loss During Training")
plt.ylabel("Loss (Mean Squared Error)")
plt.xlabel("Epoch")

plt.show()

"""## MLP + Selected Features"""

selected_features = ['Gender', 'Ethnicity', 'T_atm', 'T_offset1', 'Max1R13_1',
                     'aveAllL13_1', 'T_RC_Max1', 'LCC1', 'T_FHCC1', 'T_FHLC1',
                     'T_FHBC1', 'T_FH_Max1', 'T_Max1', 'T_OR1'
]

X_train_selected = X_train[selected_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)

model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=1)

train_mse = model.evaluate(X_train_scaled, y_train, verbose=0)
print(f"Training Mean Squared Error: {train_mse}")

y_train_pred = model.predict(X_train_scaled)

history_df = pd.DataFrame(history.history)
history_df.plot(figsize=(8, 5))
plt.grid(True)

plt.gca().set_ylim(0, history_df['loss'].max())
plt.title("Model Loss During Training")
plt.ylabel("Loss (Mean Squared Error)")
plt.xlabel("Epoch")

plt.show()

"""## MLP + Selected Features with Early Stopping and L1+L2"""

selected_features = [
    'Ethnicity', 'T_atm', 'T_offset1', 'Max1R13_1', 'aveAllL13_1',
    'T_RC1', 'T_FHCC1', 'T_FHRC1', 'T_FHBC1', 'T_FH_Max1', 'T_Max1', 'T_OR1'
]

X_train_selected = X_train[selected_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)

model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu',
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(32, activation='relu',
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])

train_mse = model.evaluate(X_train_scaled, y_train, verbose=0)
print(f"Training Mean Squared Error: {train_mse}")

y_train_pred = model.predict(X_train_scaled)

model.summary()

"""# Best Model Evaluation"""

data = {
    "Monitor Mode Model": [
        "Linear Regression", "SGD Linear Regression", "Polynomial Regression",
        "Decision Tree", "KNN", "Random Forest",
        "XGBoost", "MLP", "MLP + Selected Features", "MLP + Selected Features with Early Stopping and L1+L2"
    ],
    "Monitor Mode RMSE": [
        0.226, 0.239, 0.220, 0.199, 0.254, 0.090,
        0.003, 1.007, 0.335, 4.088
    ]
}

df = pd.DataFrame(data)

df_sorted = df.sort_values("Monitor Mode RMSE")

df_sorted



"""# Grid Search: XGBoost (best model)"""

param_grid = {
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.005, 0.01, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1]
}

xgb = XGBRegressor(random_state=42)

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=1,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

best_param = grid_search.best_params_
best_score = np.sqrt(-grid_search.best_score_)

best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("Best parameters found: ", best_param)
print("Lowest RMSE found: ", best_score)
print("RMSE on training set: ", rmse)

grid_search.best_estimator_

"""# Random Search: XGBoost"""

param_distribs = {
    'max_depth': randint(low=3, high=10),
    'learning_rate': uniform(0.005, 0.015),
    'n_estimators': randint(100, 500),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.4, 0.6)
}

xgb = XGBRegressor(random_state=42)

random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_distribs,
                                   n_iter=10, cv=5, scoring='neg_mean_squared_error',
                                   verbose=1, n_jobs=-1, random_state=42)

random_search.fit(X_train, y_train)

best_param = random_search.best_params_
best_score = np.sqrt(-random_search.best_score_)

best_model = random_search.best_estimator_
y_train_pred = best_model.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("Best parameters found: ", best_param)
print("Lowest RMSE found: ", best_score)
print("RMSE on test set: ", rmse)

random_search.best_estimator_

"""# Run on test set"""

final_best_model = grid_search.best_estimator_

test_predictions = final_best_model.predict(X_test)

test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)

print(test_rmse)

"""# aveOralF_Classification"""

X = combined_df_encoded.drop(['aveOralF', 'Fever_aveOralF','Fever_aveOralM', 'aveOralM'], axis=1)
y = combined_df_encoded['Fever_aveOralF']

strf_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in strf_split.split(X, combined_df_encoded['Fever_aveOralF']):
    X_train = X.loc[train_index]
    y_train = y[train_index]
    X_test = X.loc[test_index]
    y_test = y[test_index]

"""# Logistic Regression"""

log_reg = LogisticRegression(random_state=42, max_iter=1000)

log_reg.fit(X_train, y_train)

"""The CV results shows that the model performance is conservative on different tranining set, but the ROC threshold and related TP rate and FP rate are very close to the results of a single training set, indicating that the model is good at for balancing precision and recall, which means identifying positive and negative classes.

## We use Dr.Daniel's method to check the threshold for classification.
1. The decision score of the traning regression reflects the degree of confidence that a sample is positive class.
2. Next, we cauculate the precision and recall rate, and gain its thresholds.
3. The code remove the last unit from precision and recall so that the shape of threshoold array could be matched. For example, if the threshold of mius is umlimited, then the model predicts all examples are positive, making the recall is 1.
4. precision_recall_curve adds one extra point(precision and recall) for this extreme situation.
5. Adding np.finfo(float).eps, a little trick to prevent divide-by-zero errors.
6. Lastly, using ROC_curve to all point on ROC curve(FPR and TPR), and then calculates the Euclidean distance from these points to the (0,1) point (the ideally optimize point), then find the point closest to (0,1), then we find the best classification threshold, balancing the TP and FP rate.
"""

# calculate the decision function or predicted probability of the model on the training set
y_scores = log_reg.decision_function(X_train)

# calculate precision, recall and threshold
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

# remove the last unit in precision and recall to fit the thresholds shape
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
precisions, recalls = precisions[:-1], recalls[:-1]

# calculate F1 scores for all thresholds that the denominator is zero
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + np.finfo(float).eps)

# find the index corresponding to the maximum F1 score
best_f1_index = np.argmax(f1_scores)
best_threshold_for_f1 = thresholds[best_f1_index]
best_f1_score = f1_scores[best_f1_index]

print(f"Best F1 Score: {best_f1_score}")
print(f"Best Threshold for F1 Score: {best_threshold_for_f1}")

# use roc_curve to find optimal classification threshold
fpr, tpr, roc_thresholds = roc_curve(y_train, y_scores)

# calculate the Euclidean distance between the TP rate and FP rate for each threshold
distances = np.sqrt((0 - fpr) ** 2 + (1 - tpr) ** 2)

# Find the index of minimum distance
best_threshold_index = np.argmin(distances)
best_threshold_for_roc = roc_thresholds[best_threshold_index]
best_fpr = fpr[best_threshold_index]
best_tpr = tpr[best_threshold_index]

print(f"Best Threshold for ROC: {best_threshold_for_roc}")
print(f"Best FPR for ROC: {best_fpr}")
print(f"Best TPR for ROC: {best_tpr}")

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv = cross_val_predict(log_reg, X, y, cv=cv_strategy, method='decision_function')

precisions_cv, recalls_cv, thresholds_cv = precision_recall_curve(y, y_scores_cv)

f1_scores_cv = 2 * (precisions_cv * recalls_cv) / (precisions_cv + recalls_cv + np.finfo(float).eps)

best_f1_index_cv = np.argmax(f1_scores_cv)
best_threshold_for_f1_cv = thresholds_cv[best_f1_index_cv]
best_f1_score_cv = f1_scores_cv[best_f1_index_cv]

print(f"Cross-validated Best F1 Score: {best_f1_score_cv}")
print(f"Cross-validated Best Threshold for F1 Score: {best_threshold_for_f1_cv}")

fpr_cv, tpr_cv, roc_thresholds_cv = roc_curve(y, y_scores_cv)

distances_cv = np.sqrt((0 - fpr_cv) ** 2 + (1 - tpr_cv) ** 2)

best_threshold_index_cv = np.argmin(distances_cv)
best_threshold_for_roc_cv = roc_thresholds_cv[best_threshold_index_cv]
best_fpr_cv = fpr_cv[best_threshold_index_cv]
best_tpr_cv = tpr_cv[best_threshold_index_cv]

print(f"Cross-validated Best Threshold for ROC: {best_threshold_for_roc_cv}")
print(f"Cross-validated Best FPR for ROC: {best_fpr_cv}")
print(f"Cross-validated Best TPR for ROC: {best_tpr_cv}")

"""# SVM for Classification"""

svm_clf = SVC(random_state=42, probability=True)

svm_clf.fit(X_train, y_train)

# calculate the decision function or predicted probability of the model on the training set
y_scores_svm = svm_clf.decision_function(X_train)

# calculate precision and recall and thresholds
precisions_svm, recalls_svm, thresholds_svm = precision_recall_curve(y_train, y_scores_svm)

# remove the last element in precision and recall to match the shape of thresholds
precisions_svm, recalls_svm = precisions_svm[:-1], recalls_svm[:-1]

# calculate F1 scores for all thresholds and handle cases where the denominator is zero
f1_scores_svm = 2 * (precisions_svm * recalls_svm) / (precisions_svm + recalls_svm + np.finfo(float).eps)

# find the index corresponding to the maximum F1 score
best_f1_index_svm = np.argmax(f1_scores_svm)
best_threshold_for_f1_svm = thresholds_svm[best_f1_index_svm]
best_f1_score_svm = f1_scores_svm[best_f1_index_svm]

print(f"SVM Best F1 Score: {best_f1_score_svm}")
print(f"SVM Best Threshold for F1 Score: {best_threshold_for_f1_svm}")

# use roc_curve to find optimal classification threshold
fpr_svm, tpr_svm, roc_thresholds_svm = roc_curve(y_train, y_scores_svm)

# calculate the distance between the TP rate and FP rate for each threshold
distances_svm = np.sqrt((0 - fpr_svm) ** 2 + (1 - tpr_svm) ** 2)

# find the index of minimum distance
best_threshold_index_svm = np.argmin(distances_svm)
best_threshold_for_roc_svm = roc_thresholds_svm[best_threshold_index_svm]
best_fpr_svm = fpr_svm[best_threshold_index_svm]
best_tpr_svm = tpr_svm[best_threshold_index_svm]

print(f"SVM Best Threshold for ROC: {best_threshold_for_roc_svm}")
print(f"SVM Best FPR for ROC: {best_fpr_svm}")
print(f"SVM Best TPR for ROC: {best_tpr_svm}")

# cross-validation(CV) setting
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# initialized SVM
svm_clf = SVC(random_state=42, probability=True)

# CV prediction score for each sample
y_scores_cv_svm = cross_val_predict(svm_clf, X, y, cv=cv_strategy, method='decision_function')

# calculate CV estimates of precision, recall, and F1 score
precisions_cv_svm, recalls_cv_svm, thresholds_cv_svm = precision_recall_curve(y, y_scores_cv_svm)

# calculate F1 scores for all thresholds and handle the denominator that is zero
f1_scores_cv_svm = 2 * (precisions_cv_svm * recalls_cv_svm) / (precisions_cv_svm + recalls_cv_svm + np.finfo(float).eps)

# find the index corresponding to the maximum F1 score
best_f1_index_cv_svm = np.argmax(f1_scores_cv_svm)
best_threshold_for_f1_cv_svm = thresholds_cv_svm[best_f1_index_cv_svm]
best_f1_score_cv_svm = f1_scores_cv_svm[best_f1_index_cv_svm]

print(f"Cross-validated SVM Best F1 Score: {best_f1_score_cv_svm}")
print(f"Cross-validated SVM Best Threshold for F1 Score: {best_threshold_for_f1_cv_svm}")

# use roc_curve to find optimal classification threshold
fpr_cv_svm, tpr_cv_svm, roc_thresholds_cv_svm = roc_curve(y, y_scores_cv_svm)

# calculate the distance between the TP rate and FP rate for each threshold
distances_cv_svm = np.sqrt((0 - fpr_cv_svm) ** 2 + (1 - tpr_cv_svm) ** 2)

# find the index of minimum distance
best_threshold_index_cv_svm = np.argmin(distances_cv_svm)
best_threshold_for_roc_cv_svm = roc_thresholds_cv_svm[best_threshold_index_cv_svm]
best_fpr_cv_svm = fpr_cv_svm[best_threshold_index_cv_svm]
best_tpr_cv_svm = tpr_cv_svm[best_threshold_index_cv_svm]

print(f"Cross-validated SVM Best Threshold for ROC: {best_threshold_for_roc_cv_svm}")
print(f"Cross-validated SVM Best FPR for ROC: {best_fpr_cv_svm}")
print(f"Cross-validated SVM Best TPR for ROC: {best_tpr_cv_svm}")

"""# Decision Tree Classifier"""

decision_tree_clf = DecisionTreeClassifier(random_state=42)

decision_tree_clf.fit(X_train, y_train)

# use predict_proba to obtain the predicted probability
# because Decision Tree Classifier does not have decision_function
# predict_proba returns the probability of each category,
# we take the probability of the positive category (index 1)
y_scores_dt = decision_tree_clf.predict_proba(X_train)[:, 1]

precisions_dt, recalls_dt, thresholds_dt = precision_recall_curve(y_train, y_scores_dt)

precisions_dt, recalls_dt = precisions_dt[:-1], recalls_dt[:-1]

f1_scores_dt = 2 * (precisions_dt * recalls_dt) / (precisions_dt + recalls_dt + np.finfo(float).eps)

best_f1_index_dt = np.argmax(f1_scores_dt)
best_threshold_for_f1_dt = thresholds_dt[best_f1_index_dt]
best_f1_score_dt = f1_scores_dt[best_f1_index_dt]

print(f"Decision Tree Best F1 Score: {best_f1_score_dt}")
print(f"Decision Tree Best Threshold for F1 Score: {best_threshold_for_f1_dt}")

fpr_dt, tpr_dt, roc_thresholds_dt = roc_curve(y_train, y_scores_dt)

distances_dt = np.sqrt((0 - fpr_dt) ** 2 + (1 - tpr_dt) ** 2)

best_threshold_index_dt = np.argmin(distances_dt)
best_threshold_for_roc_dt = roc_thresholds_dt[best_threshold_index_dt]
best_fpr_dt = fpr_dt[best_threshold_index_dt]
best_tpr_dt = tpr_dt[best_threshold_index_dt]

print(f"Decision Tree Best Threshold for ROC: {best_threshold_for_roc_dt}")
print(f"Decision Tree Best FPR for ROC: {best_fpr_dt}")
print(f"Decision Tree Best TPR for ROC: {best_tpr_dt}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv_dt = cross_val_predict(decision_tree_clf, X, y, cv=cv_strategy, method="predict_proba")[:, 1]

precisions_cv_dt, recalls_cv_dt, thresholds_cv_dt = precision_recall_curve(y, y_scores_cv_dt)

precisions_cv_dt, recalls_cv_dt = precisions_cv_dt[:-1], recalls_cv_dt[:-1]

f1_scores_cv_dt = 2 * (precisions_cv_dt * recalls_cv_dt) / (precisions_cv_dt + recalls_cv_dt + np.finfo(float).eps)

best_f1_index_cv_dt = np.argmax(f1_scores_cv_dt)
best_threshold_for_f1_cv_dt = thresholds_cv_dt[best_f1_index_cv_dt]
best_f1_score_cv_dt = f1_scores_cv_dt[best_f1_index_cv_dt]

print(f"Cross-validated Decision Tree Best F1 Score: {best_f1_score_cv_dt}")
print(f"Cross-validated Decision Tree Best Threshold for F1 Score: {best_threshold_for_f1_cv_dt}")

fpr_cv_dt, tpr_cv_dt, roc_thresholds_cv_dt = roc_curve(y, y_scores_cv_dt)

distances_cv_dt = np.sqrt((0 - fpr_cv_dt) ** 2 + (1 - tpr_cv_dt) ** 2)

best_threshold_index_cv_dt = np.argmin(distances_cv_dt)
best_threshold_for_roc_cv_dt = roc_thresholds_cv_dt[best_threshold_index_cv_dt]
best_fpr_cv_dt = fpr_cv_dt[best_threshold_index_cv_dt]
best_tpr_cv_dt = tpr_cv_dt[best_threshold_index_cv_dt]

print(f"Cross-validated Decision Tree Best Threshold for ROC: {best_threshold_for_roc_cv_dt}")
print(f"Cross-validated Decision Tree Best FPR for ROC: {best_fpr_cv_dt}")
print(f"Cross-validated Decision Tree Best TPR for ROC: {best_tpr_cv_dt}")

"""# Random Forest Classifier"""

random_forest_clf = RandomForestClassifier(random_state=42)

random_forest_clf.fit(X_train, y_train)

# Random Forest Classifier also supports predict_proba
# so use it to obtain the predicted probability
# take the probability of the positive category (index 1)
y_scores_rf = random_forest_clf.predict_proba(X_train)[:, 1]

precisions_rf, recalls_rf, thresholds_rf = precision_recall_curve(y_train, y_scores_rf)

precisions_rf, recalls_rf = precisions_rf[:-1], recalls_rf[:-1]

f1_scores_rf = 2 * (precisions_rf * recalls_rf) / (precisions_rf + recalls_rf + np.finfo(float).eps)

best_f1_index_rf = np.argmax(f1_scores_rf)
best_threshold_for_f1_rf = thresholds_rf[best_f1_index_rf]
best_f1_score_rf = f1_scores_rf[best_f1_index_rf]

print(f"Random Forest Best F1 Score: {best_f1_score_rf}")
print(f"Random Forest Best Threshold for F1 Score: {best_threshold_for_f1_rf}")

fpr_rf, tpr_rf, roc_thresholds_rf = roc_curve(y_train, y_scores_rf)

distances_rf = np.sqrt((0 - fpr_rf) ** 2 + (1 - tpr_rf) ** 2)

best_threshold_index_rf = np.argmin(distances_rf)
best_threshold_for_roc_rf = roc_thresholds_rf[best_threshold_index_rf]
best_fpr_rf = fpr_rf[best_threshold_index_rf]
best_tpr_rf = tpr_rf[best_threshold_index_rf]

print(f"Random Forest Best Threshold for ROC: {best_threshold_for_roc_rf}")
print(f"Random Forest Best FPR for ROC: {best_fpr_rf}")
print(f"Random Forest Best TPR for ROC: {best_tpr_rf}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv_rf = cross_val_predict(random_forest_clf, X, y, cv=cv_strategy, method="predict_proba")[:, 1]

precisions_cv_rf, recalls_cv_rf, thresholds_cv_rf = precision_recall_curve(y, y_scores_cv_rf)

precisions_cv_rf, recalls_cv_rf = precisions_cv_rf[:-1], recalls_cv_rf[:-1]

f1_scores_cv_rf = 2 * (precisions_cv_rf * recalls_cv_rf) / (precisions_cv_rf + recalls_cv_rf + np.finfo(float).eps)

best_f1_index_cv_rf = np.argmax(f1_scores_cv_rf)
best_threshold_for_f1_cv_rf = thresholds_cv_rf[best_f1_index_cv_rf]
best_f1_score_cv_rf = f1_scores_cv_rf[best_f1_index_cv_rf]

print(f"Cross-validated Random Forest Best F1 Score: {best_f1_score_cv_rf}")
print(f"Cross-validated Random Forest Best Threshold for F1 Score: {best_threshold_for_f1_cv_rf}")

fpr_cv_rf, tpr_cv_rf, roc_thresholds_cv_rf = roc_curve(y, y_scores_cv_rf)

distances_cv_rf = np.sqrt((0 - fpr_cv_rf) ** 2 + (1 - tpr_cv_rf) ** 2)

best_threshold_index_cv_rf = np.argmin(distances_cv_rf)
best_threshold_for_roc_cv_rf = roc_thresholds_cv_rf[best_threshold_index_cv_rf]
best_fpr_cv_rf = fpr_cv_rf[best_threshold_index_cv_rf]
best_tpr_cv_rf = tpr_cv_rf[best_threshold_index_cv_rf]

print(f"Cross-validated Random Forest Best Threshold for ROC: {best_threshold_for_roc_cv_rf}")
print(f"Cross-validated Random Forest Best FPR for ROC: {best_fpr_cv_rf}")
print(f"Cross-validated Random Forest Best TPR for ROC: {best_tpr_cv_rf}")

"""# Gradient Boosting Trees"""

gbt_clf = GradientBoostingClassifier(random_state=42)

gbt_clf.fit(X_train, y_train)

# Gradient Boosting Trees support predict_proba
y_scores_gbt = gbt_clf.predict_proba(X_train)[:, 1]

precisions_gbt, recalls_gbt, thresholds_gbt = precision_recall_curve(y_train, y_scores_gbt)

precisions_gbt, recalls_gbt = precisions_gbt[:-1], recalls_gbt[:-1]

f1_scores_gbt = 2 * (precisions_gbt * recalls_gbt) / (precisions_gbt + recalls_gbt + np.finfo(float).eps)

best_f1_index_gbt = np.argmax(f1_scores_gbt)
best_threshold_for_f1_gbt = thresholds_gbt[best_f1_index_gbt]
best_f1_score_gbt = f1_scores_gbt[best_f1_index_gbt]

print(f"Gradient Boosting Trees Best F1 Score: {best_f1_score_gbt}")
print(f"Gradient Boosting Trees Best Threshold for F1 Score: {best_threshold_for_f1_gbt}")

fpr_gbt, tpr_gbt, roc_thresholds_gbt = roc_curve(y_train, y_scores_gbt)

distances_gbt = np.sqrt((0 - fpr_gbt) ** 2 + (1 - tpr_gbt) ** 2)

best_threshold_index_gbt = np.argmin(distances_gbt)
best_threshold_for_roc_gbt = roc_thresholds_gbt[best_threshold_index_gbt]
best_fpr_gbt = fpr_gbt[best_threshold_index_gbt]
best_tpr_gbt = tpr_gbt[best_threshold_index_gbt]

print(f"Gradient Boosting Trees Best Threshold for ROC: {best_threshold_for_roc_gbt}")
print(f"Gradient Boosting Trees Best FPR for ROC: {best_fpr_gbt}")
print(f"Gradient Boosting Trees Best TPR for ROC: {best_tpr_gbt}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv_gbt = cross_val_predict(gbt_clf, X, y, cv=cv_strategy, method="predict_proba")[:, 1]

precisions_cv_gbt, recalls_cv_gbt, thresholds_cv_gbt = precision_recall_curve(y, y_scores_cv_gbt)

f1_scores_cv_gbt = 2 * (precisions_cv_gbt * recalls_cv_gbt) / (precisions_cv_gbt + recalls_cv_gbt + np.finfo(float).eps)

best_f1_index_cv_gbt = np.argmax(f1_scores_cv_gbt)
best_threshold_for_f1_cv_gbt = thresholds_cv_gbt[best_f1_index_cv_gbt]
best_f1_score_cv_gbt = f1_scores_cv_gbt[best_f1_index_cv_gbt]

print(f"Cross-validated Gradient Boosting Trees Best F1 Score: {best_f1_score_cv_gbt}")
print(f"Cross-validated Gradient Boosting Trees Best Threshold for F1 Score: {best_threshold_for_f1_cv_gbt}")

fpr_cv_gbt, tpr_cv_gbt, roc_thresholds_cv_gbt = roc_curve(y, y_scores_cv_gbt)

distances_cv_gbt = np.sqrt((0 - fpr_cv_gbt) ** 2 + (1 - tpr_cv_gbt) ** 2)

best_threshold_index_cv_gbt = np.argmin(distances_cv_gbt)
best_threshold_for_roc_cv_gbt = roc_thresholds_cv_gbt[best_threshold_index_cv_gbt]
best_fpr_cv_gbt = fpr_cv_gbt[best_threshold_index_cv_gbt]
best_tpr_cv_gbt = tpr_cv_gbt[best_threshold_index_cv_gbt]

print(f"Cross-validated Gradient Boosting Trees Best Threshold for ROC: {best_threshold_for_roc_cv_gbt}")
print(f"Cross-validated Gradient Boosting Trees Best FPR for ROC: {best_fpr_cv_gbt}")
print(f"Cross-validated Gradient Boosting Trees Best TPR for ROC: {best_tpr_cv_gbt}")

"""# XGBoost"""

xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

xgb_clf.fit(X_train, y_train)

# XGBoost supports predict_proba
y_scores_xgb = xgb_clf.predict_proba(X_train)[:, 1]

precisions_xgb, recalls_xgb, thresholds_xgb = precision_recall_curve(y_train, y_scores_xgb)

precisions_xgb, recalls_xgb = precisions_xgb[:-1], recalls_xgb[:-1]

f1_scores_xgb = 2 * (precisions_xgb * recalls_xgb) / (precisions_xgb + recalls_xgb + np.finfo(float).eps)

best_f1_index_xgb = np.argmax(f1_scores_xgb)
best_threshold_for_f1_xgb = thresholds_xgb[best_f1_index_xgb]
best_f1_score_xgb = f1_scores_xgb[best_f1_index_xgb]

print(f"XGBoost Best F1 Score: {best_f1_score_xgb}")
print(f"XGBoost Best Threshold for F1 Score: {best_threshold_for_f1_xgb}")

fpr_xgb, tpr_xgb, roc_thresholds_xgb = roc_curve(y_train, y_scores_xgb)

distances_xgb = np.sqrt((0 - fpr_xgb) ** 2 + (1 - tpr_xgb) ** 2)

best_threshold_index_xgb = np.argmin(distances_xgb)
best_threshold_for_roc_xgb = roc_thresholds_xgb[best_threshold_index_xgb]
best_fpr_xgb = fpr_xgb[best_threshold_index_xgb]
best_tpr_xgb = tpr_xgb[best_threshold_index_xgb]

print(f"XGBoost Best Threshold for ROC: {best_threshold_for_roc_xgb}")
print(f"XGBoost Best FPR for ROC: {best_fpr_xgb}")
print(f"XGBoost Best TPR for ROC: {best_tpr_xgb}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv_xgb = cross_val_predict(xgb_clf, X, y, cv=cv_strategy, method="predict_proba", n_jobs=-1)[:, 1]

precisions_cv_xgb, recalls_cv_xgb, thresholds_cv_xgb = precision_recall_curve(y, y_scores_cv_xgb)

precisions_cv_xgb, recalls_cv_xgb = precisions_cv_xgb[:-1], recalls_cv_xgb[:-1]

f1_scores_cv_xgb = 2 * (precisions_cv_xgb * recalls_cv_xgb) / (precisions_cv_xgb + recalls_cv_xgb + np.finfo(float).eps)

best_f1_index_cv_xgb = np.argmax(f1_scores_cv_xgb)
best_threshold_for_f1_cv_xgb = thresholds_cv_xgb[best_f1_index_cv_xgb]
best_f1_score_cv_xgb = f1_scores_cv_xgb[best_f1_index_cv_xgb]

print(f"Cross-validated XGBoost Best F1 Score: {best_f1_score_cv_xgb}")
print(f"Cross-validated XGBoost Best Threshold for F1 Score: {best_threshold_for_f1_cv_xgb}")

fpr_cv_xgb, tpr_cv_xgb, roc_thresholds_cv_xgb = roc_curve(y, y_scores_cv_xgb)

distances_cv_xgb = np.sqrt((0 - fpr_cv_xgb) ** 2 + (1 - tpr_cv_xgb) ** 2)

best_threshold_index_cv_xgb = np.argmin(distances_cv_xgb)
best_threshold_for_roc_cv_xgb = roc_thresholds_cv_xgb[best_threshold_index_cv_xgb]
best_fpr_cv_xgb = fpr_cv_xgb[best_threshold_index_cv_xgb]
best_tpr_cv_xgb = tpr_cv_xgb[best_threshold_index_cv_xgb]

print(f"Cross-validated XGBoost Best Threshold for ROC: {best_threshold_for_roc_cv_xgb}")
print(f"Cross-validated XGBoost Best FPR for ROC: {best_fpr_cv_xgb}")
print(f"Cross-validated XGBoost Best TPR for ROC: {best_tpr_cv_xgb}")

"""# AdaBoost Classifier"""

ada_clf = AdaBoostClassifier(random_state=42)

ada_clf.fit(X_train, y_train)

# AdaBoostClassifier supports predict_proba
y_scores_ada = ada_clf.predict_proba(X_train)[:, 1]

precisions_ada, recalls_ada, thresholds_ada = precision_recall_curve(y_train, y_scores_ada)

precisions_ada, recalls_ada = precisions_ada[:-1], recalls_ada[:-1]

f1_scores_ada = 2 * (precisions_ada * recalls_ada) / (precisions_ada + recalls_ada + np.finfo(float).eps)

best_f1_index_ada = np.argmax(f1_scores_ada)
best_threshold_for_f1_ada = thresholds_ada[best_f1_index_ada]
best_f1_score_ada = f1_scores_ada[best_f1_index_ada]

print(f"AdaBoost Best F1 Score: {best_f1_score_ada}")
print(f"AdaBoost Best Threshold for F1 Score: {best_threshold_for_f1_ada}")

fpr_ada, tpr_ada, roc_thresholds_ada = roc_curve(y_train, y_scores_ada)

distances_ada = np.sqrt((0 - fpr_ada) ** 2 + (1 - tpr_ada) ** 2)

best_threshold_index_ada = np.argmin(distances_ada)
best_threshold_for_roc_ada = roc_thresholds_ada[best_threshold_index_ada]
best_fpr_ada = fpr_ada[best_threshold_index_ada]
best_tpr_ada = tpr_ada[best_threshold_index_ada]

print(f"AdaBoost Best Threshold for ROC: {best_threshold_for_roc_ada}")
print(f"AdaBoost Best FPR for ROC: {best_fpr_ada}")
print(f"AdaBoost Best TPR for ROC: {best_tpr_ada}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv_ada = cross_val_predict(ada_clf, X, y, cv=cv_strategy, method="predict_proba")[:, 1]

precisions_cv_ada, recalls_cv_ada, thresholds_cv_ada = precision_recall_curve(y, y_scores_cv_ada)

precisions_cv_ada, recalls_cv_ada = precisions_cv_ada[:-1], recalls_cv_ada[:-1]

f1_scores_cv_ada = 2 * (precisions_cv_ada * recalls_cv_ada) / (precisions_cv_ada + recalls_cv_ada + np.finfo(float).eps)

best_f1_index_cv_ada = np.argmax(f1_scores_cv_ada)
best_threshold_for_f1_cv_ada = thresholds_cv_ada[best_f1_index_cv_ada]
best_f1_score_cv_ada = f1_scores_cv_ada[best_f1_index_cv_ada]

print(f"Cross-validated AdaBoost Best F1 Score: {best_f1_score_cv_ada}")
print(f"Cross-validated AdaBoost Best Threshold for F1 Score: {best_threshold_for_f1_cv_ada}")

fpr_cv_ada, tpr_cv_ada, roc_thresholds_cv_ada = roc_curve(y, y_scores_cv_ada)

distances_cv_ada = np.sqrt((0 - fpr_cv_ada) ** 2 + (1 - tpr_cv_ada) ** 2)

best_threshold_index_cv_ada = np.argmin(distances_cv_ada)
best_threshold_for_roc_cv_ada = roc_thresholds_cv_ada[best_threshold_index_cv_ada]
best_fpr_cv_ada = fpr_cv_ada[best_threshold_index_cv_ada]
best_tpr_cv_ada = tpr_cv_ada[best_threshold_index_cv_ada]

print(f"Cross-validated AdaBoost Best Threshold for ROC: {best_threshold_for_roc_cv_ada}")
print(f"Cross-validated AdaBoost Best FPR for ROC: {best_fpr_cv_ada}")
print(f"Cross-validated AdaBoost Best TPR for ROC: {best_tpr_cv_ada}")

"""# Best Model Evaluation"""

data = {
    "Fast Mode Model": [
        "Logistic Regression", "SVM for Classification", "Decision Tree Classifier",
        "Random Forest Classifier", "Gradient Boosting Trees", "XGBoost",
        "AdaBoost Classifier"
    ],
    "Fast Mode F1": [
        0.750, 0.723, 0.577, 0.762, 0.685, 0.662,
        0.667
    ]
}

df = pd.DataFrame(data)

df_sorted = df.sort_values("Fast Mode F1")

df_sorted

"""# Grid Search: Random Forest Classifier (best model)"""

# Initializs=ed Random Forest Classifier model
random_forest_clf = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],  # tree number
    'max_depth': [None, 10, 20, 30],  # tree depth
    'min_samples_split': [2, 5, 10],  # minimum number of samples required to split internal nodes
    'min_samples_leaf': [1, 2, 4]  # minimum number of samples required for leaf nodes
}

# GridSearchCV instance
grid_search_rf = GridSearchCV(estimator=random_forest_clf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)

# grid search on train set
grid_search_rf.fit(X_train, y_train)

# best score
print("Best parameters found: ", grid_search_rf.best_params_)
print("Best F1 score found: ", grid_search_rf.best_score_)

# use the model with the best parameters to acquire the predicted probability on the training set
best_rf = grid_search_rf.best_estimator_
y_scores_best_rf = best_rf.predict_proba(X_train)[:, 1]

# calculate precision, recall, and thresholds
precisions_best_rf, recalls_best_rf, thresholds_best_rf = precision_recall_curve(y_train, y_scores_best_rf)

# calculate F1 scores for all thresholds that the denominator is zero
f1_scores_best_rf = 2 * (precisions_best_rf * recalls_best_rf) / (precisions_best_rf + recalls_best_rf + np.finfo(float).eps)

# find the index corresponding to the maximum F1 score
best_f1_index_best_rf = np.argmax(f1_scores_best_rf)
best_f1_score_best_rf = f1_scores_best_rf[best_f1_index_best_rf]
best_threshold_for_f1_best_rf = thresholds_best_rf[best_f1_index_best_rf]

# use roc_curve to find optimal classification threshold
fpr_best_rf, tpr_best_rf, roc_thresholds_best_rf = roc_curve(y_train, y_scores_best_rf)
distances_best_rf = np.sqrt((0 - fpr_best_rf) ** 2 + (1 - tpr_best_rf) ** 2)
best_threshold_index_best_rf = np.argmin(distances_best_rf)
best_threshold_for_roc_best_rf = roc_thresholds_best_rf[best_threshold_index_best_rf]
best_fpr_best_rf = fpr_best_rf[best_threshold_index_best_rf]
best_tpr_best_rf = tpr_best_rf[best_threshold_index_best_rf]

print(f"Random Forest Best F1 Score on Training Set: {best_f1_score_best_rf}")
print(f"Random Forest Best Threshold for F1 Score: {best_threshold_for_f1_best_rf}")
print(f"Random Forest Best Threshold for ROC: {best_threshold_for_roc_best_rf}")
print(f"Random Forest Best FPR for ROC: {best_fpr_best_rf}")
print(f"Random Forest Best TPR for ROC: {best_tpr_best_rf}")

"""# Random Search: Random Forest Classifier"""

random_forest_clf = RandomForestClassifier(random_state=42)

param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search_rf = RandomizedSearchCV(estimator=random_forest_clf,
                                      param_distributions=param_distributions,
                                      n_iter=10,  # number of random search
                                      cv=5,  # kfold times
                                      scoring='f1',
                                      n_jobs=-1,
                                      verbose=2,
                                      random_state=42)

random_search_rf.fit(X_train, y_train)

print("Best parameters found: ", random_search_rf.best_params_)
print("Best F1 score found: ", random_search_rf.best_score_)

best_rf = random_search_rf.best_estimator_
y_scores_best_rf = best_rf.predict_proba(X_train)[:, 1]

precisions_best_rf, recalls_best_rf, thresholds_best_rf = precision_recall_curve(y_train, y_scores_best_rf)

f1_scores_best_rf = 2 * (precisions_best_rf * recalls_best_rf) / (precisions_best_rf + recalls_best_rf + np.finfo(float).eps)

best_f1_index_best_rf = np.argmax(f1_scores_best_rf)
best_f1_score_best_rf = f1_scores_best_rf[best_f1_index_best_rf]
best_threshold_for_f1_best_rf = thresholds_best_rf[best_f1_index_best_rf]

fpr_best_rf, tpr_best_rf, roc_thresholds_best_rf = roc_curve(y_train, y_scores_best_rf)
distances_best_rf = np.sqrt((0 - fpr_best_rf) ** 2 + (1 - tpr_best_rf) ** 2)
best_threshold_index_best_rf = np.argmin(distances_best_rf)
best_threshold_for_roc_best_rf = roc_thresholds_best_rf[best_threshold_index_best_rf]
best_fpr_best_rf = fpr_best_rf[best_threshold_index_best_rf]
best_tpr_best_rf = tpr_best_rf[best_threshold_index_best_rf]

print(f"Random Forest Best F1 Score on Training Set: {best_f1_score_best_rf}")
print(f"Random Forest Best Threshold for F1 Score: {best_threshold_for_f1_best_rf}")
print(f"Random Forest Best Threshold for ROC: {best_threshold_for_roc_best_rf}")
print(f"Random Forest Best FPR for ROC: {best_fpr_best_rf}")
print(f"Random Forest Best TPR for ROC: {best_tpr_best_rf}")

"""# Run on Test Set"""

# use the model with the best parameters to acquire the predicted probability on the test set
y_scores_best_rf_test = best_rf.predict_proba(X_test)[:, 1]

# calculate precision, recall, and thresholds
precisions_best_rf_test, recalls_best_rf_test, thresholds_best_rf_test = precision_recall_curve(y_test, y_scores_best_rf_test)

f1_scores_best_rf_test = 2 * (precisions_best_rf_test * recalls_best_rf_test) / (precisions_best_rf_test + recalls_best_rf_test + np.finfo(float).eps)

best_f1_index_best_rf_test = np.argmax(f1_scores_best_rf_test)
best_f1_score_best_rf_test = f1_scores_best_rf_test[best_f1_index_best_rf_test]
best_threshold_for_f1_best_rf_test = thresholds_best_rf_test[best_f1_index_best_rf_test]

fpr_best_rf_test, tpr_best_rf_test, roc_thresholds_best_rf_test = roc_curve(y_test, y_scores_best_rf_test)
distances_best_rf_test = np.sqrt((0 - fpr_best_rf_test) ** 2 + (1 - tpr_best_rf_test) ** 2)
best_threshold_index_best_rf_test = np.argmin(distances_best_rf_test)
best_threshold_for_roc_best_rf_test = roc_thresholds_best_rf_test[best_threshold_index_best_rf_test]
best_fpr_best_rf_test = fpr_best_rf_test[best_threshold_index_best_rf_test]
best_tpr_best_rf_test = tpr_best_rf_test[best_threshold_index_best_rf_test]

print(f"Random Forest Best F1 Score on Test Set: {best_f1_score_best_rf_test}")
print(f"Random Forest Best Threshold for F1 Score: {best_threshold_for_f1_best_rf_test}")
print(f"Random Forest Best Threshold for ROC: {best_threshold_for_roc_best_rf_test}")
print(f"Random Forest Best FPR for ROC: {best_fpr_best_rf_test}")
print(f"Random Forest Best TPR for ROC: {best_tpr_best_rf_test}")

"""# aveOralM_Classification"""

X = combined_df_encoded.drop(['aveOralF', 'Fever_aveOralF','Fever_aveOralM', 'aveOralM'], axis=1)
y = combined_df_encoded['Fever_aveOralM']

strf_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in strf_split.split(X, combined_df_encoded['Fever_aveOralM']):
    X_train = X.loc[train_index]
    y_train = y[train_index]
    X_test = X.loc[test_index]
    y_test = y[test_index]

"""# Logistic Regression"""

log_reg = LogisticRegression(random_state=42, max_iter=1000)

log_reg.fit(X_train, y_train)

"""##　The best score means the model at a certain training set for positive class, meaning balanced predication ability(recision and recall). The best F1 score of CV is slightly lower than the result of single training set but more reliable because it validated on different dataset.
## The CV ROC and corressponding FPR and TPR are very close to the results obtained on a training set, means model is capable of recognize binary target. Considering that ROC curve is not affected by imbalance category distribution, so the stability can be regards as good.
"""

# calculate the decision function or predicted probability of the model on the training set
y_scores = log_reg.decision_function(X_train)

# calculate precision, recall and threshold
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

# remove the last unit in precision and recall to fit the thresholds shape
precisions, recalls = precisions[:-1], recalls[:-1]

# calculate F1 scores for all thresholds that the denominator is zero
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + np.finfo(float).eps)

# find the index corresponding to the maximum F1 score
best_f1_index = np.argmax(f1_scores)
best_threshold_for_f1 = thresholds[best_f1_index]
best_f1_score = f1_scores[best_f1_index]

print(f"Best F1 Score: {best_f1_score}")
print(f"Best Threshold for F1 Score: {best_threshold_for_f1}")

# use roc_curve to find optimal classification threshold
fpr, tpr, roc_thresholds = roc_curve(y_train, y_scores)

# calculate the Euclidean distance between the TP rate and FP rate for each threshold
distances = np.sqrt((0 - fpr) ** 2 + (1 - tpr) ** 2)

# Find the index of minimum distance
best_threshold_index = np.argmin(distances)
best_threshold_for_roc = roc_thresholds[best_threshold_index]
best_fpr = fpr[best_threshold_index]
best_tpr = tpr[best_threshold_index]

print(f"Best Threshold for ROC: {best_threshold_for_roc}")
print(f"Best FPR for ROC: {best_fpr}")
print(f"Best TPR for ROC: {best_tpr}")

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv = cross_val_predict(log_reg, X, y, cv=cv_strategy, method='decision_function')

precisions_cv, recalls_cv, thresholds_cv = precision_recall_curve(y, y_scores_cv)

f1_scores_cv = 2 * (precisions_cv * recalls_cv) / (precisions_cv + recalls_cv + np.finfo(float).eps)

best_f1_index_cv = np.argmax(f1_scores_cv)
best_threshold_for_f1_cv = thresholds_cv[best_f1_index_cv]
best_f1_score_cv = f1_scores_cv[best_f1_index_cv]

print(f"Cross-validated Best F1 Score: {best_f1_score_cv}")
print(f"Cross-validated Best Threshold for F1 Score: {best_threshold_for_f1_cv}")

fpr_cv, tpr_cv, roc_thresholds_cv = roc_curve(y, y_scores_cv)

distances_cv = np.sqrt((0 - fpr_cv) ** 2 + (1 - tpr_cv) ** 2)

best_threshold_index_cv = np.argmin(distances_cv)
best_threshold_for_roc_cv = roc_thresholds_cv[best_threshold_index_cv]
best_fpr_cv = fpr_cv[best_threshold_index_cv]
best_tpr_cv = tpr_cv[best_threshold_index_cv]

print(f"Cross-validated Best Threshold for ROC: {best_threshold_for_roc_cv}")
print(f"Cross-validated Best FPR for ROC: {best_fpr_cv}")
print(f"Cross-validated Best TPR for ROC: {best_tpr_cv}")

"""# SVM for Classification"""

svm_clf = SVC(random_state=42, probability=True)

svm_clf.fit(X_train, y_train)

y_scores_svm = svm_clf.decision_function(X_train)

precisions_svm, recalls_svm, thresholds_svm = precision_recall_curve(y_train, y_scores_svm)

precisions_svm, recalls_svm = precisions_svm[:-1], recalls_svm[:-1]

f1_scores_svm = 2 * (precisions_svm * recalls_svm) / (precisions_svm + recalls_svm + np.finfo(float).eps)

best_f1_index_svm = np.argmax(f1_scores_svm)
best_threshold_for_f1_svm = thresholds_svm[best_f1_index_svm]
best_f1_score_svm = f1_scores_svm[best_f1_index_svm]

print(f"SVM Best F1 Score: {best_f1_score_svm}")
print(f"SVM Best Threshold for F1 Score: {best_threshold_for_f1_svm}")

fpr_svm, tpr_svm, roc_thresholds_svm = roc_curve(y_train, y_scores_svm)

distances_svm = np.sqrt((0 - fpr_svm) ** 2 + (1 - tpr_svm) ** 2)

best_threshold_index_svm = np.argmin(distances_svm)
best_threshold_for_roc_svm = roc_thresholds_svm[best_threshold_index_svm]
best_fpr_svm = fpr_svm[best_threshold_index_svm]
best_tpr_svm = tpr_svm[best_threshold_index_svm]

print(f"SVM Best Threshold for ROC: {best_threshold_for_roc_svm}")
print(f"SVM Best FPR for ROC: {best_fpr_svm}")
print(f"SVM Best TPR for ROC: {best_tpr_svm}")

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_clf = SVC(random_state=42, probability=True)

y_scores_cv_svm = cross_val_predict(svm_clf, X, y, cv=cv_strategy, method='decision_function')

precisions_cv_svm, recalls_cv_svm, thresholds_cv_svm = precision_recall_curve(y, y_scores_cv_svm)

f1_scores_cv_svm = 2 * (precisions_cv_svm * recalls_cv_svm) / (precisions_cv_svm + recalls_cv_svm + np.finfo(float).eps)

best_f1_index_cv_svm = np.argmax(f1_scores_cv_svm)
best_threshold_for_f1_cv_svm = thresholds_cv_svm[best_f1_index_cv_svm]
best_f1_score_cv_svm = f1_scores_cv_svm[best_f1_index_cv_svm]

print(f"Cross-validated SVM Best F1 Score: {best_f1_score_cv_svm}")
print(f"Cross-validated SVM Best Threshold for F1 Score: {best_threshold_for_f1_cv_svm}")

fpr_cv_svm, tpr_cv_svm, roc_thresholds_cv_svm = roc_curve(y, y_scores_cv_svm)

distances_cv_svm = np.sqrt((0 - fpr_cv_svm) ** 2 + (1 - tpr_cv_svm) ** 2)

best_threshold_index_cv_svm = np.argmin(distances_cv_svm)
best_threshold_for_roc_cv_svm = roc_thresholds_cv_svm[best_threshold_index_cv_svm]
best_fpr_cv_svm = fpr_cv_svm[best_threshold_index_cv_svm]
best_tpr_cv_svm = tpr_cv_svm[best_threshold_index_cv_svm]

print(f"Cross-validated SVM Best Threshold for ROC: {best_threshold_for_roc_cv_svm}")
print(f"Cross-validated SVM Best FPR for ROC: {best_fpr_cv_svm}")
print(f"Cross-validated SVM Best TPR for ROC: {best_tpr_cv_svm}")

"""# Decision Tree Classifier"""

decision_tree_clf = DecisionTreeClassifier(random_state=42)

decision_tree_clf.fit(X_train, y_train)

y_scores_dt = decision_tree_clf.predict_proba(X_train)[:, 1]

precisions_dt, recalls_dt, thresholds_dt = precision_recall_curve(y_train, y_scores_dt)

precisions_dt, recalls_dt = precisions_dt[:-1], recalls_dt[:-1]

f1_scores_dt = 2 * (precisions_dt * recalls_dt) / (precisions_dt + recalls_dt + np.finfo(float).eps)

best_f1_index_dt = np.argmax(f1_scores_dt)
best_threshold_for_f1_dt = thresholds_dt[best_f1_index_dt]
best_f1_score_dt = f1_scores_dt[best_f1_index_dt]

print(f"Decision Tree Best F1 Score: {best_f1_score_dt}")
print(f"Decision Tree Best Threshold for F1 Score: {best_threshold_for_f1_dt}")

fpr_dt, tpr_dt, roc_thresholds_dt = roc_curve(y_train, y_scores_dt)

distances_dt = np.sqrt((0 - fpr_dt) ** 2 + (1 - tpr_dt) ** 2)

best_threshold_index_dt = np.argmin(distances_dt)
best_threshold_for_roc_dt = roc_thresholds_dt[best_threshold_index_dt]
best_fpr_dt = fpr_dt[best_threshold_index_dt]
best_tpr_dt = tpr_dt[best_threshold_index_dt]

print(f"Decision Tree Best Threshold for ROC: {best_threshold_for_roc_dt}")
print(f"Decision Tree Best FPR for ROC: {best_fpr_dt}")
print(f"Decision Tree Best TPR for ROC: {best_tpr_dt}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv_dt = cross_val_predict(decision_tree_clf, X, y, cv=cv_strategy, method="predict_proba")[:, 1]

precisions_cv_dt, recalls_cv_dt, thresholds_cv_dt = precision_recall_curve(y, y_scores_cv_dt)

precisions_cv_dt, recalls_cv_dt = precisions_cv_dt[:-1], recalls_cv_dt[:-1]

f1_scores_cv_dt = 2 * (precisions_cv_dt * recalls_cv_dt) / (precisions_cv_dt + recalls_cv_dt + np.finfo(float).eps)

best_f1_index_cv_dt = np.argmax(f1_scores_cv_dt)
best_threshold_for_f1_cv_dt = thresholds_cv_dt[best_f1_index_cv_dt]
best_f1_score_cv_dt = f1_scores_cv_dt[best_f1_index_cv_dt]

print(f"Cross-validated Decision Tree Best F1 Score: {best_f1_score_cv_dt}")
print(f"Cross-validated Decision Tree Best Threshold for F1 Score: {best_threshold_for_f1_cv_dt}")

fpr_cv_dt, tpr_cv_dt, roc_thresholds_cv_dt = roc_curve(y, y_scores_cv_dt)

distances_cv_dt = np.sqrt((0 - fpr_cv_dt) ** 2 + (1 - tpr_cv_dt) ** 2)

best_threshold_index_cv_dt = np.argmin(distances_cv_dt)
best_threshold_for_roc_cv_dt = roc_thresholds_cv_dt[best_threshold_index_cv_dt]
best_fpr_cv_dt = fpr_cv_dt[best_threshold_index_cv_dt]
best_tpr_cv_dt = tpr_cv_dt[best_threshold_index_cv_dt]

print(f"Cross-validated Decision Tree Best Threshold for ROC: {best_threshold_for_roc_cv_dt}")
print(f"Cross-validated Decision Tree Best FPR for ROC: {best_fpr_cv_dt}")
print(f"Cross-validated Decision Tree Best TPR for ROC: {best_tpr_cv_dt}")

"""# Random Forest Classifier"""

random_forest_clf = RandomForestClassifier(random_state=42)

random_forest_clf.fit(X_train, y_train)

y_scores_rf = random_forest_clf.predict_proba(X_train)[:, 1]

precisions_rf, recalls_rf, thresholds_rf = precision_recall_curve(y_train, y_scores_rf)

precisions_rf, recalls_rf = precisions_rf[:-1], recalls_rf[:-1]

f1_scores_rf = 2 * (precisions_rf * recalls_rf) / (precisions_rf + recalls_rf + np.finfo(float).eps)

best_f1_index_rf = np.argmax(f1_scores_rf)
best_threshold_for_f1_rf = thresholds_rf[best_f1_index_rf]
best_f1_score_rf = f1_scores_rf[best_f1_index_rf]

print(f"Random Forest Best F1 Score: {best_f1_score_rf}")
print(f"Random Forest Best Threshold for F1 Score: {best_threshold_for_f1_rf}")

fpr_rf, tpr_rf, roc_thresholds_rf = roc_curve(y_train, y_scores_rf)

distances_rf = np.sqrt((0 - fpr_rf) ** 2 + (1 - tpr_rf) ** 2)

best_threshold_index_rf = np.argmin(distances_rf)
best_threshold_for_roc_rf = roc_thresholds_rf[best_threshold_index_rf]
best_fpr_rf = fpr_rf[best_threshold_index_rf]
best_tpr_rf = tpr_rf[best_threshold_index_rf]

print(f"Random Forest Best Threshold for ROC: {best_threshold_for_roc_rf}")
print(f"Random Forest Best FPR for ROC: {best_fpr_rf}")
print(f"Random Forest Best TPR for ROC: {best_tpr_rf}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv_rf = cross_val_predict(random_forest_clf, X, y, cv=cv_strategy, method="predict_proba")[:, 1]

precisions_cv_rf, recalls_cv_rf, thresholds_cv_rf = precision_recall_curve(y, y_scores_cv_rf)

precisions_cv_rf, recalls_cv_rf = precisions_cv_rf[:-1], recalls_cv_rf[:-1]

f1_scores_cv_rf = 2 * (precisions_cv_rf * recalls_cv_rf) / (precisions_cv_rf + recalls_cv_rf + np.finfo(float).eps)

best_f1_index_cv_rf = np.argmax(f1_scores_cv_rf)
best_threshold_for_f1_cv_rf = thresholds_cv_rf[best_f1_index_cv_rf]
best_f1_score_cv_rf = f1_scores_cv_rf[best_f1_index_cv_rf]

print(f"Cross-validated Random Forest Best F1 Score: {best_f1_score_cv_rf}")
print(f"Cross-validated Random Forest Best Threshold for F1 Score: {best_threshold_for_f1_cv_rf}")

fpr_cv_rf, tpr_cv_rf, roc_thresholds_cv_rf = roc_curve(y, y_scores_cv_rf)

distances_cv_rf = np.sqrt((0 - fpr_cv_rf) ** 2 + (1 - tpr_cv_rf) ** 2)

best_threshold_index_cv_rf = np.argmin(distances_cv_rf)
best_threshold_for_roc_cv_rf = roc_thresholds_cv_rf[best_threshold_index_cv_rf]
best_fpr_cv_rf = fpr_cv_rf[best_threshold_index_cv_rf]
best_tpr_cv_rf = tpr_cv_rf[best_threshold_index_cv_rf]

print(f"Cross-validated Random Forest Best Threshold for ROC: {best_threshold_for_roc_cv_rf}")
print(f"Cross-validated Random Forest Best FPR for ROC: {best_fpr_cv_rf}")
print(f"Cross-validated Random Forest Best TPR for ROC: {best_tpr_cv_rf}")

"""# Gradient Boosting Trees"""

gbt_clf = GradientBoostingClassifier(random_state=42)

gbt_clf.fit(X_train, y_train)

y_scores_gbt = gbt_clf.predict_proba(X_train)[:, 1]

precisions_gbt, recalls_gbt, thresholds_gbt = precision_recall_curve(y_train, y_scores_gbt)

precisions_gbt, recalls_gbt = precisions_gbt[:-1], recalls_gbt[:-1]

f1_scores_gbt = 2 * (precisions_gbt * recalls_gbt) / (precisions_gbt + recalls_gbt + np.finfo(float).eps)

best_f1_index_gbt = np.argmax(f1_scores_gbt)
best_threshold_for_f1_gbt = thresholds_gbt[best_f1_index_gbt]
best_f1_score_gbt = f1_scores_gbt[best_f1_index_gbt]

print(f"Gradient Boosting Trees Best F1 Score: {best_f1_score_gbt}")
print(f"Gradient Boosting Trees Best Threshold for F1 Score: {best_threshold_for_f1_gbt}")

fpr_gbt, tpr_gbt, roc_thresholds_gbt = roc_curve(y_train, y_scores_gbt)

distances_gbt = np.sqrt((0 - fpr_gbt) ** 2 + (1 - tpr_gbt) ** 2)

best_threshold_index_gbt = np.argmin(distances_gbt)
best_threshold_for_roc_gbt = roc_thresholds_gbt[best_threshold_index_gbt]
best_fpr_gbt = fpr_gbt[best_threshold_index_gbt]
best_tpr_gbt = tpr_gbt[best_threshold_index_gbt]

print(f"Gradient Boosting Trees Best Threshold for ROC: {best_threshold_for_roc_gbt}")
print(f"Gradient Boosting Trees Best FPR for ROC: {best_fpr_gbt}")
print(f"Gradient Boosting Trees Best TPR for ROC: {best_tpr_gbt}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv_gbt = cross_val_predict(gbt_clf, X, y, cv=cv_strategy, method="predict_proba")[:, 1]

precisions_cv_gbt, recalls_cv_gbt, thresholds_cv_gbt = precision_recall_curve(y, y_scores_cv_gbt)

f1_scores_cv_gbt = 2 * (precisions_cv_gbt * recalls_cv_gbt) / (precisions_cv_gbt + recalls_cv_gbt + np.finfo(float).eps)

best_f1_index_cv_gbt = np.argmax(f1_scores_cv_gbt)
best_threshold_for_f1_cv_gbt = thresholds_cv_gbt[best_f1_index_cv_gbt]
best_f1_score_cv_gbt = f1_scores_cv_gbt[best_f1_index_cv_gbt]

print(f"Cross-validated Gradient Boosting Trees Best F1 Score: {best_f1_score_cv_gbt}")
print(f"Cross-validated Gradient Boosting Trees Best Threshold for F1 Score: {best_threshold_for_f1_cv_gbt}")

fpr_cv_gbt, tpr_cv_gbt, roc_thresholds_cv_gbt = roc_curve(y, y_scores_cv_gbt)

distances_cv_gbt = np.sqrt((0 - fpr_cv_gbt) ** 2 + (1 - tpr_cv_gbt) ** 2)

best_threshold_index_cv_gbt = np.argmin(distances_cv_gbt)
best_threshold_for_roc_cv_gbt = roc_thresholds_cv_gbt[best_threshold_index_cv_gbt]
best_fpr_cv_gbt = fpr_cv_gbt[best_threshold_index_cv_gbt]
best_tpr_cv_gbt = tpr_cv_gbt[best_threshold_index_cv_gbt]

print(f"Cross-validated Gradient Boosting Trees Best Threshold for ROC: {best_threshold_for_roc_cv_gbt}")
print(f"Cross-validated Gradient Boosting Trees Best FPR for ROC: {best_fpr_cv_gbt}")
print(f"Cross-validated Gradient Boosting Trees Best TPR for ROC: {best_tpr_cv_gbt}")

"""# XGBoost"""

xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

xgb_clf.fit(X_train, y_train)

y_scores_xgb = xgb_clf.predict_proba(X_train)[:, 1]

precisions_xgb, recalls_xgb, thresholds_xgb = precision_recall_curve(y_train, y_scores_xgb)

precisions_xgb, recalls_xgb = precisions_xgb[:-1], recalls_xgb[:-1]

f1_scores_xgb = 2 * (precisions_xgb * recalls_xgb) / (precisions_xgb + recalls_xgb + np.finfo(float).eps)

best_f1_index_xgb = np.argmax(f1_scores_xgb)
best_threshold_for_f1_xgb = thresholds_xgb[best_f1_index_xgb]
best_f1_score_xgb = f1_scores_xgb[best_f1_index_xgb]

print(f"XGBoost Best F1 Score: {best_f1_score_xgb}")
print(f"XGBoost Best Threshold for F1 Score: {best_threshold_for_f1_xgb}")

fpr_xgb, tpr_xgb, roc_thresholds_xgb = roc_curve(y_train, y_scores_xgb)

distances_xgb = np.sqrt((0 - fpr_xgb) ** 2 + (1 - tpr_xgb) ** 2)

best_threshold_index_xgb = np.argmin(distances_xgb)
best_threshold_for_roc_xgb = roc_thresholds_xgb[best_threshold_index_xgb]
best_fpr_xgb = fpr_xgb[best_threshold_index_xgb]
best_tpr_xgb = tpr_xgb[best_threshold_index_xgb]

print(f"XGBoost Best Threshold for ROC: {best_threshold_for_roc_xgb}")
print(f"XGBoost Best FPR for ROC: {best_fpr_xgb}")
print(f"XGBoost Best TPR for ROC: {best_tpr_xgb}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv_xgb = cross_val_predict(xgb_clf, X, y, cv=cv_strategy, method="predict_proba", n_jobs=-1)[:, 1]

precisions_cv_xgb, recalls_cv_xgb, thresholds_cv_xgb = precision_recall_curve(y, y_scores_cv_xgb)

precisions_cv_xgb, recalls_cv_xgb = precisions_cv_xgb[:-1], recalls_cv_xgb[:-1]

f1_scores_cv_xgb = 2 * (precisions_cv_xgb * recalls_cv_xgb) / (precisions_cv_xgb + recalls_cv_xgb + np.finfo(float).eps)

best_f1_index_cv_xgb = np.argmax(f1_scores_cv_xgb)
best_threshold_for_f1_cv_xgb = thresholds_cv_xgb[best_f1_index_cv_xgb]
best_f1_score_cv_xgb = f1_scores_cv_xgb[best_f1_index_cv_xgb]

print(f"Cross-validated XGBoost Best F1 Score: {best_f1_score_cv_xgb}")
print(f"Cross-validated XGBoost Best Threshold for F1 Score: {best_threshold_for_f1_cv_xgb}")

fpr_cv_xgb, tpr_cv_xgb, roc_thresholds_cv_xgb = roc_curve(y, y_scores_cv_xgb)

distances_cv_xgb = np.sqrt((0 - fpr_cv_xgb) ** 2 + (1 - tpr_cv_xgb) ** 2)

best_threshold_index_cv_xgb = np.argmin(distances_cv_xgb)
best_threshold_for_roc_cv_xgb = roc_thresholds_cv_xgb[best_threshold_index_cv_xgb]
best_fpr_cv_xgb = fpr_cv_xgb[best_threshold_index_cv_xgb]
best_tpr_cv_xgb = tpr_cv_xgb[best_threshold_index_cv_xgb]

print(f"Cross-validated XGBoost Best Threshold for ROC: {best_threshold_for_roc_cv_xgb}")
print(f"Cross-validated XGBoost Best FPR for ROC: {best_fpr_cv_xgb}")
print(f"Cross-validated XGBoost Best TPR for ROC: {best_tpr_cv_xgb}")

"""# AdaBoost Classifier"""

ada_clf = AdaBoostClassifier(random_state=42)

ada_clf.fit(X_train, y_train)

y_scores_ada = ada_clf.predict_proba(X_train)[:, 1]

precisions_ada, recalls_ada, thresholds_ada = precision_recall_curve(y_train, y_scores_ada)

precisions_ada, recalls_ada = precisions_ada[:-1], recalls_ada[:-1]

f1_scores_ada = 2 * (precisions_ada * recalls_ada) / (precisions_ada + recalls_ada + np.finfo(float).eps)

best_f1_index_ada = np.argmax(f1_scores_ada)
best_threshold_for_f1_ada = thresholds_ada[best_f1_index_ada]
best_f1_score_ada = f1_scores_ada[best_f1_index_ada]

print(f"AdaBoost Best F1 Score: {best_f1_score_ada}")
print(f"AdaBoost Best Threshold for F1 Score: {best_threshold_for_f1_ada}")

fpr_ada, tpr_ada, roc_thresholds_ada = roc_curve(y_train, y_scores_ada)

distances_ada = np.sqrt((0 - fpr_ada) ** 2 + (1 - tpr_ada) ** 2)

best_threshold_index_ada = np.argmin(distances_ada)
best_threshold_for_roc_ada = roc_thresholds_ada[best_threshold_index_ada]
best_fpr_ada = fpr_ada[best_threshold_index_ada]
best_tpr_ada = tpr_ada[best_threshold_index_ada]

print(f"AdaBoost Best Threshold for ROC: {best_threshold_for_roc_ada}")
print(f"AdaBoost Best FPR for ROC: {best_fpr_ada}")
print(f"AdaBoost Best TPR for ROC: {best_tpr_ada}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_scores_cv_ada = cross_val_predict(ada_clf, X, y, cv=cv_strategy, method="predict_proba")[:, 1]

precisions_cv_ada, recalls_cv_ada, thresholds_cv_ada = precision_recall_curve(y, y_scores_cv_ada)

precisions_cv_ada, recalls_cv_ada = precisions_cv_ada[:-1], recalls_cv_ada[:-1]

f1_scores_cv_ada = 2 * (precisions_cv_ada * recalls_cv_ada) / (precisions_cv_ada + recalls_cv_ada + np.finfo(float).eps)

best_f1_index_cv_ada = np.argmax(f1_scores_cv_ada)
best_threshold_for_f1_cv_ada = thresholds_cv_ada[best_f1_index_cv_ada]
best_f1_score_cv_ada = f1_scores_cv_ada[best_f1_index_cv_ada]

print(f"Cross-validated AdaBoost Best F1 Score: {best_f1_score_cv_ada}")
print(f"Cross-validated AdaBoost Best Threshold for F1 Score: {best_threshold_for_f1_cv_ada}")

fpr_cv_ada, tpr_cv_ada, roc_thresholds_cv_ada = roc_curve(y, y_scores_cv_ada)

distances_cv_ada = np.sqrt((0 - fpr_cv_ada) ** 2 + (1 - tpr_cv_ada) ** 2)

best_threshold_index_cv_ada = np.argmin(distances_cv_ada)
best_threshold_for_roc_cv_ada = roc_thresholds_cv_ada[best_threshold_index_cv_ada]
best_fpr_cv_ada = fpr_cv_ada[best_threshold_index_cv_ada]
best_tpr_cv_ada = tpr_cv_ada[best_threshold_index_cv_ada]

print(f"Cross-validated AdaBoost Best Threshold for ROC: {best_threshold_for_roc_cv_ada}")
print(f"Cross-validated AdaBoost Best FPR for ROC: {best_fpr_cv_ada}")
print(f"Cross-validated AdaBoost Best TPR for ROC: {best_tpr_cv_ada}")



"""# Best Model Evaluation"""

data = {
    "Monitor Mode Model": [
        "Logistic Regression", "SVM for Classification", "Decision Tree Classifier",
        "Random Forest Classifier", "Gradient Boosting Trees", "XGBoost",
        "AdaBoost Classifier"
    ],
    "Monitor Mode F1": [
        0.883, 0.879, 0.735, 0.816, 0.810, 0.807,
        0.766
    ]
}

df = pd.DataFrame(data)
# Sort the DataFrame based on RMSE from low to high
df_sorted = df.sort_values("Monitor Mode F1")

df_sorted

"""# Same Result: Grid Search and Random Search
## Both GridSearchCV and RandomizedSearchCV address the same optimise parameter combination, {'C': 11.288378916846883, 'solver': 'liblinear'}. It could because the C's range is set to np.logspace(-4, 4, 20), and n_iter=100 is set in the random search, means the model have a huge room to find the best parameter. Besides,  the random state both are 42 , which means for even if the serach is randomly, but it could follow the same pattern.

# Grid Search: Logistic Regression (best model)
"""

logistic_reg_clf = LogisticRegression(random_state=42, max_iter=1000)

# Define the parameter grid
param_grid_lr = {
    'C': np.logspace(-4, 4, 20),  # the reciprocal of regularization strength
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']  # optimization
}

# create GridSearchCV instance
grid_search_lr = GridSearchCV(estimator=logistic_reg_clf, param_grid=param_grid_lr, cv=5, scoring='f1', n_jobs=-1, verbose=2)

# grid search on train set
grid_search_lr.fit(X_train, y_train)

print("Best parameters found: ", grid_search_lr.best_params_)
print("Best F1 score found: ", grid_search_lr.best_score_)

# use the model with the best parameters to qcquire the predicted probability on the training set
best_lr = grid_search_lr.best_estimator_
y_scores_best_lr = best_lr.predict_proba(X_train)[:, 1]

# calculate precision, recall, and thresholds
precisions_best_lr, recalls_best_lr, thresholds_best_lr = precision_recall_curve(y_train, y_scores_best_lr)

# calculate F1 scores for all thresholds that the denominator is zero
f1_scores_best_lr = 2 * (precisions_best_lr * recalls_best_lr) / (precisions_best_lr + recalls_best_lr + np.finfo(float).eps)

# find the index corresponding to the maximum F1 score
best_f1_index_best_lr = np.argmax(f1_scores_best_lr)
best_f1_score_best_lr = f1_scores_best_lr[best_f1_index_best_lr]
best_threshold_for_f1_best_lr = thresholds_best_lr[best_f1_index_best_lr]

# use roc_curve to find optimal classification threshold
fpr_best_lr, tpr_best_lr, roc_thresholds_best_lr = roc_curve(y_train, y_scores_best_lr)
distances_best_lr = np.sqrt((0 - fpr_best_lr) ** 2 + (1 - tpr_best_lr) ** 2)
best_threshold_index_best_lr = np.argmin(distances_best_lr)
best_threshold_for_roc_best_lr = roc_thresholds_best_lr[best_threshold_index_best_lr]

print(f"Logistic Regression Best F1 Score on Training Set: {best_f1_score_best_lr}")
print(f"Logistic Regression Best Threshold for F1 Score: {best_threshold_for_f1_best_lr}")
print(f"Logistic Regression Best Threshold for ROC: {best_threshold_for_roc_best_lr}")

"""# Random Search: Logistic Regression"""

# initialize the Logistic Regression model
logistic_reg_clf = LogisticRegression(random_state=42)

# define the parametric distribution to be randomly searched
param_distributions = {
    'C': np.logspace(-4, 4, 20),  # the reciprocal of regularization strength
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']  # optimization
}

# create RandomizedSearchCV instance
random_search_lr = RandomizedSearchCV(estimator=logistic_reg_clf, param_distributions=param_distributions,
                                      n_iter=100, cv=5, scoring='f1', n_jobs=-1, verbose=2, random_state=42)

# perform random search on the training set
random_search_lr.fit(X_train, y_train)

# best parameters
print("Best parameters found: ", random_search_lr.best_params_)
print("Best F1 score found: ", random_search_lr.best_score_)

# use the model with the best parameters to qcquire the predicted probability on the training set
best_lr = random_search_lr.best_estimator_
y_scores_best_lr = best_lr.predict_proba(X_train)[:, 1]

# calculate precision, recall, and thresholds
precisions_best_lr, recalls_best_lr, thresholds_best_lr = precision_recall_curve(y_train, y_scores_best_lr)

# calculate F1 scores for all thresholds that the denominator is zero
f1_scores_best_lr = 2 * (precisions_best_lr * recalls_best_lr) / (precisions_best_lr + recalls_best_lr + np.finfo(float).eps)

# find the index corresponding to the maximum F1 score
best_f1_index_best_lr = np.argmax(f1_scores_best_lr)
best_f1_score_best_lr = f1_scores_best_lr[best_f1_index_best_lr]
best_threshold_for_f1_best_lr = thresholds_best_lr[best_f1_index_best_lr]

# use roc_curve to find optimal classification threshold
fpr_best_lr, tpr_best_lr, roc_thresholds_best_lr = roc_curve(y_train, y_scores_best_lr)
distances_best_lr = np.sqrt((0 - fpr_best_lr) ** 2 + (1 - tpr_best_lr) ** 2)
best_threshold_index_best_lr = np.argmin(distances_best_lr)
best_threshold_for_roc_best_lr = roc_thresholds_best_lr[best_threshold_index_best_lr]

print(f"Logistic Regression Best F1 Score on Training Set: {best_f1_score_best_lr}")
print(f"Logistic Regression Best Threshold for F1 Score: {best_threshold_for_f1_best_lr}")
print(f"Logistic Regression Best Threshold for ROC: {best_threshold_for_roc_best_lr}")

"""# Run on Test Set"""

# use the model with the best parameters to acquire the predicted probability on the test set
y_scores_best_lr_test = best_lr.predict_proba(X_test)[:, 1]

# calculate precision, recall, and thresholds
precisions_best_lr_test, recalls_best_lr_test, thresholds_best_lr_test = precision_recall_curve(y_test, y_scores_best_lr_test)

# calculate F1 scores for all thresholds that the denominator is zero
f1_scores_best_lr_test = 2 * (precisions_best_lr_test * recalls_best_lr_test) / (precisions_best_lr_test + recalls_best_lr_test + np.finfo(float).eps)

# find the index related to the maximum F1 score
best_f1_index_best_lr_test = np.argmax(f1_scores_best_lr_test)
best_f1_score_best_lr_test = f1_scores_best_lr_test[best_f1_index_best_lr_test]
best_threshold_for_f1_best_lr_test = thresholds_best_lr_test[best_f1_index_best_lr_test]

# use roc_curve to find optimal classification threshold
fpr_best_lr_test, tpr_best_lr_test, roc_thresholds_best_lr_test = roc_curve(y_test, y_scores_best_lr_test)
distances_best_lr_test = np.sqrt((0 - fpr_best_lr_test) ** 2 + (1 - tpr_best_lr_test) ** 2)
best_threshold_index_best_lr_test = np.argmin(distances_best_lr_test)
best_threshold_for_roc_best_lr_test = roc_thresholds_best_lr_test[best_threshold_index_best_lr_test]
best_fpr_best_lr_test = fpr_best_lr_test[best_threshold_index_best_lr_test]
best_tpr_best_lr_test = tpr_best_lr_test[best_threshold_index_best_lr_test]

# print the best F1 score and the corresponding threshold and the best threshold for the ROC curve
print(f"Logistic Regression Best F1 Score on Test Set: {best_f1_score_best_lr_test}")
print(f"Logistic Regression Best Threshold for F1 Score: {best_threshold_for_f1_best_lr_test}")
print(f"Logistic Regression Best Threshold for ROC: {best_threshold_for_roc_best_lr_test}")
print(f"Logistic Regression Best FPR for ROC: {best_fpr_best_lr_test}")
print(f"Logistic Regression Best TPR for ROC: {best_tpr_best_lr_test}")

"""# Python packages"""

!pip list
