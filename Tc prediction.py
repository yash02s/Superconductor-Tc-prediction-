from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt

# Read the data from a csv file
data = pd.read_csv('superconductor_data.csv')

# Split data into training and testing sets
X = data.drop(['critical_temp'], axis=1)  # Input features
y = data['critical_temp']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Ridge regression
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

# Lasso regression
lasso = Lasso(alpha=0.5)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

# Elastic-net regression
elastic = ElasticNet(alpha=0.5, l1_ratio=0.5)
elastic.fit(X_train, y_train)
elastic_pred = elastic.predict(X_test)

# Decision tree
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train, y_train)
dtr_pred = dtr.predict(X_test)

# XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000,
                         max_depth=5, learning_rate=0.2, colsample_bytree=0.5)
model.fit(X_train, y_train)

model_pred = model.predict(X_test)


#residual values
residuals_test = y_test - dtr_pred

# pdf = pd.DataFrame()
# pdf['Actual Values'] = y_test
# pdf['Predicted Values'] = model_pred
# pdf.reset_index(drop=True, inplace=True)
# print(pdf)

# pdf1 = pd.DataFrame()
# pdf1['Actual Values'] = y_test
# pdf1['Predicted Values'] = dtr_pred
# pdf1.reset_index(drop=True, inplace=True)
# print(pdf1)

# pdf2 = pd.DataFrame()
# pdf2['Actual Values'] = y_test
# pdf2['Predicted Values'] = lasso_pred
# pdf2.reset_index(drop=True, inplace=True)
# print(pdf2)


# plt.scatter(model_pred, residuals_test)
# plt.axhline(y=0, color='r', linestyle='-')
# plt.xlabel('Predicted values')
# plt.ylabel('Residuals')
# plt.title('Residual plot')
# plt.show()

# plt.scatter(dtr_pred, residuals_test)
# plt.axhline(y=0, color='r', linestyle='-')
# plt.xlabel('Predicted values')
# plt.ylabel('Residuals')
# plt.title('Residual plot')
# plt.show()


# Cross Validation for XGBoost
scores = cross_val_score(model, X, y, cv=5)
# Print the accuracy scores for each fold
# print("Accuracy scores in each fold for XGBoost model:", scores)

# Calculate the mean and standard deviation of the scores
mean_score = scores.mean()
std_score = scores.std()

# print("Mean accuracy for XGBoost model:", mean_score)
# print("Standard deviation of accuracy for XGBoost model:", std_score)


# Cross Validation for DT
scores1 = cross_val_score(dtr, X, y, cv=5)
# Print the accuracy scores for each fold
# print("Accuracy scores in each fold for DT regresion model:", scores1)

# Calculate the mean and standard deviation of the scores
mean_score1 = scores1.mean()
std_score1 = scores1.std()

# print("Mean accuracy for DT model:", mean_score1)
# print("Standard deviation of accuracy for DT model:", std_score1)

# plt.scatter(dtr_pred, model_pred)
# plt.plot([min(dtr_pred), max(dtr_pred)], [min(dtr_pred), max(dtr_pred)], 'k--')
# plt.axhline(y=0, color='red', linestyle='-')
# plt.xlabel('DT Predicted values')
# plt.ylabel('XGBoost predicted values')
# plt.show()

tolerance = 0.01  # set a tolerance level
count = 0

# loop over all points and count the ones lying on y=x line
# for i in range(len(dtr_pred)):
#     if abs(dtr_pred[i] - model_pred[i]) < tolerance:
#         count += 1

# print("Number of points lying on y=x line:", count)



# Print the results
# print('Ridge RMSE:', np.sqrt(np.mean((ridge_pred - y_test) ** 2)))
# print('Lasso RMSE:', np.sqrt(np.mean((lasso_pred - y_test) ** 2)))
# print('Elastic-net RMSE:', np.sqrt(np.mean((elastic_pred - y_test) ** 2)))
# print('DT RMSE:', np.sqrt(mean_squared_error(y_test, dtr_pred)))
# print('XGBoost RMSE:', np.sqrt(np.mean((y_test - model_pred) ** 2)))
# print('Ridge R2:', r2_score(y_test, ridge_pred))
# print('Lasso R2:', r2_score(y_test, lasso_pred))
# print('Elastic-net R2:', r2_score(y_test, elastic_pred))
# print('DT R2:', r2_score(y_test, dtr_pred))
# print('XGBoost R2:', r2_score(y_test, model_pred))
# print('Ridge MAE:', mean_absolute_error(y_test, ridge_pred))
# print('Lasso MAE:', mean_absolute_error(y_test, lasso_pred))
# print('Elastic-net MAE:', mean_absolute_error(y_test, elastic_pred))
# print('DT MAE:', mean_absolute_error(y_test, dtr_pred))
# print('XGBoost MAE:', mean_absolute_error(y_test, model_pred))


# rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# Feature importance



pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)


rf = RandomForestRegressor()
rf.fit(X_pca, y)

importances = rf.feature_importances_
feature_names = data.columns[:-1]
indices = np.argsort(importances)[::-1]
names = [feature_names[i] for i in indices]

print('Top 10 Features:')
for i in range(10):
    print('%d. %s (%f)' %
          (i + 1, feature_names[indices[i]], importances[indices[i]]))

# width = 0.3
# gap = 0.5
# plt.figure()

# # Create plot title
# plt.title("Feature Importance")

# # Add bars
# plt.bar(np.arange(X.shape[1]-1)*(width + gap), importances[indices],width=width)

# # Add feature names as x-axis labels
# plt.xticks(np.arange(X.shape[1]-1)*(width + gap) + width/2, names, fontsize=8)

# # Show plot
# plt.show()


# print('Top 10 Features:')
# for i in range(10):
#     print('%d. %s (%f)' %
#           (i + 1, feature_names[indices[i]], importances[indices[i]]))