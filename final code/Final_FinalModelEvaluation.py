#Import used liberies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, cross_val_score, KFold)
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Import the pre-processing dataset into a varibale
df = pd.read_csv("/Users/scottspreadborough/Downloads/final_with_no_missing_data.csv")

#Collect all the final model features
features = [
    'sleep_efficiency', 'sleep_score', 'deep_sleep',
    'rem_sleep', 'hr_avg', 'hrv',
    'temp_delta'
]

#Array for each models r squared value in the model evaluation table
results = []

X = df[features]
y = df['isolate']

#Split the data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Create the final chosen model
model = ExtraTreesRegressor(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)

model.fit(X_train, y_train)

#Predict social isolation from the features
y_pred = model.predict(X_test)

#Evaluate the performace
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("="*50)
print("TEST SET PERFORMANCE")
print("="*50)

print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

#Cross validation analysis
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

print("\n" + "="*50)
print("CROSS VALIDATION")
print("="*50)

print(f"Mean R² : {cv_scores.mean():.4f}")
print(f"Std R²  : {cv_scores.std():.4f}")

#Permutation importance analysis
perm = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

importance = pd.DataFrame({'Feature': features, 'Importance': perm.importances_mean}).sort_values( 'Importance', ascending=False)

print("\n" + "="*50)
print("PERMUTATION IMPORTANCE")
print("="*50)

print(importance)

#Create a table showing the different ML models and their accuracy through R sqared
#Create the different models
models = {'Linear': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
    'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.01))]),
    'Random Forest': RandomForestRegressor(n_estimators=500, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(n_estimators=500, random_state=42)
}

#Assess each models acurracy
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    results.append({'Model': name, 'Mean_R2': scores.mean(), 'Std_R2': scores.std()})

#collect the results and output to the user
results = pd.DataFrame(results)
results = results.sort_values('Mean_R2')

print(results)

#Create a plot grpah on the difference between the predicted and actual soical isolation values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

plt.xlabel("Actual Isolation")
plt.ylabel("Predicted Isolation")
plt.title("Extra Trees: Actual vs Predicted")

plt.show()

#Create a bar chart on each features correlation/importance to soical isolation
importance.plot(x='Feature', y='Importance', kind='bar', legend=False, figsize=(10,6))

plt.ylabel("Permutation Importance")
plt.title("Feature Importance for Predicting Social Isolation")
plt.tight_layout()
plt.show()
