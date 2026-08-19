#import used liberies
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor)

#Import the pre-processing dataset into a varibale
df = pd.read_csv("/Users/scottspreadborough/Downloads/Final_with_no_missing_data.csv")

TARGET = "isolate"

#Create a varibale with only the sleep features
sleep_features = [
    'sleep_duration', 'sleep_efficiency', 'sleep_score',
    'deep_sleep', 'rem_sleep', 'awake_time',
    'latency', 'hr_avg', 'hrv',
    'temp_delta', 'sleep_debt'
]

X = df[sleep_features]
y = df[TARGET]

#Person correlation analysis
print("\n" + "="*60)
print("PEARSON CORRELATIONS")
print("="*60)

pearson_corr = X.corrwith(y).sort_values(key=lambda x: np.abs(x), ascending=False)

print(pearson_corr)

#Spearman correlation analysis
print("\n" + "="*60)
print("SPEARMAN CORRELATIONS")
print("="*60)

spearman_corr = X.corrwith(y, method="spearman").sort_values(key=lambda x: np.abs(x), ascending=False)

print(spearman_corr)

#Mutual information correlation analysis
print("\n" + "="*60)
print("MUTUAL INFORMATION")
print("="*60)

mi = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({"Feature": X.columns, "Mutual_Information": mi }).sort_values("Mutual_Information", ascending=False)

print(mi_df)

#RFE feature importance
print("\n" + "="*60)
print("RFE FEATURE RANKING")
print("="*60)

rfe_model = RandomForestRegressor(n_estimators=300, random_state=42)
rfe = RFE(estimator=rfe_model, n_features_to_select=5)
rfe.fit(X, y)

rfe_df = pd.DataFrame({"Feature": X.columns, "Rank": rfe.ranking_, "Selected": rfe.support_ }).sort_values("Rank")

print(rfe_df)

#Random forrest feature importance
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X, y)

importance_df = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_}).sort_values( "Importance", ascending=False)

print("\n" + "="*60)
print("RANDOM FOREST IMPORTANCE")
print("="*60)

print(importance_df)

#Models to be tested
models = {"Linear": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
    "Lasso": Pipeline([("scaler", StandardScaler()),("model", Lasso(alpha=0.01))]),
    "RandomForest": RandomForestRegressor(n_estimators=500, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

#Model comparison alanysis
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

model_results = []

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    mean_score = scores.mean()
    
    model_results.append({"Model": name, "Mean_R2": mean_score, "Std_R2": scores.std()})

    print(f"{name:<20}" f"Mean R²: {mean_score:.4f}" f"  Std: {scores.std():.4f}")

#Finding the highest correlated features to social isolation
print("\n" + "="*60)
print("SEARCHING BEST FEATURE SUBSETS")
print("="*60)

feature_names = list(X.columns)

best_score = -999
best_features = None
best_model = None

candidate_models = { "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

for model_name, model in candidate_models.items():
    for k in range(2, len(feature_names) + 1):
        for subset in combinations(feature_names, k):
            subset_X = X[list(subset)]

            scores = cross_val_score(model, subset_X, y, cv=cv, scoring="r2")

            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_features = subset
                best_model = model_name

print("\n" + "="*60)
print("BEST MODEL + FEATURE SET")
print("="*60)

print(f"Best Model: {best_model}")
print(f"Best Cross-Validated R²: {best_score:.4f}")

print("\nSelected Features:")

for feature in best_features:
    print("-", feature)

#Create and train the final model
final_model = candidate_models[best_model]

final_model.fit(X[list(best_features)], y)
