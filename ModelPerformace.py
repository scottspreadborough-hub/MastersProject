#Import used liberies
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

results = []

#Create the train, test split
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

#Create the different models
models = {
    'Linear': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),

    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ]),

    'Lasso': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.01))
    ]),

    'Random Forest': RandomForestRegressor(
        n_estimators=500,
        random_state=42
    ),

    'Gradient Boosting': GradientBoostingRegressor(
        random_state=42
    ),

    'Extra Trees': ExtraTreesRegressor(
        n_estimators=500,
        random_state=42
    )
}

#Assess each models acurracy
for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring='r2'
    )

    results.append({
        'Model': name,
        'Mean_R2': scores.mean(),
        'Std_R2': scores.std()
    })

#collect the results and output to the user
results = pd.DataFrame(results)
results = results.sort_values('Mean_R2')

print(results)
