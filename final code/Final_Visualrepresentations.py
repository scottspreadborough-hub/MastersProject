#Import used liberies
import seaborn as sns
from sklearn.inspection import permutation_importance

#Create a bar graph of the R sqaared values of each ML model
plt.figure(figsize=(10,6))

plt.barh(results['Model'], results['Mean_R2'])

plt.xlabel('Cross-Validated R²')
plt.ylabel('Model')
plt.title('Model Performance Comparison')

for i, v in enumerate(results['Mean_R2']):
    plt.text(v + 0.002, i, f'{v:.3f}')

plt.tight_layout()
plt.show()
       
#Create a bar graph of the R sqaared values of each ML model with standard deviation
plt.figure(figsize=(10,6))

plt.barh(results['Model'], results['Mean_R2'], xerr=results['Std_R2'], capsize=5)

plt.xlabel('Cross-Validated R²')
plt.ylabel('Model')
plt.title('Model Performance (Mean ± SD)')

for i, v in enumerate(results['Mean_R2']):
    plt.text(v + 0.002, i, f'{v:.3f}')

plt.tight_layout()
plt.show()

#Create a plot graph of the accual and predicted values
plt.figure(figsize=(8,8))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

plt.xlabel('Actual Isolation')
plt.ylabel('Predicted Isolation')
plt.title('Extra Trees: Actual vs Predicted')

plt.show()

#Create a bar graph showing the permutation importance of each feature
perm = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
importance = pd.DataFrame({'Feature': features, 'Importance': perm.importances_mean})
importance = importance.sort_values('Importance', ascending=True)

plt.figure(figsize=(10,6))

plt.barh(importance['Feature'], importance['Importance'])

plt.xlabel('Permutation Importance')
plt.title('Importance of Sleep Variables')

plt.show()

#Create a plot grpah of redudules of the distance between the predicted and actual values
residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(y_pred, residuals, alpha=0.6)

plt.axhline(y=0, linestyle='--')

plt.xlabel('Predicted Isolation')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.show()

#Create a bar chart showing the predictive error distrubution
errors = y_test - y_pred

plt.figure(figsize=(8,6))

plt.hist(errors, bins=20)

plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')

plt.show()
        
#create a line graph showing the cross validation performace (R squared)
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

plt.figure(figsize=(8,6))

plt.plot(range(1,6), scores, marker='o')

plt.xticks(range(1,6))
plt.xlabel('Fold')
plt.ylabel('R²')
plt.title('Cross-Validation Performance')

plt.show()

#create a heat map graph showing the correlation between the features
corr = df[features + ['isolate']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Selected Sleep Features and Isolation')
plt.show()
