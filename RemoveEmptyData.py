#Import used libaries
from sklearn.ensemble import RandomForestRegressor

#Function to predict the values using random forrest
def impute_with_model(final_df, target_col):
    train_df = final_df[final_df[target_col].notnull()]
    test_df = final_df[final_df[target_col].isnull()]

    if test_df.shape[0] == 0:
        print(f"{target_col}: has no missing values")
        return final_df
    
    X_train = train_df.drop(columns=[target_col, 'participant_id', 'date'], errors='ignore')
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=[target_col, 'participant_id', 'date'], errors='ignore')
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    final_df.loc[final_df[target_col].isnull(), target_col] = model.predict(X_test)
    
    return final_df


#Use the random forrest model predictions to fill in the missing values in the sleep_score and temp_delta coulmns
for col in ['sleep_score', 'temp_delta']:
    if col in final_df.columns:
        final_df = impute_with_model(final_df, col)


#Use the mean of the coulmn to fill in the missing values in the hr_avg and hrv columns
for col in ['hr_avg', 'hrv']:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(final_df[col].mean())

final_df.to_csv("/Users/scottspreadborough/Downloads/final_with_no_missing_data.csv", index=False)
