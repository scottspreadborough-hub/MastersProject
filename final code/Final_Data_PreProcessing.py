#Import needed Libaries
import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

#Create the variables for the dataset and the output files
DATASET_PATH = "/Users/scottspreadborough/Downloads/Loneliness_Dataset_Nov10"
EMA_DATA_PATH = "/Users/scottspreadborough/Downloads/Loneliness_Dataset_Nov10/ema_data/*.csv"
OUTPUT_FILE = "/Users/scottspreadborough/Downloads/Final_with_no_missing_data.csv"

#Variables that contain a subset of columns for analysis or removal
SLEEP_COLUMNS_TO_REMOVE = [
    "rolling_sleep_3d", "rolling_sleep_7d", "hrv_3d_avg",
    "hrv_7d_avg", "sleep_efficiency_3d_avg"
]

SURVEY_COLUMNS = [
    "lonely", "connect", "isolate"
]

FINAL_COLUMNS = [
    "participant_id", "date", "sleep_duration",
    "sleep_efficiency", "sleep_score", "deep_sleep",
    "rem_sleep", "awake_time", "latency",
    "hr_avg", "hrv", "temp_delta",
    "sleep_debt", "rolling_sleep_3d", "rolling_sleep_7d",
    "hrv_3d_avg", "hrv_7d_avg", "sleep_efficiency_3d_avg",
    "steps", "activity_score", "sedentary_time",
    "lonely", "connect", "isolate"
]

#Array for all participants data to be put into
all_participants = []

#Function to work out sleep debt
def compute_sleep_debt(duration, baseline=420):
    return duration - baseline

#Function to work out average of sleep and HRV (Heart Rate varibality) over 3 or 7 days and add it to the final table
def add_rolling_features(df):
    df = df.sort_values("date")
    df["rolling_sleep_3d"] = df["sleep_duration"].rolling(3).mean()
    df["rolling_sleep_7d"] = df["sleep_duration"].rolling(7).mean()
    df["hrv_3d_avg"] = df["hrv"].rolling(3).mean()
    df["hrv_7d_avg"] = df["hrv"].rolling(7).mean()
    df["sleep_efficiency_3d_avg"] = df["sleep_efficiency"].rolling(3).mean()
    return df

#Function to covert date from a number into datetime format
def safe_to_date_inclusive(df, col, unit=None):
    if col not in df.columns:
        return pd.Series([pd.NaT] * len(df))

    if unit:
        dates = pd.to_datetime(df[col], unit=unit, errors="coerce")
    else:
        dates = pd.to_datetime(df[col], errors="coerce")
    return dates

#Function to predict the values using random forrest
def impute_with_model(final_df, target_col):
    train_df = final_df[final_df[target_col].notnull()]

    test_df = final_df[final_df[target_col].isnull()]

    if test_df.shape[0] == 0:
        print(f"{target_col}: has no missing values")
        return final_df

    X_train = train_df.drop(columns= [target_col, "participant_id","date"], errors="ignore")

    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col, "participant_id", "date"], errors="ignore")

    X_train = X_train.select_dtypes(include=[np.number])

    X_test = X_test[X_train.columns]

    X_train = X_train.fillna(X_train.mean())

    X_test = X_test.fillna(X_train.mean())

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    final_df.loc[final_df[target_col].isnull(), target_col] = model.predict(X_test)
    return final_df

#Import the data from the dataset
ema_files = glob.glob(EMA_DATA_PATH)

if len(ema_files) == 0:
    raise ValueError("No files found. Check your EMA_DATA_PATH.")

#Create an array for each participants data to go into
df_list = []

#fixing any partipant_id errors in EMA surveys
for file in ema_files:
    temp = pd.read_csv(file)

    participant_id = os.path.basename(file).split(".")[0]

    participant_id = participant_id.replace("ema_data_", "")

    temp["participant_id"] = participant_id

    df_list.append(temp)

ema_df = pd.concat(df_list, ignore_index=True)

ema_df["date"] = pd.to_datetime(ema_df["date"], errors="coerce")

ema_df = ema_df.dropna(subset=["date"])

#Make all the data the same format and style
ema_df["date"] = ema_df["date"].dt.normalize()

#Merge by participant and date
daily_avg = (ema_df.groupby(["participant_id", "date"], as_index=False).agg({"lonely": lambda x: x.dropna().mean(),"connect": lambda x: x.dropna().mean(),"isolate": lambda x: x.dropna().mean()})
)

#Create a daily average removing any duplicates or missing data
daily_avg = daily_avg.sort_values(["participant_id", "date"])

for participant in os.listdir(DATASET_PATH):
    p_path = os.path.join(DATASET_PATH, participant)

    if not os.path.isdir(p_path):
        continue

    try:
        #Oura ring processing
        oura_path = os.path.join(p_path, "Oura")

        oura_files = [f for f in os.listdir(oura_path) if f.endswith(".csv")]

        if len(oura_files) == 0:
            print(f"Error: No Oura file found for {participant}")
            continue

        oura_df = pd.read_csv(os.path.join(oura_path, oura_files[0]))

        sleep_df = pd.DataFrame({
            "date": safe_to_date_inclusive(oura_df,"timestamp",unit="ms"),
            "sleep_duration": oura_df.get("OURA_sleep_duration", np.nan),
            "sleep_efficiency": oura_df.get("OURA_sleep_efficiency", np.nan),
            "sleep_score": oura_df.get("OURA_sleep_score_total", np.nan),
            "deep_sleep": oura_df.get("OURA_sleep_deep", np.nan),
            "rem_sleep": oura_df.get("OURA_sleep_rem", np.nan),
            "awake_time": oura_df.get("OURA_sleep_awake", np.nan),
            "latency": oura_df.get("OURA_sleep_onset_latency", np.nan),
            "hr_avg": oura_df.get("OURA_sleep_hr_average", np.nan),
            "hrv": oura_df.get("OURA_sleep_rmssd", np.nan),
            "temp_delta": oura_df.get("OURA_sleep_temperature_delta", np.nan),
            "steps": oura_df.get("OURA_activity_steps", np.nan),
            "activity_score": oura_df.get("OURA_activity_score", np.nan),
            "sedentary_time": oura_df.get("OURA_activity_inactive", np.nan)
        })

        #Phone data processing
        phone_daily = pd.DataFrame()

        #Add the EMA serveys and Phone data by date to the table
        ema_sub = daily_avg[daily_avg["participant_id"] == participant].copy()

        sleep_df["participant_id"] = participant

        sleep_df["date"] = pd.to_datetime(sleep_df["date"], errors="coerce")

        ema_sub["date"] = pd.to_datetime(ema_sub["date"], errors="coerce")

        sleep_df = sleep_df.dropna(subset=["date"])

        ema_sub = ema_sub.dropna(subset=["date"])

        if ema_sub.empty:
            sleep_df[["lonely", "connect", "isolate"]] = pd.NA
            df = sleep_df.copy()
                            
        else:
            sleep_df = sleep_df.sort_values("date")
            ema_sub = ema_sub.sort_values("date")
            merged_sub = pd.merge_asof(sleep_df,ema_sub[["date", "lonely", "connect", "isolate"]], on="date", direction="nearest")
            df = merged_sub

        #Remove unwanted data
        df["sleep_debt"] = (df["sleep_duration"].apply(compute_sleep_debt))

        df = add_rolling_features(df)

        df["participant_id"] = participant

        for col in SURVEY_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        df = df[FINAL_COLUMNS]

        all_participants.append(df)

    except Exception as e:
        print(f"Error: processing {participant}: {e}")


# Saving the final dataset
if all_participants:
    final_df = pd.concat(all_participants, ignore_index=True)
else:
    raise ValueError("Error: All particpant data failed to load or does not exist")

# Columns to remove
final_df = final_df.drop(columns=[col for col in SLEEP_COLUMNS_TO_REMOVE if col in final_df.columns])

# Ensure survey columns exist
for col in SURVEY_COLUMNS:
    if col not in final_df.columns:
        final_df[col] = 0

#Fill missing values with 0.0 (null)
final_df[["lonely", "connect", "isolate"]] = final_df[["lonely", "connect", "isolate"]].fillna(0.0)

# Columns to check for 70% completeness (exclude participant_id, date, and newly computed score) and keep only rows with >=70% of the data present
data_cols = [col for col in final_df.columns if col not in ["participant_id", "date", "loneliness_score"]]

filtered_rows_df = final_df[final_df[data_cols].notna().mean(axis=1) >= 0.7]

#Create a overall lonliness score from lonely, isolate and connect
final_df = filtered_rows_df.copy()

final_df["loneliness_score"] = (final_df["lonely"] + final_df["isolate"] + (10 - final_df["connect"])) / 3

#Use the random forrest model predictions to fill in the missing values in the sleep_score and temp_delta coulmns
for col in ["sleep_score", "temp_delta"]:
    if col in final_df.columns:
        final_df = impute_with_model(final_df,col)

#Use the mean of the coulmn to fill in the missing values in the hr_avg and hrv columns
for col in ["hr_avg", "hrv", "steps", "activity_score", "sedentary_time"]:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(final_df[col].mean())

#Remove activity columns after they have been used for random forrest predictions
final_df = final_df.drop(columns=["steps", "activity_score", "sedentary_time"], errors="ignore")

final_df.to_csv(OUTPUT_FILE, index=False)

print(f"Final dataset saved to: {OUTPUT_FILE}")
