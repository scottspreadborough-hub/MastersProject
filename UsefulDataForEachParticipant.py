#Import needed Libaries
import os
import pandas as pd
import numpy as np

#Create the variables for the dataset and the output files
DATASET_PATH = "/Users/scottspreadborough/Downloads/Loneliness_Dataset_Nov10"
OUTPUT_FILE = "sleep_social_isolation_all_included.csv"

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
        return pd.Series([pd.NaT]*len(df))
    
    if unit:
        dates = pd.to_datetime(df[col], unit=unit, errors="coerce")  # removed deprecated arg
    else:
        dates = pd.to_datetime(df[col], errors="coerce")

    return dates

for participant in os.listdir(DATASET_PATH):
    p_path = os.path.join(DATASET_PATH, participant)
    if not os.path.isdir(p_path):
        continue

    try:
        #Oura ring processing
        oura_path = os.path.join(p_path, "Oura")
        oura_files = [f for f in os.listdir(oura_path) if f.endswith(".csv")]
        if len(oura_files) == 0:
            print(f"No Oura file for {participant}")
            continue
        
        oura_df = pd.read_csv(os.path.join(oura_path, oura_files[0]))
        sleep_df = pd.DataFrame({
            "date": safe_to_date_inclusive(oura_df, "timestamp", unit="ms"),
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

        #EMA surveys processing
        ema_path = os.path.join(p_path, "Surveys")
        ema_files = [f for f in os.listdir(ema_path) if "ema_data" in f.lower()]
        if len(ema_files) == 0:
            print(f"No EMA file for {participant}")
            continue

        ema_df = pd.read_csv(os.path.join(ema_path, ema_files[0]))
        ema_df["date"] = safe_to_date_inclusive(ema_df, "timestamp")

        required_cols = ["lonely", "isolate", "connect"]
        for col in required_cols:
            if col not in ema_df.columns:
                ema_df[col] = np.nan

        ema_df["loneliness_score"] = (ema_df["lonely"] + ema_df["isolate"] - ema_df["connect"]) / 3

        if ema_df["date"].notna().any():
            ema_daily = ema_df.groupby("date")["loneliness_score"].mean().reset_index()
        else:
            ema_daily = ema_df[["date", "loneliness_score"]].copy()

        #Phone data processing
        aware_path = os.path.join(p_path, "Aware")
        phone_daily = pd.DataFrame()
        if os.path.exists(aware_path):
            screen_file = os.path.join(aware_path, "screen.csv")
            if os.path.exists(screen_file):
                screen_df = pd.read_csv(screen_file)
                screen_df["date"] = safe_to_date_inclusive(screen_df, "timestamp")
                
                if screen_df["date"].notna().any():
                    screen_daily = screen_df.groupby("date").size().reset_index(name="screen_time")
                    unlock_count = screen_daily.copy(); unlock_count.columns = ["date","unlock_count"]
                    screen_df["hour"] = pd.to_datetime(screen_df.get("timestamp", np.nan), errors="coerce").dt.hour
                    night_df = screen_df[(screen_df["hour"] >= 0) & (screen_df["hour"] <= 6)]
                    night_usage = night_df.groupby("date").size().reset_index(name="night_phone_usage")
                    phone_daily = screen_daily.merge(unlock_count, on="date", how="outer")
                    phone_daily = phone_daily.merge(night_usage, on="date", how="outer")
                else:
                    phone_daily = pd.DataFrame(columns=["date","screen_time","unlock_count","night_phone_usage"])

        for col in ["screen_time","unlock_count","night_phone_usage"]:
            if col not in phone_daily.columns:
                phone_daily[col] = np.nan

        #Add the EMA serveys and Phone data by date to the table
        df = sleep_df.merge(ema_daily, on="date", how="left")
        df = df.merge(phone_daily, on="date", how="left")

        #Remove unwanted data
        df["sleep_debt"] = df["sleep_duration"].apply(compute_sleep_debt)
        df = add_rolling_features(df)
        df["participant_id"] = participant

        final_cols = [
            "participant_id","date","sleep_duration","sleep_efficiency","sleep_score",
            "deep_sleep","rem_sleep","awake_time","latency",
            "hr_avg","hrv","temp_delta",
            "sleep_debt","rolling_sleep_3d","rolling_sleep_7d",
            "hrv_3d_avg","hrv_7d_avg","sleep_efficiency_3d_avg",
            "steps","activity_score","sedentary_time",
            "screen_time","unlock_count","night_phone_usage",
            "loneliness_score"
        ]
        df = df[final_cols]

        all_participants.append(df)

    except Exception as e:
        print(f"Error processing {participant}: {e}")

# -----------------------------
# Save final dataset
# -----------------------------
if all_participants:
    final_df = pd.concat(all_participants)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("CSV: ", OUTPUT_FILE, "created")
else:
    print("Error: All particpant data failed")
