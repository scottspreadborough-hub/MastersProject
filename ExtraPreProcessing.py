#Importing python libaries
import pandas as pd

df = pd.read_csv("ema_data.csv")

#Changing the timestamp from a number to datatime datatype
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

#Group and compute means (NaNs automatically ignored per column)
daily_avg = (
    df.groupby(['participant_id', 'date'])
    .agg({
        'lonely': lambda x: x.dropna().mean(),
        'connect': lambda x: x.dropna().mean(),
        'isolate': lambda x: x.dropna().mean()
    })
    .reset_index()
)

print(daily_avg.head())

# Columns to remove
cols_to_remove = [
    "rolling_sleep_3d", "rolling_sleep_7d", "hrv_3d_avg", "hrv_7d_avg",
    "sleep_efficiency_3d_avg", "steps", "activity_score", "sedentary_time",
    "screen_time", "unlock_count", "night_phone_usage"
]

# Drop unwanted columns if they exist
final_df = final_df.drop(columns=[c for c in cols_to_remove if c in final_df.columns])

# Ensure survey columns exist
for col in ["lonely", "isolate", "connect"]:
    if col not in final_df.columns:
        final_df[col] = 0  # Fill missing survey columns with 0

# Compute loneliness score
final_df["loneliness_score"] = (final_df["lonely"] + final_df["isolate"] - final_df["connect"]) / 3

# Columns to check for 70% completeness (exclude participant_id, date, and newly computed score) and keep only rows with >=70% of the data present
data_cols = [col for col in final_df.columns if col not in ["participant_id", "date", "loneliness_score"]]

filtered_rows_df = final_df[final_df[data_cols].notna().mean(axis=1) >= 0.7]

# Save file
filtered_rows_df.to_csv(r"/Users/scottspreadborough/Downloads/sleep_social_isolation_all_included.csv", index=False)

print(f"Saved filtered dataset with {len(filtered_rows_df)} rows to Downloads.")
