#Import needed Libaries
import pandas as pd
import glob
import os

#Import the data from the dataset
DATA_PATH = "/Users/scottspreadborough/Downloads/Loneliness_Dataset_Nov10/ema_data/*.csv"
files = glob.glob(DATA_PATH)

if len(files) == 0:
    raise ValueError("No files found. Check your DATA_PATH.")

#Create an array for each participants data to go into
df_list = []

#
for file in files:
    temp = pd.read_csv(file)
    participant_id = os.path.basename(file).split('.')[0]
    temp['participant_id'] = participant_id
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

#Make all the data the same format and style
df['date'] = df['date'].dt.normalize()

#Merge by participant and date
daily_avg = (
    df.groupby(['participant_id', 'date'], as_index=False)
    .agg({
        'lonely': lambda x: x.dropna().mean(),
        'connect': lambda x: x.dropna().mean(),
        'isolate': lambda x: x.dropna().mean()
    })
)

#Create a daily average removing any duplicates or missing data
daily_avg = daily_avg.sort_values(['participant_id', 'date'])

# Save
daily_avg.to_csv("/Users/scottspreadborough/Downloads/daily_emaAverage_all_participants.csv", index=False)
