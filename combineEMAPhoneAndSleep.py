#Import needed Libaries
import pandas as pd

# Load files
sleep_df = pd.read_csv("/Users/scottspreadborough/Downloads/sleep_social_isolation_all_included.csv")
ema_df = pd.read_csv("/Users/scottspreadborough/Downloads/Loneliness_Dataset_Nov10/daily_ema_all_participants.csv")

merged_list = []

#Make sure all column names are the same across participants
sleep_df.columns = sleep_df.columns.str.lower()
ema_df.columns = ema_df.columns.str.lower()

#Fix participant_id in EMA
ema_df['participant_id'] = ema_df['participant_id'].str.replace('ema_data_', '', regex=False)

#Convert dates to correct format
sleep_df['date'] = pd.to_datetime(sleep_df['date'], errors='coerce')
ema_df['date'] = pd.to_datetime(ema_df['date'], errors='coerce')

#Drop missing rows
sleep_df = sleep_df.dropna(subset=['participant_id', 'date'])
ema_df = ema_df.dropna(subset=['participant_id', 'date'])

for pid in sleep_df['participant_id'].unique():
    
    sleep_sub = sleep_df[sleep_df['participant_id'] == pid].copy()
    ema_sub = ema_df[ema_df['participant_id'] == pid].copy()
    
    #Skip EMA null rows
    if ema_sub.empty:
        sleep_sub[['lonely', 'connect', 'isolate']] = pd.NA
        merged_list.append(sleep_sub)
        continue
    
    #Sort by date
    sleep_sub = sleep_sub.sort_values('date')
    ema_sub = ema_sub.sort_values('date')
    
    #Merge participants survey with nearest date technology data
    merged_sub = pd.merge_asof(
        sleep_sub,
        ema_sub[['date', 'lonely', 'connect', 'isolate']],
        on='date',
        direction='nearest'
    )
    
    merged_list.append(merged_sub)

#Merge all participants back together dropping unneeded columns
final_df = pd.concat(merged_list).sort_values(['participant_id', 'date'])
final_df = final_df.drop(columns=['lonely_x', 'connect_x', 'isolate_x'], errors='ignore')

final_df = final_df.rename(columns={
    'lonely_y': 'lonely',
    'connect_y': 'connect',
    'isolate_y': 'isolate'
})

#Fill missing values with 0.0 (null)
final_df[['lonely', 'connect', 'isolate']] = final_df[['lonely', 'connect', 'isolate']].fillna(0.0)

final_df.to_csv("/Users/scottspreadborough/Downloads/filled_sleep_social_isolation_all_included.csv", index=False)
