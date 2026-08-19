#Bring in the dataset and create an array for the output
DATASET_PATH = "/Users/scottspreadborough/Downloads/Loneliness_Dataset_Nov10"
participants_info = []

# Check each participants files for Oura, EMA and Phone data
for participant in os.listdir(DATASET_PATH):
    p_path = os.path.join(DATASET_PATH, participant)
    if not os.path.isdir(p_path):
        continue

    has_sleep_timing = False
    has_ema = False
    has_phone = False

    #Check Oura Ring Data is available
    try:
        oura_path = os.path.join(p_path, "Oura")
        oura_files = [f for f in os.listdir(oura_path) if f.endswith(".csv")]
        if len(oura_files) > 0:
            oura_df = pd.read_csv(os.path.join(oura_path, oura_files[0]))
            if "OURA_bedtime_start" in oura_df.columns and "OURA_bedtime_end" in oura_df.columns:
                if not oura_df["OURA_bedtime_start"].isna().all() and not oura_df["OURA_bedtime_end"].isna().all():
                    has_sleep_timing = True
    except:
        pass

    #Check EMA survey data is available
    try:
        ema_path = os.path.join(p_path, "Surveys")
        ema_files = [f for f in os.listdir(ema_path) if "ema_data" in f.lower()]
        if len(ema_files) > 0:
            ema_df = pd.read_csv(os.path.join(ema_path, ema_files[0]))
            required_cols = ["lonely","isolate","connect"]
            if all(c in ema_df.columns for c in required_cols):
                has_ema = True
    except:
        pass

    #Check Phone data is available
    try:
        aware_path = os.path.join(p_path, "Aware")
        if os.path.exists(aware_path):
            screen_file = os.path.join(aware_path, "screen.csv")
            if os.path.exists(screen_file):
                screen_df = pd.read_csv(screen_file)
                if not screen_df.empty:
                    has_phone = True
    except:
        pass

    participants_info.append({
        "participant": participant,
        "sleep_timing": has_sleep_timing,
        "ema": has_ema,
        "phone": has_phone
    })

#Collect all the participants into a table
participants = pd.DataFrame(participants_info)

#Collect all partipants that have all the data and put them into a table
Particpants_with_fullData = participants[(participants["sleep_timing"]) &
                            (participants["ema"]) &
                            (participants["phone"])]

#Output the people who dont have all the data to the screen
print("Participants with all needed data:")
print(full_data["participant"].tolist())
