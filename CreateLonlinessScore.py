#import used liberies
import pandas as pd

#Create a overall lonliness score from lonely, isolate and connect
final_df["loneliness_score"] = (
    final_df["lonely"] +
    final_df["isolate"] +
    (10 - final_df["connect"])
) / 3

final_df.to_csv("/Users/scottspreadborough/Downloads/final_with_loneliness_score.csv", index=False)
