import pandas as pd

# File paths
swing_file = "US_County_Level_Swing.csv"
pop_file = "cc-est2024-alldata_cleaned copy 2.csv"

# Read CSV files
df_swing = pd.read_csv(swing_file)
df_pop = pd.read_csv(pop_file)

# Merge on state and county name columns
df_merged = pd.merge(
    df_swing,
    df_pop,
    left_on=["state_name_2020", "county_name_2020"],
    right_on=["STNAME", "CTYNAME"],
    how="inner"
)

# Add 'Southern' boolean column
southern_states = [
    "Alabama", "Georgia", "Tennessee", "Florida", "South Carolina", "North Carolina",
    "Mississippi", "Louisiana", "Texas", "Oklahoma", "Arkansas", "Missouri"
]
df_merged['Southern'] = df_merged['state_name_2020'].isin(southern_states).astype(int)

# Save merged result with 'Southern' as the last column
cols = list(df_merged.columns)
if cols[-1] != 'Southern':
    cols = [col for col in cols if col != 'Southern'] + ['Southern']
df_merged = df_merged[cols]
df_merged.to_csv("US_County_Level_Swing_with_Pop.csv", index=False)

print("Joined data with 'Southern' column saved to US_County_Level_Swing_with_Pop.csv")
