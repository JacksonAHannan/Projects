import pandas as pd

# File paths
file_2020 = "2020_US_County_Level_Presidential_Results copy.csv"
file_2024 = "2024_US_County_Level_Presidential_Results copy.csv"

# Read CSV files
df_2020 = pd.read_csv(file_2020)
df_2024 = pd.read_csv(file_2024)

# Merge on 'county_fips'
df_merged = pd.merge(df_2020, df_2024, on="county_fips", suffixes=("_2020", "_2024"))

# Columns to drop (from both years if present)
cols_to_drop = [
    "votes_gop_2020", "votes_dem_2020", "total_votes_2020", "diff_2020",
    "votes_gop_2024", "votes_dem_2024", "total_votes_2024", "diff_2024"
]
existing_cols_to_drop = [col for col in cols_to_drop if col in df_merged.columns]
df_cleaned = df_merged.drop(columns=existing_cols_to_drop)

# Calculate 'swing' column
# swing = per_point_diff_2024 - per_point_diff_2020 - 0.06039777585
if 'per_point_diff_2024' in df_cleaned.columns and 'per_point_diff_2020' in df_cleaned.columns:
    df_cleaned['swing'] = df_cleaned['per_point_diff_2024'] - df_cleaned['per_point_diff_2020'] - 0.06039777585
else:
    print("Warning: per_point_diff columns not found. 'swing' column not calculated.")

# Drop additional columns as requested
final_drop_cols = [
    "county_fips", "per_gop_2020", "per_dem_2020", "per_point_diff_2020",
    "state_name_2024", "county_name_2024", "per_gop_2024", "per_dem_2024", "per_point_diff_2024"
]
final_cols_to_drop = [col for col in final_drop_cols if col in df_cleaned.columns]
df_final = df_cleaned.drop(columns=final_cols_to_drop)

# Save to new CSV
df_final.to_csv("US_County_Level_Swing.csv", index=False)

print("Final cleaned data with specified columns dropped saved to US_County_Level_Swing.csv")
