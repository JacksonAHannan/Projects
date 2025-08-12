import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data
csv_path = "/Users/jacksonhannan/Desktop/Python Projects/LA Crime Data/Crime_Data_from_2020_to_Present.csv"
df = pd.read_csv(csv_path)

# Preview the data
print(df.head())

# Plot total crime for every distinct year
if 'DATE OCC' in df.columns:
    # Convert 'DATE OCC' to datetime
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
    df['Year'] = df['DATE OCC'].dt.year
    # Exclude 2025 (partial year) from analysis
    df = df[df['Year'] != 2025]
    crimes_per_year = df['Year'].value_counts().sort_index()
    plt.figure(figsize=(8,5))
    sns.barplot(x=crimes_per_year.index.astype(int), y=crimes_per_year.values, palette="viridis")
    plt.xlabel("Year")
    plt.ylabel("Number of Crimes")
    plt.title("Total Crimes per Year in LA (2020-Present, Excluding 2025)")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'DATE OCC' not found in the CSV.")

# Define violent and non-violent crime keyword patterns (expanded)
violent_keywords = [
    "ASSAULT", "BATTERY", "ROBBERY", "HOMICIDE", "MURDER", "RAPE", "SEXUAL", "KIDNAPPING",
    "ARSON", "MANSLAUGHTER", "WEAPON", "SHOOTING", "STABBING", "CARJACKING"
]
non_violent_keywords = [
    "BURGLARY", "THEFT", "LARCENY", "SHOPLIFT", "VANDALISM", "TRESPASS", "FRAUD",
    "EMBEZZLEMENT", "FORGERY", "DRUNK", "DUI", "NARCOTIC", "DRUG", "STOLEN VEHICLE",
    "IDENTITY THEFT", "EXTORTION", "BRIBERY", "COUNTERFEIT", "TAMPERING"
]

# Categorize crimes using keyword presence
def categorize_crime(crime_desc: str) -> str:
    if not isinstance(crime_desc, str):
        return "Other"
    text = crime_desc.upper()
    if any(k in text for k in violent_keywords):
        return "Violent"
    if any(k in text for k in non_violent_keywords):
        return "Non-Violent"
    return "Other"

# Ensure Year column exists before proceeding
if 'Crm Cd Desc' in df.columns and 'Year' in df.columns:
    df['Crime Category'] = df['Crm Cd Desc'].apply(categorize_crime)

    # Build summary counts by Year + Category
    summary = (
        df.groupby(['Year', 'Crime Category'])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )

    # Guarantee both columns exist even if zero
    for col in ['Violent', 'Non-Violent']:
        if col not in summary.columns:
            summary[col] = 0
    summary = summary[['Violent', 'Non-Violent'] + [c for c in summary.columns if c not in ['Violent', 'Non-Violent']]]

    # 1. Stacked bar chart (Violent vs Non-Violent) — exclude 'Other' from stack unless you want it
    stacked_df = summary[['Violent', 'Non-Violent']].copy()
    plt.figure(figsize=(10, 6))
    stacked_df.plot(kind='bar', stacked=True, color=['#d73027', '#4575b4'])
    plt.xlabel('Year')
    plt.ylabel('Number of Crimes')
    plt.title('Violent vs Non-Violent Crimes in LA (Stacked)')
    plt.legend(title='Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Separate bar chart: Violent crimes
    plt.figure(figsize=(8, 5))
    sns.barplot(x=stacked_df.index.astype(int), y=stacked_df['Violent'].values, color='#d73027')
    plt.xlabel('Year')
    plt.ylabel('Violent Crime Count')
    plt.title('Violent Crimes per Year in LA')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Separate bar chart: Non-Violent crimes
    plt.figure(figsize=(8, 5))
    sns.barplot(x=stacked_df.index.astype(int), y=stacked_df['Non-Violent'].values, color='#4575b4')
    plt.xlabel('Year')
    plt.ylabel('Non-Violent Crime Count')
    plt.title('Non-Violent Crimes per Year in LA')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Optional: If you want to also see 'Other', uncomment below
    # if 'Other' in summary.columns:
    #     plt.figure(figsize=(8,5))
    #     sns.barplot(x=summary.index.astype(int), y=summary['Other'].values, color='#aaaaaa')
    #     plt.xlabel('Year')
    #     plt.ylabel('Other Crime Count')
    #     plt.title('Other Classified Crimes per Year in LA')
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()
else:
    print("Required columns 'Crm Cd Desc' and/or 'Year' are missing for categorized visualizations.")

# Output directory for figures
output_dir = "/Users/jacksonhannan/Desktop/Python Projects/LA Crime Data/figures"
os.makedirs(output_dir, exist_ok=True)

# Approximate Los Angeles city population by year (can refine with official data)
population_by_year = {
    2020: 3979576,
    2021: 3985516,
    2022: 3990456,
    2023: 3998783,
    2024: 4002154,
    2025: 4005000
}

# Helper to save current figure
def save_current_fig(filename: str):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    print(f"Saved figure: {path}")

# After creating stacked_df (violent/non-violent counts per year) we add rates and smoothing
if 'Crm Cd Desc' in df.columns and 'Year' in df.columns and 'Crime Category' in df.columns:
    # Compute rates per 100k
    counts_df = stacked_df.copy()
    rates_df = counts_df.div(counts_df.index.map(population_by_year), axis=0) * 100000

    # Ensure proper mapping of population data and alignment in rates_df calculation
    if 'Crm Cd Desc' in df.columns and 'Year' in df.columns and 'Crime Category' in df.columns:
        # Filter out years not in population_by_year to avoid mismatches
        valid_years = set(population_by_year.keys())
        counts_df = stacked_df[stacked_df.index.isin(valid_years)].copy()

        # Compute rates per 100k using valid years
        rates_df = counts_df.div(counts_df.index.map(population_by_year), axis=0) * 100000

        # (A) Stacked absolute counts (already plotted above) -> replot to save
        plt.figure(figsize=(10, 6))
        counts_df.plot(kind='bar', stacked=True, color=['#d73027', '#4575b4'])
        plt.xlabel('Year')
        plt.ylabel('Number of Crimes')
        plt.title('Violent vs Non-Violent Crimes in LA (Counts)')
        plt.legend(title='Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_current_fig('stacked_counts.png')
        plt.close()

        # (B) Stacked rates per 100k
        plt.figure(figsize=(10, 6))
        rates_df.plot(kind='bar', stacked=True, color=['#d73027', '#4575b4'])
        plt.xlabel('Year')
        plt.ylabel('Crimes per 100k population')
        plt.title('Violent vs Non-Violent Crimes in LA (Rates per 100k)')
        plt.legend(title='Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_current_fig('stacked_rates_per_100k.png')
        plt.close()

        # (C) Separate violent counts
        plt.figure(figsize=(8, 5))
        sns.barplot(x=counts_df.index.astype(int), y=counts_df['Violent'].values, color='#d73027')
        plt.xlabel('Year')
        plt.ylabel('Violent Crime Count')
        plt.title('Violent Crimes per Year (Counts)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_current_fig('violent_counts.png')
        plt.close()

        # (D) Separate non-violent counts
        plt.figure(figsize=(8, 5))
        sns.barplot(x=counts_df.index.astype(int), y=counts_df['Non-Violent'].values, color='#4575b4')
        plt.xlabel('Year')
        plt.ylabel('Non-Violent Crime Count')
        plt.title('Non-Violent Crimes per Year (Counts)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_current_fig('non_violent_counts.png')
        plt.close()

        # (E) Separate violent rates
        plt.figure(figsize=(8, 5))
        sns.barplot(x=rates_df.index.astype(int), y=rates_df['Violent'].values, color='#d73027')
        plt.xlabel('Year')
        plt.ylabel('Violent Crimes per 100k')
        plt.title('Violent Crimes per Year (Rate per 100k)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_current_fig('violent_rates_per_100k.png')
        plt.close()

        # (F) Separate non-violent rates
        plt.figure(figsize=(8, 5))
        sns.barplot(x=rates_df.index.astype(int), y=rates_df['Non-Violent'].values, color='#4575b4')
        plt.xlabel('Year')
        plt.ylabel('Non-Violent Crimes per 100k')
        plt.title('Non-Violent Crimes per Year (Rate per 100k)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_current_fig('non_violent_rates_per_100k.png')
        plt.close()

        # (G) Monthly smoothing (3-month rolling average) for violent & non-violent
        if 'DATE OCC' in df.columns:
            monthly = (
                df.set_index('DATE OCC')
                  .groupby([pd.Grouper(freq='M'), 'Crime Category'])
                  .size()
                  .unstack(fill_value=0)
            )
            # Ensure columns exist
            for col in ['Violent', 'Non-Violent']:
                if col not in monthly.columns:
                    monthly[col] = 0
            monthly = monthly[['Violent', 'Non-Violent']]
            rolling = monthly.rolling(window=3, min_periods=1).mean()

            plt.figure(figsize=(11, 6))
            plt.plot(monthly.index, monthly['Violent'], color='#d73027', alpha=0.4, label='Violent (Monthly)')
            plt.plot(rolling.index, rolling['Violent'], color='#d73027', linewidth=2, label='Violent (3-mo MA)')
            plt.plot(monthly.index, monthly['Non-Violent'], color='#4575b4', alpha=0.4, label='Non-Violent (Monthly)')
            plt.plot(rolling.index, rolling['Non-Violent'], color='#4575b4', linewidth=2, label='Non-Violent (3-mo MA)')
            plt.xlabel('Month')
            plt.ylabel('Number of Crimes')
            plt.title('Monthly Violent vs Non-Violent Crimes with 3-Month Moving Average')
            plt.legend()
            plt.tight_layout()
            save_current_fig('monthly_counts_with_smoothing.png')
            plt.close()

            # Monthly rates per 100k (use population of the year the month belongs to)
            monthly_rates = monthly.copy()
            monthly_rates['Year'] = monthly_rates.index.year
            monthly_rates = monthly_rates.div(monthly_rates['Year'].map(population_by_year), axis=0) * 100000
            monthly_rates = monthly_rates.drop(columns=['Year'])
            rolling_rates = monthly_rates.rolling(window=3, min_periods=1).mean()

            plt.figure(figsize=(11, 6))
            plt.plot(monthly_rates.index, monthly_rates['Violent'], color='#d73027', alpha=0.35, label='Violent Rate (Monthly)')
            plt.plot(rolling_rates.index, rolling_rates['Violent'], color='#d73027', linewidth=2, label='Violent Rate (3-mo MA)')
            plt.plot(monthly_rates.index, monthly_rates['Non-Violent'], color='#4575b4', alpha=0.35, label='Non-Violent Rate (Monthly)')
            plt.plot(rolling_rates.index, rolling_rates['Non-Violent'], color='#4575b4', linewidth=2, label='Non-Violent Rate (3-mo MA)')
            plt.xlabel('Month')
            plt.ylabel('Crimes per 100k')
            plt.title('Monthly Violent vs Non-Violent Crime Rates (per 100k) with 3-Month Moving Average')
            plt.legend()
            plt.tight_layout()
            save_current_fig('monthly_rates_with_smoothing.png')
            plt.close()
        else:
            print("Column 'DATE OCC' missing; monthly smoothing skipped.")
else:
    print("Required columns 'Crm Cd Desc', 'Year', and/or 'Crime Category' are missing for rate and smoothing visualizations.")
