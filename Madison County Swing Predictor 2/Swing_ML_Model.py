import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
csv_file = "US_County_Level_Swing_with_Pop.csv"
df = pd.read_csv(csv_file)

# Find the index of 'TOT_MALE_PER_TOT_POP'
if 'TOT_MALE_PER_TOT_POP' not in df.columns:
    raise ValueError("Column 'TOT_MALE_PER_TOT_POP' not found in the CSV file.")

start_idx = df.columns.get_loc('TOT_MALE_PER_TOT_POP') + 1

# Select features: all columns after 'TOT_MALE_PER_TOT_POP'
X = df.iloc[:, start_idx:]

y = df['swing']

# Drop 'swing' from features if present
if 'swing' in X.columns:
    X = X.drop(columns=['swing'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Predict swing for Madison County, Alabama using correct columns
row_mask = (df['state_name_2020'] == 'Alabama') & (df['county_name_2020'] == 'Madison County')
if row_mask.any():
    madison_row = df.loc[row_mask]
    madison_features = madison_row.iloc[:, start_idx:]
    if 'swing' in madison_features.columns:
        madison_features = madison_features.drop(columns=['swing'])
    predicted_swing = model.predict(madison_features)
    print(f"Predicted swing for Madison County, Alabama: {predicted_swing[0]}")
else:
    print("Row for Madison County, Alabama not found in the data.")
