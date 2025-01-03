import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the Excel file
file_path = '/Users/jacksonhannan/Downloads/GDL-Indicators-(2022)-data.xlsx'
df = pd.read_excel(file_path)

# Inspect the data to identify relevant columns
print(df.head())

# Replace with actual column names
life_expectancy_col = 'Life expectancy'  # Life Expectancy Variable
gdp_per_capita_col = 'Log Gross National Income per capita'    # GDP/Capita Variable

# Filter rows with missing data in relevant columns
df = df.dropna(subset=[life_expectancy_col, gdp_per_capita_col])

# Extract features and target variables
X = df[[gdp_per_capita_col]].values.reshape(-1, 1)
y = df[life_expectancy_col].values

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Generate predictions
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(x_range, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('GDP per capita')
plt.ylabel('Life Expectancy')
plt.title('Linear Regression: Life Expectancy vs GDP per capita')
plt.legend()
plt.grid(True)
plt.show()
