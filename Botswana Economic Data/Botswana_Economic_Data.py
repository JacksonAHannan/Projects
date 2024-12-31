import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

le_db = pd.read_csv('life-expectancy.csv')
gdpc_db = pd.read_csv('gdp-per-capita-worldbank.csv')
botswana_le_db = le_db[le_db["Code"] == "BWA"]
botswana_gdpc_db = gdpc_db[gdpc_db['Code'] == "BWA"]


print(botswana_le_db.head(5))
print(botswana_gdpc_db.head(5))

plt.plot(botswana_le_db['Year'], botswana_le_db['Life Expectancy']) 
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title('Life Expectancy in Botswana by year')
plt.show()

plt.plot(botswana_gdpc_db['Year'], botswana_gdpc_db['GDP per capita']) 
plt.xlabel('Year')
plt.ylabel('GDP per capita PPP 2017 USD')
plt.title('GDP per capita in Botswana by year')
plt.show()

plt.plot()
