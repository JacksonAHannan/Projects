# Streamlit app for time series sales analysis and model comparison
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# --- SARIMA Model Training and Prediction ---
def train_sarima(train, test, order=(1,1,1), seasonal_order=(1,1,1,52)):
	# 52 for weekly seasonality (1 year)
	model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
	model_fit = model.fit(disp=False)
	forecast = model_fit.forecast(steps=len(test))
	return forecast
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# --- Data Loading and Preprocessing ---
@st.cache_data(show_spinner=True)
def load_and_preprocess_data(csv_path):
	# Read in chunks to handle large files
	chunks = []
	for chunk in pd.read_csv(csv_path, chunksize=100000):
		chunks.append(chunk)
	df = pd.concat(chunks, ignore_index=True)
	# Ensure InvoiceDate is datetime
	# Try to infer the format from a sample, then use it for all parsing
	sample_date = df['InvoiceDate'].dropna().astype(str).iloc[0]
	# Try common formats, fallback to default if not matched
	date_formats = ["%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M", "%m/%d/%Y %H:%M"]
	fmt = None
	for f in date_formats:
		try:
			pd.to_datetime(sample_date, format=f)
			fmt = f
			break
		except Exception:
			continue
	if fmt:
		df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format=fmt, errors='coerce')
	else:
		df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
	# Example preprocessing
	df = df.dropna(subset=['InvoiceDate', 'Quantity', 'Price'])
	# Ensure Quantity and Price are numeric
	df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
	df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
	df = df.dropna(subset=['Quantity', 'Price'])
	df['Sales'] = df['Quantity'] * df['Price']
	df['Week'] = df['InvoiceDate'].dt.to_period('W').apply(lambda r: r.start_time)
	weekly_sales = df.groupby('Week')['Sales'].sum().reset_index()
	weekly_sales = weekly_sales.sort_values('Week')
	return weekly_sales

# --- Model Training and Prediction ---
# --- Model Training and Prediction ---
def train_arima(train, test):
	model = ARIMA(train, order=(1,1,1))
	model_fit = model.fit()
	forecast = model_fit.forecast(steps=len(test))
	return forecast

def train_seasonal_naive(train, test, season_length=52):
	# Repeat last season's values
	reps = int(np.ceil(len(test) / season_length))
	last_season = train[-season_length:]
	forecast = np.tile(last_season, reps)[:len(test)]
	return forecast


def train_prophet(train_df, test_periods):
	prophet_df = train_df.rename(columns={'Week': 'ds', 'Sales': 'y'})
	m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
	m.fit(prophet_df)
	future = m.make_future_dataframe(periods=test_periods, freq='W')
	forecast = m.predict(future)
	return forecast['yhat'][-test_periods:].values

def train_linear_regression(train, test):
	X_train = np.arange(len(train)).reshape(-1,1)
	y_train = train.values
	X_test = np.arange(len(train), len(train)+len(test)).reshape(-1,1)
	model = LinearRegression().fit(X_train, y_train)
	preds = model.predict(X_test)
	return preds


# --- Streamlit UI ---
st.title('Time Series Sales Analysis & Model Comparison')
csv_path = os.path.join(os.getcwd(), 'online_retail_II.csv')
st.write('Loading and preprocessing data...')
weekly_sales = load_and_preprocess_data(csv_path)

st.write('Sample of weekly sales data:')
st.dataframe(weekly_sales.head())

# Train/test split
split_ratio = st.slider('Train/Test Split (%)', 50, 95, 80)
split_idx = int(len(weekly_sales) * split_ratio / 100)
train = weekly_sales['Sales'][:split_idx]
test = weekly_sales['Sales'][split_idx:]
train_df = weekly_sales.iloc[:split_idx]
test_df = weekly_sales.iloc[split_idx:]

# --- Model Comparison ---
st.header('Model Comparison')
models = ['ARIMA', 'SARIMA', 'Prophet', 'Linear Regression', 'Seasonal Naive']
selected_models = st.multiselect('Select models to compare:', models, default=models)
results = {}

if 'ARIMA' in selected_models:
	arima_pred = train_arima(train, test)
	results['ARIMA'] = arima_pred
if 'SARIMA' in selected_models:
	sarima_pred = train_sarima(train, test)
	results['SARIMA'] = sarima_pred
if 'Prophet' in selected_models:
	prophet_pred = train_prophet(train_df[['Week','Sales']], len(test))
	results['Prophet'] = prophet_pred
if 'Linear Regression' in selected_models:
	lr_pred = train_linear_regression(train, test)
	results['Linear Regression'] = lr_pred
if 'Seasonal Naive' in selected_models:
	# Use 52 for weekly seasonality (1 year)
	sn_pred = train_seasonal_naive(train, test, season_length=52)
	results['Seasonal Naive'] = sn_pred

# --- Plotting ---
fig = px.line(weekly_sales, x='Week', y='Sales', title='Weekly Sales and Model Forecasts')
for name, preds in results.items():
	preds_clean = np.array(preds)
	mask = ~np.isnan(preds_clean)
	# Only plot if mask length matches test_df['Week'] and is nonzero
	if len(mask) == len(test_df['Week']) and mask.sum() > 0:
		fig.add_scatter(x=test_df['Week'][mask], y=preds_clean[mask], mode='lines', name=f'{name} Forecast')
	elif len(mask) == len(test_df['Week']):
		# All NaN, skip plotting
		continue
	else:
		# Fallback: plot only up to min length
		min_len = min(len(test_df['Week']), len(preds_clean[mask]))
		if min_len > 0:
			fig.add_scatter(x=test_df['Week'][:min_len], y=preds_clean[mask][:min_len], mode='lines', name=f'{name} Forecast')
fig.add_scatter(x=train_df['Week'], y=train, mode='lines', name='Train', line=dict(color='black', dash='dot'))
fig.add_scatter(x=test_df['Week'], y=test, mode='lines', name='Test', line=dict(color='red', dash='dot'))
st.plotly_chart(fig, use_container_width=True)

# --- Metrics ---
st.subheader('Model Accuracy (Test Set)')
mae_dict = {}
rmse_dict = {}
for name, preds in results.items():
	preds_clean = np.array(preds)
	mask = ~np.isnan(preds_clean)
	if mask.sum() == 0:
		st.write(f'**{name}**: No valid predictions (all NaN)')
		continue
	mae = mean_absolute_error(test.values[mask], preds_clean[mask])
	mse = mean_squared_error(test.values[mask], preds_clean[mask])
	rmse = np.sqrt(mse)
	mae_dict[name] = mae
	rmse_dict[name] = rmse
	st.write(f'**{name}**: MAE = {mae:.2f}, RMSE = {rmse:.2f}')

# --- Bar Chart for Model Accuracy ---
if len(rmse_dict) > 0:
	acc_df = pd.DataFrame({
		'Model': list(rmse_dict.keys()),
		'RMSE': list(rmse_dict.values()),
		'MAE': list(mae_dict.values())
	})
	acc_df = acc_df.sort_values('RMSE')
	bar_fig = px.bar(acc_df, x='Model', y=['RMSE', 'MAE'], barmode='group', title='Model Accuracy (Lower is Better)')
	st.plotly_chart(bar_fig, use_container_width=True)
