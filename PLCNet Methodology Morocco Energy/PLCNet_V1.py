"""
Dual-path LSTM + CNN model for multivariate time series forecasting.

Pipeline steps:
1. Load & merge CSV files on DateTime.
2. Sort by DateTime and (optionally) filter / resample if needed.
3. Scale features (MinMaxScaler) based on train split only.
4. Create supervised learning sequences (window -> next step prediction).
5. Build two parallel branches over the same input:
   - LSTM branch (captures long-range temporal dependencies).
   - CNN branch (captures local trends) consisting of:
	   Conv1D(filters=64, kernel_size=2, activation='relu') ->
	   MaxPooling1D(pool_size=2) ->
	   Conv1D(filters=32, kernel_size=2, activation='relu') ->
	   Flatten.
6. Concatenate extracted features and pass through fully connected (Dense + Dropout) layers.
7. Train, evaluate on chronological hold‑out test set, and plot predictions vs actual.

Adjustable hyperparameters are collected near the CONFIG section below.
"""

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import threading
import tkinter as tk
from tkinter import ttk, messagebox

try:  # Soft import; defer hard failure until model build
	import tensorflow as tf  # type: ignore
	from tensorflow.keras import layers, models, callbacks  # type: ignore
except Exception as e:  # pragma: no cover
	print("[Warning] TensorFlow not available at import time. UI will still load. Details:", e)
	tf = None  # sentinel
	layers = models = callbacks = None  # type: ignore

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred):
	"""Calculate Mean Absolute Percentage Error (MAPE)."""
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	# Avoid division by zero by adding small epsilon to denominator
	epsilon = 1e-8
	return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
	"""Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def mean_absolute_scaled_error(y_true, y_pred, y_train):
	"""Calculate Mean Absolute Scaled Error (MASE)."""
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	y_train = np.array(y_train)
	
	# Calculate naive forecast error (seasonal naive with period=1 for simplicity)
	naive_error = np.mean(np.abs(np.diff(y_train)))
	
	# Avoid division by zero
	if naive_error == 0:
		return np.inf if np.mean(np.abs(y_true - y_pred)) > 0 else 0
	
	return np.mean(np.abs(y_true - y_pred)) / naive_error


def coefficient_of_determination(y_true, y_pred):
	"""Calculate R-squared (coefficient of determination)."""
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	ss_res = np.sum((y_true - y_pred) ** 2)
	ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
	
	# Avoid division by zero
	if ss_tot == 0:
		return 1.0 if ss_res == 0 else 0.0
	
	return 1 - (ss_res / ss_tot)


# ===================== CONFIG ===================== #
WINDOW_SIZE = 30             # timesteps per input sample
TRAIN_FRACTION = 0.8         # chronological split
TARGET_COLUMN = "Total_Energy"  # target now set to aggregated energy sum
OUTPUT_HORIZON = 1           # number of future steps to predict (>1 enables multi-step)
EPOCHS = 10                  # increase for real training (UI overrideable)
BATCH_SIZE = 256             # UI overrideable
LSTM_UNITS = 48              # per updated specification (branch LSTM)
# CNN path uses fixed architecture per specification (local trend capture)
CNN_FILTERS = 64             # first conv filters (fixed by spec)
KERNEL_SIZE = 2              # first & second conv kernel size (spec)
MERGE_SEQUENCE_LENGTH = 3    # artificial sequence length for merge LSTM (>=1)
MERGE_LSTM_UNITS = 300       # LSTM after concatenation (spec)
FC_UNITS = [128, 64]         # fully connected dense layers after merge LSTM
FC_DROPOUT = 0.30            # dropout after first dense (spec 30%) and optionally between others
PATIENCE = 3
RANDOM_SEED = 42
SAVE_MERGED_CSV = True
PLOT_RESULTS = True          # show line plot of merged vs actual
UI_ENABLE = True             # launch Tkinter UI
EVALUATE_BRANCHES = True     # evaluate individual LSTM & CNN branches in addition to merged model

np.random.seed(RANDOM_SEED)
if tf is not None:
	tf.random.set_seed(RANDOM_SEED)


# ===================== DATA LOADING ===================== #
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOUJDOUR_PATH = os.path.join(SCRIPT_DIR, "Energy_Data_Morocco", "Boujdour-Table 1.csv")
FOUM_ELOUED_PATH = os.path.join(SCRIPT_DIR, "Energy_Data_Morocco", "Foum eloued-Table 1.csv")
LAAYOUNE_PATH = os.path.join(SCRIPT_DIR, "Energy_Data_Morocco", "Laayoune-Table 1.csv")


def load_and_merge() -> pd.DataFrame:
	df_bouj = pd.read_csv(BOUJDOUR_PATH)
	df_foum = pd.read_csv(FOUM_ELOUED_PATH)
	df_laay = pd.read_csv(LAAYOUNE_PATH)

	merged = df_bouj.merge(df_foum, on="DateTime").merge(df_laay, on="DateTime")
	merged["DateTime"] = pd.to_datetime(merged["DateTime"], errors="coerce")
	merged = merged.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)

	# Clean columns that may have arisen from trailing commas (Unnamed columns full of NaN)
	unnamed_cols = [c for c in merged.columns if c.startswith("Unnamed")]
	if unnamed_cols:
		merged = merged.drop(columns=unnamed_cols)

	# Remove unnamed columns created by trailing commas before computing total
	unnamed_cols = [c for c in merged.columns if c.startswith("Unnamed")]
	if unnamed_cols:
		merged = merged.drop(columns=unnamed_cols)

	# Compute Total_Energy as sum of all numeric feature columns (excluding DateTime & itself)
	numeric_cols = [c for c in merged.columns if c != "DateTime"]
	# Coerce potential numeric strings
	for c in numeric_cols:
		merged[c] = pd.to_numeric(merged[c], errors="coerce")
	merged["Total_Energy"] = merged[numeric_cols].sum(axis=1, skipna=True)

	return merged


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Remove or impute NaNs to prevent NaN propagation into the model.

	Steps:
	1. Drop columns that are entirely NaN or non-numeric except DateTime.
	2. Coerce to numeric.
	3. Interpolate + forward/back fill.
	4. Drop any remaining rows with NaN.
	"""
	feature_cols = [c for c in df.columns if c != "DateTime"]
	# Drop all-NaN columns
	all_nan = [c for c in feature_cols if df[c].isna().all()]
	if all_nan:
		print(f"Dropping all-NaN cols: {all_nan}")
		df = df.drop(columns=all_nan)
		feature_cols = [c for c in df.columns if c != "DateTime"]

	# Coerce to numeric
	for c in feature_cols:
		df[c] = pd.to_numeric(df[c], errors="coerce")

	# Interpolate (time-based if index is DateTime) then forward/back fill
	if not df["DateTime"].is_monotonic_increasing:
		df = df.sort_values("DateTime")
	# Set index temporarily for interpolation
	temp_index = df.set_index("DateTime")
	temp_index = temp_index.interpolate(method="time", limit_direction="both")
	temp_index = temp_index.fillna(method="ffill").fillna(method="bfill")
	df = temp_index.reset_index()

	# Final drop if still any NaNs
	remaining_nans = df.isna().sum().sum()
	if remaining_nans:
		print(f"Warning: dropping rows with residual NaNs: {remaining_nans}")
		df = df.dropna().reset_index(drop=True)

	return df


def chronological_split(df: pd.DataFrame, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
	split_idx = int(len(df) * frac)
	return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_sequences(features: np.ndarray, target: np.ndarray, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
	"""Create sliding window sequences without target leakage.

	features: (N, F)
	target: (N,)
	Returns X: (samples, window, F), y shape depends on horizon.
	"""
	X, y = [], []
	limit = len(features) - (horizon - 1)
	for i in range(window, limit):
		X.append(features[i - window:i, :])
		if horizon == 1:
			y.append(target[i])
		else:
			y.append(target[i:i + horizon])
	return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
	if TARGET_COLUMN not in df.columns:
		raise ValueError(f"TARGET_COLUMN '{TARGET_COLUMN}' not found. Available: {list(df.columns)}")

	train_df, test_df = chronological_split(df, TRAIN_FRACTION)
	train_df = clean_features(train_df)
	test_df = clean_features(test_df)

	# Separate target to avoid leakage
	feature_cols = [c for c in train_df.columns if c not in ("DateTime", TARGET_COLUMN)]
	# Scale features
	feature_scaler = MinMaxScaler()
	train_feat_scaled = feature_scaler.fit_transform(train_df[feature_cols])
	test_feat_scaled = feature_scaler.transform(test_df[feature_cols])

	# Scale target separately (reshape for scaler)
	target_scaler = MinMaxScaler()
	train_target_scaled = target_scaler.fit_transform(train_df[[TARGET_COLUMN]]).ravel()
	test_target_scaled = target_scaler.transform(test_df[[TARGET_COLUMN]]).ravel()

	if np.isnan(train_feat_scaled).any() or np.isnan(test_feat_scaled).any():
		raise ValueError("NaNs detected in scaled feature data.")

	X_train, y_train = build_sequences(train_feat_scaled, train_target_scaled, WINDOW_SIZE, OUTPUT_HORIZON)

	# For test continuity join tail of train features/targets
	if len(train_feat_scaled) >= WINDOW_SIZE:
		feat_test_concat = np.vstack([train_feat_scaled[-WINDOW_SIZE:], test_feat_scaled])
		target_test_concat = np.concatenate([train_target_scaled[-WINDOW_SIZE:], test_target_scaled])
	else:
		feat_test_concat = test_feat_scaled
		target_test_concat = test_target_scaled
	X_test, y_test = build_sequences(feat_test_concat, target_test_concat, WINDOW_SIZE, OUTPUT_HORIZON)
	if len(train_feat_scaled) >= WINDOW_SIZE:
		drop_n = WINDOW_SIZE
		X_test = X_test[drop_n:]
		y_test = y_test[drop_n:]

	return X_train, y_train, X_test, y_test, feature_scaler, target_scaler


# ===================== MODEL ===================== #
def build_model(timesteps: int, n_features: int):  # return Keras model
	if tf is None or layers is None:
		raise ImportError("TensorFlow not installed or failed to load. Install with: pip install tensorflow==2.15.0")
	inp = layers.Input(shape=(timesteps, n_features), name="input_sequence")

	# LSTM branch restored to use temporal sequence directly
	x_lstm = layers.LSTM(64, return_sequences=True)(inp)
	x_lstm = layers.LSTM(LSTM_UNITS, activation="relu")(x_lstm)

	# CNN branch (local trend extraction path per user specification)
	x_cnn = layers.Conv1D(CNN_FILTERS, KERNEL_SIZE, activation="relu")(inp)
	x_cnn = layers.MaxPooling1D(pool_size=2)(x_cnn)
	x_cnn = layers.Conv1D(32, KERNEL_SIZE, activation="relu")(x_cnn)
	x_cnn = layers.Flatten()(x_cnn)

	# Branch-specific heads (for standalone evaluation)
	if EVALUATE_BRANCHES:
		lstm_head = layers.Dense(64, activation="sigmoid")(x_lstm)
		lstm_pred = layers.Dense(OUTPUT_HORIZON, activation="sigmoid", name="lstm_pred")(lstm_head)

		cnn_head = layers.Dense(64, activation="sigmoid")(x_cnn)
		cnn_pred = layers.Dense(OUTPUT_HORIZON, activation="sigmoid", name="cnn_pred")(cnn_head)

	# Concatenate for merged path
	x_merge_in = layers.Concatenate()([x_lstm, x_cnn])

	# Merge LSTM path: repeat the merged feature vector to form a short artificial sequence
	x_seq = layers.RepeatVector(MERGE_SEQUENCE_LENGTH)(x_merge_in) if MERGE_SEQUENCE_LENGTH > 1 else layers.Reshape((1, -1))(x_merge_in)
	x_seq = layers.LSTM(MERGE_LSTM_UNITS, activation="relu")(x_seq)

	# Fully connected sigmoid-activated dense layers
	for i, units in enumerate(FC_UNITS):
		x_seq = layers.Dense(units, activation="sigmoid")(x_seq)
		if i == 0:
			x_seq = layers.Dropout(FC_DROPOUT)(x_seq)

	output_units = OUTPUT_HORIZON
	merged_pred = layers.Dense(output_units, activation="sigmoid", name="merged_pred")(x_seq)

	if EVALUATE_BRANCHES:
		outputs = [lstm_pred, cnn_pred, merged_pred]
		losses = {"lstm_pred": "mse", "cnn_pred": "mse", "merged_pred": "mse"}
		loss_weights = {"lstm_pred": 0.3, "cnn_pred": 0.3, "merged_pred": 1.0}
		metrics = {"lstm_pred": ["mae"], "cnn_pred": ["mae"], "merged_pred": ["mae"]}
		model = models.Model(inputs=inp, outputs=outputs, name="DualPath_LSTM_CNN_MultiOut")
		model.compile(optimizer="adam", loss=losses, loss_weights=loss_weights, metrics=metrics)
	else:
		model = models.Model(inputs=inp, outputs=merged_pred, name="DualPath_LSTM_CNN")
		model.compile(optimizer="adam", loss="mse", metrics=["mae"])

	return model


def evaluate_and_report(model, X_test: np.ndarray, y_test: np.ndarray, y_train: np.ndarray = None):
	preds = model.predict(X_test, verbose=0)

	metrics: Dict[str, Dict[str, float]] = {}

	def _metric_report(name: str, pred_arr: np.ndarray):
		if np.isnan(pred_arr).any():
			print(f"Warning: NaNs in {name} predictions; imputing with column means.")
			col_means = np.nanmean(pred_arr, axis=0)
			inds = np.where(np.isnan(pred_arr))
			pred_arr[inds] = np.take(col_means, inds[1])
		y_true_flat = y_test.reshape(-1)
		pred_flat = pred_arr.reshape(-1)
		
		# Basic metrics
		mae = mean_absolute_error(y_true_flat, pred_flat)
		rmse = math.sqrt(mean_squared_error(y_true_flat, pred_flat))
		mape = mean_absolute_percentage_error(y_true_flat, pred_flat)
		
		# Advanced metrics
		smape = symmetric_mean_absolute_percentage_error(y_true_flat, pred_flat)
		r2 = coefficient_of_determination(y_true_flat, pred_flat)
		
		# MASE (only if training data is provided)
		mase = None
		if y_train is not None:
			mase = mean_absolute_scaled_error(y_true_flat, pred_flat, y_train.reshape(-1))
		
		# Print metrics
		print(f"{name} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | SMAPE: {smape:.2f}% | R²: {r2:.4f}", end="")
		if mase is not None:
			print(f" | MASE: {mase:.4f}")
		else:
			print()
		
		# Store metrics
		metric_dict = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape, "R2": r2}
		if mase is not None:
			metric_dict["MASE"] = mase
			
		metrics[name] = metric_dict
		return mae, rmse, mape

	if EVALUATE_BRANCHES:
		lstm_pred, cnn_pred, merged_pred = preds
		print("Branch & merged evaluation:")
		_metric_report("LSTM", lstm_pred)
		_metric_report("CNN", cnn_pred)
		_metric_report("Merged", merged_pred)
		return {"preds": {"lstm": lstm_pred, "cnn": cnn_pred, "merged": merged_pred}, "metrics": metrics}
	else:
		_metric_report("Merged", preds)
		return {"preds": {"merged": preds}, "metrics": metrics}


def plot_model_comparisons(results: Dict, y_test: np.ndarray, max_samples: int = 500):
	"""Create comprehensive plots comparing all model predictions vs actual values."""
	try:
		import matplotlib.pyplot as plt
		
		# Extract actual values for plotting - ensure 1D array
		if y_test.ndim == 2:
			actual_plot = y_test[:, 0] if y_test.shape[1] > 1 else y_test.flatten()
		else:
			actual_plot = y_test
		actual_plot = actual_plot[:max_samples]
		
		print(f"Debug: actual_plot shape: {actual_plot.shape}")
		
		if EVALUATE_BRANCHES:
			# Create subplot for each model + combined view
			fig, axes = plt.subplots(2, 2, figsize=(15, 10))
			fig.suptitle(f'Model Predictions vs Actual - {TARGET_COLUMN}', fontsize=16)
			
			models_data = [
				('LSTM Branch', results["preds"]["lstm"], 'blue'),
				('CNN Branch', results["preds"]["cnn"], 'red'), 
				('Merged Model', results["preds"]["merged"], 'green')
			]
			
			# Individual model plots
			for i, (name, pred, color) in enumerate(models_data):
				row, col = i // 2, i % 2
				ax = axes[row, col]
				
				# Extract prediction values for plotting - ensure 1D array
				if pred.ndim == 2:
					pred_plot = pred[:, 0] if pred.shape[1] > 1 else pred.flatten()
				else:
					pred_plot = pred
				pred_plot = pred_plot[:max_samples]
				
				print(f"Debug: {name} pred_plot shape: {pred_plot.shape}")
				
				# Ensure both arrays have the same length
				min_len = min(len(actual_plot), len(pred_plot))
				actual_truncated = actual_plot[:min_len]
				pred_truncated = pred_plot[:min_len]
				
				ax.plot(actual_truncated, label='Actual', color='black', linewidth=1.5, alpha=0.8)
				ax.plot(pred_truncated, label=f'{name} Prediction', color=color, linewidth=1.2)
				ax.set_title(f'{name}')
				ax.set_xlabel('Sample (chronological)')
				ax.set_ylabel('Scaled value')
				ax.legend()
				ax.grid(True, alpha=0.3)
				
				# Add metrics to plot
				if name.split()[0].lower() in results["metrics"]:
					metrics = results["metrics"][name.split()[0].lower()]
					metric_text = f'MAE: {metrics["MAE"]:.4f}\nRMSE: {metrics["RMSE"]:.4f}\nMAPE: {metrics["MAPE"]:.2f}%\nSMAPE: {metrics["SMAPE"]:.2f}%\nR²: {metrics["R2"]:.4f}'
					if "MASE" in metrics:
						metric_text += f'\nMASE: {metrics["MASE"]:.4f}'
					ax.text(0.02, 0.98, metric_text, 
						transform=ax.transAxes, verticalalignment='top', 
						bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
			
			# Combined comparison plot
			ax_combined = axes[1, 1]
			ax_combined.plot(actual_plot, label='Actual', color='black', linewidth=2, alpha=0.8)
			
			for name, pred, color in models_data:
				# Extract prediction values for plotting - ensure 1D array
				if pred.ndim == 2:
					pred_plot = pred[:, 0] if pred.shape[1] > 1 else pred.flatten()
				else:
					pred_plot = pred
				pred_plot = pred_plot[:max_samples]
				
				# Ensure both arrays have the same length
				min_len = min(len(actual_plot), len(pred_plot))
				pred_truncated = pred_plot[:min_len]
				
				ax_combined.plot(pred_truncated, label=f'{name}', color=color, linewidth=1.2, alpha=0.7)
			
			ax_combined.set_title('All Models Comparison')
			ax_combined.set_xlabel('Sample (chronological)')
			ax_combined.set_ylabel('Scaled value')
			ax_combined.legend()
			ax_combined.grid(True, alpha=0.3)
			
		else:
			# Single model plot
			fig, ax = plt.subplots(1, 1, figsize=(12, 6))
			
			pred = results["preds"]["merged"]
			if pred.ndim == 2:
				pred_plot = pred[:, 0] if pred.shape[1] > 1 else pred.flatten()
			else:
				pred_plot = pred
			pred_plot = pred_plot[:max_samples]
			
			# Ensure both arrays have the same length
			min_len = min(len(actual_plot), len(pred_plot))
			actual_truncated = actual_plot[:min_len]
			pred_truncated = pred_plot[:min_len]
			
			ax.plot(actual_truncated, label='Actual', color='black', linewidth=1.5)
			ax.plot(pred_truncated, label='Merged Model Prediction', color='green', linewidth=1.2)
			ax.set_title(f'Merged Model Predictions vs Actual - {TARGET_COLUMN}')
			ax.set_xlabel('Sample (chronological)')
			ax.set_ylabel('Scaled value')
			ax.legend()
			ax.grid(True, alpha=0.3)
			
			# Add metrics
			if "merged" in results["metrics"]:
				metrics = results["metrics"]["merged"]
				metric_text = f'MAE: {metrics["MAE"]:.4f}\nRMSE: {metrics["RMSE"]:.4f}\nMAPE: {metrics["MAPE"]:.2f}%\nSMAPE: {metrics["SMAPE"]:.2f}%\nR²: {metrics["R2"]:.4f}'
				if "MASE" in metrics:
					metric_text += f'\nMASE: {metrics["MASE"]:.4f}'
				ax.text(0.02, 0.98, metric_text, 
					transform=ax.transAxes, verticalalignment='top',
					bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
		
		plt.tight_layout()
		plt.show()
		
		# Also create a scatter plot for correlation analysis
		if EVALUATE_BRANCHES:
			fig, axes = plt.subplots(1, 3, figsize=(15, 4))
			fig.suptitle('Prediction vs Actual Correlation Analysis', fontsize=14)
			
			for i, (name, pred, color) in enumerate(models_data):
				# Extract prediction values for plotting - ensure 1D array
				if pred.ndim == 2:
					pred_plot = pred[:, 0] if pred.shape[1] > 1 else pred.flatten()
				else:
					pred_plot = pred
				pred_plot = pred_plot[:max_samples]
				
				# Ensure both arrays have the same length
				min_len = min(len(actual_plot), len(pred_plot))
				actual_scatter = actual_plot[:min_len]
				pred_scatter = pred_plot[:min_len]
				
				axes[i].scatter(actual_scatter, pred_scatter, alpha=0.6, color=color, s=1)
				axes[i].plot([actual_scatter.min(), actual_scatter.max()], 
					[actual_scatter.min(), actual_scatter.max()], 'k--', alpha=0.8)
				axes[i].set_xlabel('Actual')
				axes[i].set_ylabel('Predicted')
				axes[i].set_title(f'{name}')
				axes[i].grid(True, alpha=0.3)
				
				# Calculate correlation coefficient
				corr = np.corrcoef(actual_scatter, pred_scatter)[0, 1]
				axes[i].text(0.05, 0.95, f'R = {corr:.3f}', transform=axes[i].transAxes,
					bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
			
			plt.tight_layout()
			plt.show()
			
	except Exception as e:
		print(f"Plotting failed: {e}")
		import traceback
		traceback.print_exc()


def main(progress_callback=None):
	df = load_and_merge()
	if SAVE_MERGED_CSV:
		out_path = os.path.join(SCRIPT_DIR, "Energy_Data_Morocco", "merged_energy_data.csv")
		df.to_csv(out_path, index=False)
		print(f"Merged data saved to {out_path} (rows={len(df)})")

	print("Preparing data sequences ...")
	X_train, y_train, X_test, y_test, scaler, target_index = prepare_data(df)
	print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}, Features: {X_train.shape[-1]}, Horizon: {OUTPUT_HORIZON}")

	model = build_model(WINDOW_SIZE, X_train.shape[-1])
	model.summary()

	es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

	class ProgressCB(callbacks.Callback):
		def on_epoch_end(self, epoch, logs=None):
			if progress_callback:
				msg = f"Epoch {epoch+1}/{EPOCHS} - " + ", ".join([f"{k}: {v:.4f}" for k, v in (logs or {}).items()])
				progress_callback(msg)
		def on_train_begin(self, logs=None):
			if progress_callback:
				progress_callback("Training started")
		def on_train_end(self, logs=None):
			if progress_callback:
				progress_callback("Training finished")

	cb_list = [es, ProgressCB()]

	if EVALUATE_BRANCHES:
		train_targets = {"lstm_pred": y_train, "cnn_pred": y_train, "merged_pred": y_train}
		val_split = 0.1
		history = model.fit(
			X_train,
			train_targets,
			validation_split=val_split,
			epochs=EPOCHS,
			batch_size=BATCH_SIZE,
			callbacks=cb_list,
			verbose=1,
			)
	else:
		history = model.fit(
			X_train,
			y_train,
			validation_split=0.1,
			epochs=EPOCHS,
			batch_size=BATCH_SIZE,
			callbacks=cb_list,
			verbose=1,
		)

	results = evaluate_and_report(model, X_test, y_test, y_train)

	# Enhanced plotting for all models
	print(f"Debug: PLOT_RESULTS = {PLOT_RESULTS}")
	print(f"Debug: progress_callback is None = {progress_callback is None}")
	print(f"Debug: About to attempt plotting...")
	
	if PLOT_RESULTS:
		if progress_callback is None:
			print("Displaying plots...")
			plot_model_comparisons(results, y_test, max_samples=500)
		else:
			print("Skipping plots due to progress_callback (running from UI)")
	else:
		print("Plotting disabled (PLOT_RESULTS = False)")

	# Save model
	model_path = os.path.join(SCRIPT_DIR, "dualpath_lstm_cnn_model.keras")
	model.save(model_path)
	print(f"Model saved to {model_path}")

	return results


# ===================== TKINTER UI ===================== #
def launch_ui():  # pragma: no cover (UI not covered by tests)
	import matplotlib
	matplotlib.use("TkAgg")
	from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
	import matplotlib.pyplot as plt

	root = tk.Tk()
	root.title("PLCNet Trainer")

	params: Dict[str, Any] = {
		"WINDOW_SIZE": WINDOW_SIZE,
		"EPOCHS": EPOCHS,
		"BATCH_SIZE": BATCH_SIZE,
		"LSTM_UNITS": LSTM_UNITS,
		"CNN_FILTERS": CNN_FILTERS,
		"KERNEL_SIZE": KERNEL_SIZE,
		"MERGE_LSTM_UNITS": MERGE_LSTM_UNITS,
		"FC_UNITS": ",".join(map(str, FC_UNITS)),
		"FC_DROPOUT": FC_DROPOUT,
		"OUTPUT_HORIZON": OUTPUT_HORIZON,
	}

	entries: Dict[str, tk.Entry] = {}

	form_frame = ttk.Frame(root)
	form_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

	for i, (k, v) in enumerate(params.items()):
		ttk.Label(form_frame, text=k).grid(row=i, column=0, sticky="w")
		e = ttk.Entry(form_frame, width=15)
		e.insert(0, str(v))
		e.grid(row=i, column=1, padx=4, pady=2)
		entries[k] = e

	result_box = tk.Text(root, height=10, width=80)
	result_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

	plot_frame = ttk.Frame(root)
	plot_frame.pack(fill=tk.BOTH, expand=True)

	fig, axes = plt.subplots(1, 2, figsize=(8, 3))
	canvas = FigureCanvasTkAgg(fig, master=plot_frame)
	canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

	train_button = ttk.Button(root, text="Train", command=lambda: start_training())
	train_button.pack(pady=6)

	def start_training():
		train_button.config(state=tk.DISABLED)
		result_box.delete("1.0", tk.END)
		cfg = {}
		try:
			cfg["WINDOW_SIZE"] = int(entries["WINDOW_SIZE"].get())
			cfg["EPOCHS"] = int(entries["EPOCHS"].get())
			cfg["BATCH_SIZE"] = int(entries["BATCH_SIZE"].get())
			cfg["LSTM_UNITS"] = int(entries["LSTM_UNITS"].get())
			cfg["CNN_FILTERS"] = int(entries["CNN_FILTERS"].get())
			cfg["KERNEL_SIZE"] = int(entries["KERNEL_SIZE"].get())
			cfg["MERGE_LSTM_UNITS"] = int(entries["MERGE_LSTM_UNITS"].get())
			cfg["FC_UNITS"] = [int(x.strip()) for x in entries["FC_UNITS"].get().split(',') if x.strip()]
			cfg["FC_DROPOUT"] = float(entries["FC_DROPOUT"].get())
			cfg["OUTPUT_HORIZON"] = int(entries["OUTPUT_HORIZON"].get())
		except ValueError as ve:
			messagebox.showerror("Input Error", f"Invalid parameter: {ve}")
			train_button.config(state=tk.NORMAL)
			return

		thread = threading.Thread(target=lambda: run_training(cfg))
		thread.start()

	def run_training(cfg: Dict[str, Any]):
		global WINDOW_SIZE, EPOCHS, BATCH_SIZE, LSTM_UNITS, CNN_FILTERS, KERNEL_SIZE, MERGE_LSTM_UNITS, FC_UNITS, FC_DROPOUT, OUTPUT_HORIZON
		WINDOW_SIZE = cfg["WINDOW_SIZE"]
		EPOCHS = cfg["EPOCHS"]
		BATCH_SIZE = cfg["BATCH_SIZE"]
		LSTM_UNITS = cfg["LSTM_UNITS"]
		CNN_FILTERS = cfg["CNN_FILTERS"]
		KERNEL_SIZE = cfg["KERNEL_SIZE"]
		MERGE_LSTM_UNITS = cfg["MERGE_LSTM_UNITS"]
		FC_UNITS = cfg["FC_UNITS"]
		FC_DROPOUT = cfg["FC_DROPOUT"]
		OUTPUT_HORIZON = cfg["OUTPUT_HORIZON"]

		def progress(msg: str):
			root.after(0, lambda: (result_box.insert(tk.END, msg + "\n"), result_box.see(tk.END)))

		import traceback
		try:
			results = main(progress_callback=progress)
			metrics = results["metrics"]
			# Update text box
			lines = ["Training complete. Metrics:\n"]
			for name, vals in metrics.items():
				line = f"{name}: MAE={vals['MAE']:.4f} RMSE={vals['RMSE']:.4f} MAPE={vals['MAPE']:.2f}% SMAPE={vals['SMAPE']:.2f}% R²={vals['R2']:.4f}"
				if "MASE" in vals:
					line += f" MASE={vals['MASE']:.4f}"
				lines.append(line)
			result_text = "\n".join(lines)
		except Exception as e:
			trace = traceback.format_exc()
			result_text = f"Error: {e}\n{trace}"
			metrics = {}

		def update_ui():
			result_box.insert(tk.END, result_text + "\n")
			# Draw bar charts if metrics available
			if metrics:
				axes[0].clear(); axes[1].clear()
				model_names = list(metrics.keys())
				maes = [metrics[m]["MAE"] for m in model_names]
				rmses = [metrics[m]["RMSE"] for m in model_names]
				x = list(range(len(model_names)))
				axes[0].bar(x, maes, color=['#1f77b4','#ff7f0e','#2ca02c'][:len(x)])
				axes[0].set_title('MAE')
				axes[1].bar(x, rmses, color=['#1f77b4','#ff7f0e','#2ca02c'][:len(x)])
				axes[1].set_title('RMSE')
				for ax in axes:
					ax.set_ylabel('Error')
					ax.set_xticks(x)
					ax.set_xticklabels(model_names, rotation=15, ha='right')
				fig.tight_layout()
				canvas.draw()
			train_button.config(state=tk.NORMAL)
		root.after(0, update_ui)

	root.mainloop()


if __name__ == "__main__":
	if UI_ENABLE:
		launch_ui()
	else:
		main()
