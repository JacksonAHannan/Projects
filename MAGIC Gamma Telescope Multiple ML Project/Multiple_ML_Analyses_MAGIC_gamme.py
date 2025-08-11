from sklearn.naive_bayes import GaussianNB
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Default parameters
params = {
	'test_size': 0.2,
	'random_state': 42,
	'k': 5,
	'n_features': 10,
	'n_estimators': 100
}

# Neural Network training function
def on_train_nn():
	global X_train, X_test, y_train, y_test, selected_features
	try:
		if X_train is None or X_test is None:
			error_label.config(text="Please run preprocessing first.")
			return
		import tensorflow as tf
		from tensorflow.keras.models import Sequential
		from tensorflow.keras.layers import Dense, Dropout
		from tensorflow.keras.optimizers import Adam

		epochs = int(epochs_var.get())

		# Build a simple feedforward neural network
		model = Sequential([
			Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
			Dropout(0.2),
			Dense(16, activation='relu'),
			Dropout(0.2),
			Dense(1, activation='sigmoid')
		])
		model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

		# Train the model
		history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)

		# Evaluate
		y_pred_prob = model.predict(X_test)
		y_pred = (y_pred_prob > 0.5).astype(int).flatten()
		acc = accuracy_score(y_test, y_pred)
		report = classification_report(y_test, y_pred, target_names=['hadron', 'gamma'], digits=3)
		cm = confusion_matrix(y_test, y_pred)
		cm_str = f"Confusion Matrix:\n[[TN FP]\n [FN TP]]\n{cm}"

		# Training/validation loss and accuracy report
		train_loss = history.history['loss']
		val_loss = history.history['val_loss']
		train_acc = history.history['accuracy']
		val_acc = history.history['val_accuracy']
		epoch_report = "Epoch\tTrain Loss\tVal Loss\tTrain Acc\tVal Acc\n"
		for i in range(epochs):
			epoch_report += f"{i+1}\t{train_loss[i]:.4f}\t{val_loss[i]:.4f}\t{train_acc[i]:.4f}\t{val_acc[i]:.4f}\n"

		results = (
			f"Neural Network Test Accuracy: {acc:.4f}\n"
			f"Classification Report:\n{report}\n"
			f"{cm_str}\n"
			f"Features used: {list(selected_features) if selected_features is not None else 'N/A'}\n\n"
			f"Training/Validation Loss and Accuracy by Epoch:\n{epoch_report}"
		)
		results_text.config(state='normal')
		results_text.delete(1.0, tk.END)
		results_text.insert(tk.END, results)
		results_text.config(state='disabled')

		# Plot training/validation loss and accuracy as two adjacent line charts
		fig, axes = plt.subplots(1, 2, figsize=(12, 4))
		# Loss plot
		axes[0].plot(range(1, epochs+1), train_loss, label='Train Loss')
		axes[0].plot(range(1, epochs+1), val_loss, label='Val Loss')
		axes[0].set_xlabel('Epoch')
		axes[0].set_ylabel('Loss')
		axes[0].set_title('Loss over Epochs')
		axes[0].legend()
		axes[0].grid(True)
		# Accuracy plot
		axes[1].plot(range(1, epochs+1), train_acc, label='Train Acc')
		axes[1].plot(range(1, epochs+1), val_acc, label='Val Acc')
		axes[1].set_xlabel('Epoch')
		axes[1].set_ylabel('Accuracy')
		axes[1].set_title('Accuracy over Epochs')
		axes[1].legend()
		axes[1].grid(True)
		plt.tight_layout()
		plt.show()

		# Plot graphical confusion matrix
		plt.figure(figsize=(4,4))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
					xticklabels=['hadron', 'gamma'], yticklabels=['hadron', 'gamma'])
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.title('Neural Network Confusion Matrix')
		plt.tight_layout()
		plt.show()
		error_label.config(text="")
	except Exception as e:
		error_label.config(text=f"Error: {e}")

# Random Forest training function
def on_train_rf():
	global X_train, X_test, y_train, y_test, selected_features
	try:
		if X_train is None or X_test is None:
			error_label.config(text="Please run preprocessing first.")
			return
		# Always use the current value from the UI
		n_estimators = int(n_estimators_var.get())
		rf = RandomForestClassifier(n_estimators=n_estimators, random_state=params['random_state'])
		rf.fit(X_train, y_train)
		y_pred = rf.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		report = classification_report(y_test, y_pred, target_names=['hadron', 'gamma'], digits=3)
		cm = confusion_matrix(y_test, y_pred)
		cm_str = f"Confusion Matrix:\n[[TN FP]\n [FN TP]]\n{cm}"
		results = (
			f"Random Forest (n_estimators={n_estimators}) Test Accuracy: {acc:.4f}\n"
			f"Classification Report:\n{report}\n"
			f"{cm_str}\n"
			f"Features used: {list(selected_features) if selected_features is not None else 'N/A'}"
		)
		results_text.config(state='normal')
		results_text.delete(1.0, tk.END)
		results_text.insert(tk.END, results)
		results_text.config(state='disabled')
		# Plot graphical confusion matrix
		plt.figure(figsize=(4,4))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
					xticklabels=['hadron', 'gamma'], yticklabels=['hadron', 'gamma'])
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.title('Random Forest Confusion Matrix')
		plt.tight_layout()
		plt.show()
		error_label.config(text="")
	except Exception as e:
		error_label.config(text=f"Error: {e}")

# KNN training function
def on_train_knn():
	global X_train, X_test, y_train, y_test, selected_features
	try:
		if X_train is None or X_test is None:
			error_label.config(text="Please run preprocessing first.")
			return
		k = int(k_var.get())
		params['k'] = k
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		report = classification_report(y_test, y_pred, target_names=['hadron', 'gamma'], digits=3)
		cm = confusion_matrix(y_test, y_pred)
		cm_str = f"Confusion Matrix:\n[[TN FP]\n [FN TP]]\n{cm}"
		results = (
			f"KNN (k={k}) Test Accuracy: {acc:.4f}\n"
			f"Classification Report:\n{report}\n"
			f"{cm_str}\n"
			f"Features used: {list(selected_features) if selected_features is not None else 'N/A'}"
		)
		results_text.config(state='normal')
		results_text.delete(1.0, tk.END)
		results_text.insert(tk.END, results)
		results_text.config(state='disabled')
		# Plot graphical confusion matrix
		plt.figure(figsize=(4,4))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
					xticklabels=['hadron', 'gamma'], yticklabels=['hadron', 'gamma'])
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.title(f'KNN Confusion Matrix (k={k})')
		plt.tight_layout()
		plt.show()
		error_label.config(text="")
	except Exception as e:
		error_label.config(text=f"Error: {e}")

# Naive Bayes training function
def on_train_nb():
	global X_train, X_test, y_train, y_test, selected_features
	try:
		if X_train is None or X_test is None:
			error_label.config(text="Please run preprocessing first.")
			return
		nb = GaussianNB()
		nb.fit(X_train, y_train)
		y_pred = nb.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		report = classification_report(y_test, y_pred, target_names=['hadron', 'gamma'], digits=3)
		cm = confusion_matrix(y_test, y_pred)
		cm_str = f"Confusion Matrix:\n[[TN FP]\n [FN TP]]\n{cm}"
		results = (
			f"Naive Bayes Test Accuracy: {acc:.4f}\n"
			f"Classification Report:\n{report}\n"
			f"{cm_str}\n"
			f"Features used: {list(selected_features) if selected_features is not None else 'N/A'}"
		)
		results_text.config(state='normal')
		results_text.delete(1.0, tk.END)
		results_text.insert(tk.END, results)
		results_text.config(state='disabled')
		# Plot graphical confusion matrix
		plt.figure(figsize=(4,4))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
					xticklabels=['hadron', 'gamma'], yticklabels=['hadron', 'gamma'])
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.title('Naive Bayes Confusion Matrix')
		plt.tight_layout()
		plt.show()
		error_label.config(text="")
	except Exception as e:
		error_label.config(text=f"Error: {e}")
from sklearn.linear_model import LogisticRegression

# Logistic Regression training function
def on_train_logreg():
	global X_train, X_test, y_train, y_test, selected_features
	try:
		if X_train is None or X_test is None:
			error_label.config(text="Please run preprocessing first.")
			return
		logreg = LogisticRegression(max_iter=1000, random_state=params['random_state'])
		logreg.fit(X_train, y_train)
		y_pred = logreg.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		report = classification_report(y_test, y_pred, target_names=['hadron', 'gamma'], digits=3)
		cm = confusion_matrix(y_test, y_pred)
		cm_str = f"Confusion Matrix:\n[[TN FP]\n [FN TP]]\n{cm}"
		results = (
			f"Logistic Regression Test Accuracy: {acc:.4f}\n"
			f"Classification Report:\n{report}\n"
			f"{cm_str}\n"
			f"Features used: {list(selected_features) if selected_features is not None else 'N/A'}"
		)
		results_text.config(state='normal')
		results_text.delete(1.0, tk.END)
		results_text.insert(tk.END, results)
		results_text.config(state='disabled')
		# Plot graphical confusion matrix
		plt.figure(figsize=(4,4))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
					xticklabels=['hadron', 'gamma'], yticklabels=['hadron', 'gamma'])
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.title('Logistic Regression Confusion Matrix')
		plt.tight_layout()
		plt.show()
		error_label.config(text="")
	except Exception as e:
		error_label.config(text=f"Error: {e}")
		


# Global variables for data and feature selection
X_train = X_test = y_train = y_test = None
selected_features = None

def on_run():
	global X_train, X_test, y_train, y_test, selected_features
	try:
		params['test_size'] = float(test_size_var.get())
		params['random_state'] = int(random_state_var.get())
		params['n_features'] = int(n_features_var.get())
		params['n_estimators'] = int(n_estimators_var.get())

		# Column names from magic04.names
		columns = [
			'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
			'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'
		]
		# Load the data
		data_path = "magic+gamma+telescope/magic04.data"
		df = pd.read_csv(data_path, names=columns)
		# Encode class labels: gamma (g) = 1, hadron (h) = 0
		df['class'] = df['class'].map({'g': 1, 'h': 0})

		# Count class distribution
		count_a = (df['class'] == 1).sum()
		count_b = (df['class'] == 0).sum()

		# Oversample the minority class
		if count_a > count_b:
			majority_class = 1
			minority_class = 0
			n_majority = count_a
			n_minority = count_b
		else:
			majority_class = 0
			minority_class = 1
			n_majority = count_b
			n_minority = count_a

		df_majority = df[df['class'] == majority_class]
		df_minority = df[df['class'] == minority_class]
		df_minority_oversampled = df_minority.sample(n=n_majority, replace=True, random_state=params['random_state'])
		df_balanced = pd.concat([df_majority, df_minority_oversampled], axis=0).sample(frac=1, random_state=params['random_state']).reset_index(drop=True)

		# Separate features and target
		X = df_balanced.drop('class', axis=1)
		y = df_balanced['class']

		# Feature importance using RandomForest
		rf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
		rf.fit(X, y)
		importances = rf.feature_importances_
		indices = np.argsort(importances)[::-1]
		selected_features = X.columns[indices][:params['n_features']]
		# Scale only selected features
		scaler = StandardScaler()
		X_selected = scaler.fit_transform(X[selected_features])
		# Dimensionality reduction analysis (PCA)
		pca = PCA()
		X_pca = pca.fit_transform(X_selected)
		# Plot explained variance ratio
		plt.figure(figsize=(8,5))
		plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
		plt.xlabel('Number of Components')
		plt.ylabel('Cumulative Explained Variance')
		plt.title('PCA - Explained Variance vs. Number of Components')
		plt.grid(True)
		plt.show()
		# Prepare results string
		explained_var = np.round(pca.explained_variance_ratio_, 4)
		X_train, X_test, y_train, y_test = train_test_split(
			X_selected, y, test_size=params['test_size'], random_state=params['random_state'], stratify=y
		)
		results = (
			f"Original class counts: 1={count_a}, 0={count_b}\n"
			f"Balanced class count: {y.value_counts().to_dict()}\n"
			f"Selected features: {list(selected_features)}\n"
			f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n"
			f"Explained variance ratio (first 5): {explained_var[:5]}\n"
			f"Cumulative explained variance (first 5): {np.round(np.cumsum(pca.explained_variance_ratio_)[:5], 4)}"
		)
		results_text.config(state='normal')
		results_text.delete(1.0, tk.END)
		results_text.insert(tk.END, results)
		results_text.config(state='disabled')
		error_label.config(text="")
	except Exception as e:
		error_label.config(text=f"Error: {e}")

# Tkinter UI
root = tk.Tk()
root.title("MAGIC ML Parameter Selection")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(row=0, column=0, sticky=("N", "W", "E", "S"))

ttk.Label(mainframe, text="Test Size (0-1):").grid(row=0, column=0, sticky="W")
test_size_var = tk.StringVar(value=str(params['test_size']))
test_size_entry = ttk.Entry(mainframe, width=7, textvariable=test_size_var)
test_size_entry.grid(row=0, column=1)

ttk.Label(mainframe, text="Random State:").grid(row=1, column=0, sticky="W")
random_state_var = tk.StringVar(value=str(params['random_state']))
random_state_entry = ttk.Entry(mainframe, width=7, textvariable=random_state_var)
random_state_entry.grid(row=1, column=1)


# Feature selector
ttk.Label(mainframe, text="Number of Features:").grid(row=2, column=0, sticky="W")
n_features_var = tk.StringVar(value=str(params['n_features']))
n_features_spinbox = ttk.Spinbox(mainframe, from_=1, to=10, textvariable=n_features_var, width=7)
n_features_spinbox.grid(row=2, column=1)


# K selector
ttk.Label(mainframe, text="K for KNN:").grid(row=3, column=0, sticky="W")
k_var = tk.StringVar(value=str(params['k']))
k_spinbox = ttk.Spinbox(mainframe, from_=1, to=50, textvariable=k_var, width=7)
k_spinbox.grid(row=3, column=1)

# n_estimators
ttk.Label(mainframe, text="Number of Estimators (RF):").grid(row=4, column=0, sticky="W")
n_estimators_var = tk.StringVar(value=str(params['n_estimators']))
n_estimators_spinbox = ttk.Spinbox(mainframe, from_=10, to=150, textvariable=n_estimators_var, width=7)
n_estimators_spinbox.grid(row=4, column=1)




# Epochs selector for NN
ttk.Label(mainframe, text="Epochs (NN):").grid(row=10, column=0, sticky="W")
epochs_var = tk.StringVar(value="20")
epochs_spinbox = ttk.Spinbox(mainframe, from_=1, to=200, textvariable=epochs_var, width=7)
epochs_spinbox.grid(row=10, column=1)

# Buttons (each on its own row)
run_button = ttk.Button(mainframe, text="Preprocess Data", command=on_run)
run_button.grid(row=11, column=0, columnspan=2, pady=5, sticky="ew")
knn_button = ttk.Button(mainframe, text="Train KNN", command=on_train_knn)
knn_button.grid(row=12, column=0, columnspan=2, pady=5, sticky="ew")
logreg_button = ttk.Button(mainframe, text="Train Logistic Regression", command=on_train_logreg)
logreg_button.grid(row=13, column=0, columnspan=2, pady=5, sticky="ew")
nb_button = ttk.Button(mainframe, text="Train Naive Bayes", command=on_train_nb)
nb_button.grid(row=14, column=0, columnspan=2, pady=5, sticky="ew")
rf_button = ttk.Button(mainframe, text="Train Random Forest", command=on_train_rf)
rf_button.grid(row=15, column=0, columnspan=2, pady=5, sticky="ew")
nn_button = ttk.Button(mainframe, text="Train Neural Network", command=on_train_nn)
nn_button.grid(row=16, column=0, columnspan=2, pady=5, sticky="ew")

error_label = ttk.Label(mainframe, text="", foreground="red")
error_label.grid(row=17, column=0, columnspan=2)

# Results area at the bottom
results_text = tk.Text(root, height=14, width=70, state='disabled', wrap='word')
results_text.grid(row=1, column=0, padx=10, pady=(0,10))

root.mainloop()


