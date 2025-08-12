import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import scrolledtext
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib
from tkinter import filedialog, ttk

# Constants
DATA_PATH = "/Users/jacksonhannan/Desktop/Python Projects/Customer Churn ML Project/WA_Fn-UseC_-Telco-Customer-Churn.csv"
CATEGORICAL_COLUMNS = [
    "Partner", "Dependents", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
]
NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COLUMN = "Churn"

# New global structures
MODEL_RESULTS = []  # list of dicts summarizing model performance
TRAINED_MODELS = {}  # name -> fitted pipeline/model
SEED = 42

# Ensure threshold global exists before NN wrapper uses it
if 'CURRENT_THRESHOLD' not in globals():
    CURRENT_THRESHOLD = 0.5

# Helper: build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ], remainder="drop"
)

def load_data():
    """Load the CSV file into a Pandas DataFrame."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"The file at {DATA_PATH} does not exist.")
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
    return df

def preprocess_data(df):
    """Preprocess the data: encode target, encode categorical features, and scale numerical features."""
    # Ensure columns exist in the dataset
    available_categorical = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    available_numerical = [col for col in NUMERICAL_COLUMNS if col in df.columns]

    # Remove specific columns if they exist
    columns_to_remove = ['gender', 'MultipleLines', 'PhoneService']
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')

    # Encode target column
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    # Replace missing TotalCharges with the average of rows with the same target value
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
        df['TotalCharges'] = df.groupby(TARGET_COLUMN)['TotalCharges'].transform(
            lambda x: x.fillna(x.mean())
        )

    # Drop the 'customerID' column if it exists
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # One-hot encode categorical columns
    if available_categorical:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_categorical = encoder.fit_transform(df[available_categorical])
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical, columns=encoder.get_feature_names_out(available_categorical), index=df.index
        )
    else:
        encoded_categorical_df = pd.DataFrame(index=df.index)  # Empty DataFrame if no categorical columns

    # Scale numerical columns
    if available_numerical:
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(df[available_numerical])
        scaled_numerical_df = pd.DataFrame(
            scaled_numerical, columns=available_numerical, index=df.index
        )
    else:
        scaled_numerical_df = pd.DataFrame(index=df.index)  # Empty DataFrame if no numerical columns

    # Combine processed columns
    processed_df = pd.concat([
        df.drop(columns=available_categorical + available_numerical, errors='ignore'),
        encoded_categorical_df,
        scaled_numerical_df
    ], axis=1)

    # Count the number of 0s and 1s in the target column
    if TARGET_COLUMN in df.columns:
        class_counts = df[TARGET_COLUMN].value_counts()
        print(f"Class counts before oversampling: {class_counts.to_dict()}")

        # Oversample the minority class
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        majority_df = df[df[TARGET_COLUMN] == majority_class]
        minority_df = df[df[TARGET_COLUMN] == minority_class]
        oversampled_minority_df = minority_df.sample(n=len(majority_df), replace=True, random_state=42)
        df = pd.concat([majority_df, oversampled_minority_df], axis=0).sample(frac=1, random_state=42)

        # Print class counts after oversampling
        print(f"Class counts after oversampling: {df[TARGET_COLUMN].value_counts().to_dict()}")

    print("Data preprocessing completed.")
    return processed_df

def split_data(df, test_size):
    """Split the data into train and test sets."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

class SimpleNNWrapper:
    def __init__(self, epochs_getter, seed=42):
        self.epochs_getter = epochs_getter  # function returning epochs
        self.seed = seed
        self.model = None
    def fit(self, X, y):
        try:
            import tensorflow as tf
            tf.random.set_seed(self.seed)
            from tensorflow.keras.models import Sequential  # type: ignore
            from tensorflow.keras.layers import Dense  # type: ignore
        except ImportError:
            raise ImportError("TensorFlow is required for the NeuralNet model. Please install tensorflow.")
        epochs = self.epochs_getter()
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=epochs, verbose=0)
        return self
    def predict(self, X):
        probs = self.model.predict(X, verbose=0).ravel()
        return (probs >= CURRENT_THRESHOLD).astype(int)
    def predict_proba(self, X):
        probs = self.model.predict(X, verbose=0).ravel()
        return np.vstack([1-probs, probs]).T

def run_feature_regressions(X, y):
    """Run logistic regressions for each feature and generate graphics."""
    feature_names = X.columns
    graphics = []

    for feature in feature_names:
        try:
            X_feature = X[[feature]]  # Use only the current feature
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_feature, y)
            coef = model.coef_[0][0]

            # Create a plot for the feature
            fig, ax = plt.subplots()
            ax.scatter(X_feature, y, alpha=0.5, label="Data")
            ax.plot(X_feature, model.predict_proba(X_feature)[:, 1], color="red", label="Logistic Fit")
            ax.set_title(f"Feature: {feature}\nCoefficient: {coef:.4f}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Probability of Churn")
            ax.legend()

            # Save the figure to the list
            graphics.append(fig)
        except Exception as e:
            print(f"Error processing feature {feature}: {e}")

    return graphics

def display_graphics(graphics):
    """Display graphics in a scrollable Tkinter window organized into three columns."""
    graphics_window = tk.Toplevel(root)
    graphics_window.title("Feature Regression Results")

    # Create a canvas and a scrollbar
    canvas = tk.Canvas(graphics_window)
    scrollbar = tk.Scrollbar(graphics_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    # Configure the canvas
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Pack the canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    row, col = 0, 0
    image_refs = []  # Store references to PhotoImage objects

    for fig in graphics:
        # Adjust figure size for better readability
        fig.set_size_inches(6, 4)  # Set width to 6 inches and height to 4 inches

        # Save the figure to a temporary file
        fig_path = f"temp_{row}_{col}.png"
        fig.savefig(fig_path, dpi=100)  # Increase DPI for better resolution

        # Load the image into Tkinter
        img = tk.PhotoImage(file=fig_path)
        image_refs.append(img)  # Prevent garbage collection

        # Create a canvas and display the image
        canvas_item = tk.Canvas(scrollable_frame, width=600, height=400)  # Match canvas size to figure
        canvas_item.grid(row=row, column=col, padx=5, pady=5)
        canvas_item.create_image(0, 0, anchor="nw", image=img)

        col += 1
        if col == 2:  # Move to the next row after two columns
            col = 0
            row += 1

    graphics_window.mainloop()

def preprocess_and_display():
    try:
        data = load_data()
        processed_data = preprocess_data(data)
        text_box.delete(1.0, tk.END)  # Clear the text box
        text_box.insert(tk.END, "Data preprocessing completed!\n")  # Only display success message
    except Exception as e:
        text_box.delete(1.0, tk.END)
        text_box.insert(tk.END, f"Error: {e}")

def preprocess_and_split():
    try:
        data = load_data()
        processed_data = preprocess_data(data)
        test_size = validation_slider.get()
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = split_data(processed_data, test_size)
        text_box.delete(1.0, tk.END)
        text_box.insert(tk.END, "Data preprocessing and splitting completed!\n")
    except Exception as e:
        text_box.delete(1.0, tk.END)
        text_box.insert(tk.END, f"Error: {e}\n")

def run_regressions_and_display():
    """Run logistic regressions for each feature and display graphics."""
    feature_names = X_train.columns
    graphics = []

    for feature in feature_names:
        try:
            X_feature = X_train[[feature]]  # Use only the current feature
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_feature, y_train)
            coef = model.coef_[0][0]

            # Create a plot for the feature
            fig, ax = plt.subplots()
            ax.scatter(X_feature, y_train, alpha=0.5, label="Data")
            ax.plot(X_feature, model.predict_proba(X_feature)[:, 1], color="red", label="Logistic Fit")
            ax.set_title(f"Feature: {feature}\nCoefficient: {coef:.4f}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Probability of Churn")
            ax.legend()

            # Save the figure to the list
            graphics.append(fig)
        except Exception as e:
            print(f"Error processing feature {feature}: {e}")

    display_graphics(graphics)

PROBA_CACHE = {}

# =============================================
# Helper functions required by unified train_model
# =============================================
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning without feature engineering (leave raw categoricals for pipeline).
    Operations:
    - Drop unused columns if present
    - Map target to 0/1
    - Coerce TotalCharges to numeric (handling blanks)
    - Drop rows with missing target
    - Drop customerID
    Returns cleaned dataframe preserving original feature columns for ColumnTransformer.
    """
    df = df.copy()
    # Drop explicit unused columns if they exist (legacy removals)
    drop_cols = [c for c in ['customerID', 'gender', 'MultipleLines', 'PhoneService'] if c in df.columns]
    # Handle TotalCharges blanks
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    # Encode target
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'No': 0, 'Yes': 1})
    # Impute TotalCharges by overall mean if NaN remain
    if 'TotalCharges' in df.columns:
        df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    return df

def oversample_training(X: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Random upsample minority class in training data only (no leakage)."""
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return X, y  # can't oversample
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]
    if counts.min() == counts.max():
        return X, y  # already balanced
    # indices
    idx_majority = np.where(y == majority_class)[0]
    idx_minority = np.where(y == minority_class)[0]
    n_to_sample = len(idx_majority) - len(idx_minority)
    sampled_idx = rng.choice(idx_minority, size=n_to_sample, replace=True)
    X_new = np.concatenate([X, X[sampled_idx]], axis=0)
    y_new = np.concatenate([y, y[sampled_idx]], axis=0)
    # shuffle
    shuffle_idx = rng.permutation(len(y_new))
    return X_new[shuffle_idx], y_new[shuffle_idx]

def show_confusion_matrix(model_name: str, cm: np.ndarray):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def show_classification_report(model_name: str, report_dict: dict):
    """Visualize classification report as heatmap (precision/recall/f1 for classes)."""
    # Convert to DataFrame ignoring 'accuracy', 'macro avg', 'weighted avg' for concise view
    rows = {k: v for k, v in report_dict.items() if k in ['0', '1']}
    if not rows:
        return
    df_rep = pd.DataFrame(rows).T[['precision','recall','f1-score','support']]
    plt.figure(figsize=(5,2.5))
    sns.heatmap(df_rep[['precision','recall','f1-score']].astype(float), annot=True, cmap='Greens', vmin=0, vmax=1)
    plt.title(f"{model_name} Classification Report")
    plt.tight_layout()
    plt.show()

def plot_curves(model_name: str, y_true: np.ndarray, y_proba: np.ndarray):
    """Plot ROC and Precision-Recall curves."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        fig, axes = plt.subplots(1,2, figsize=(8,3))
        axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        axes[0].plot([0,1],[0,1],'--', color='grey')
        axes[0].set_title(f"ROC - {model_name}")
        axes[0].set_xlabel('FPR')
        axes[0].set_ylabel('TPR')
        axes[0].legend()
        axes[1].plot(recall, precision, label=f"AUC={pr_auc:.3f}")
        axes[1].set_title(f"PR Curve - {model_name}")
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        text_box.insert(tk.END, f"Curve plotting error for {model_name}: {e}\n")

def plot_feature_importance(model_name: str, inner_model, feature_names):
    """Plot feature importance for RF or coefficients for LogisticRegression.
    inner_model may be the wrapped model (SimpleNNWrapper not supported).
    """
    try:
        if hasattr(inner_model, 'feature_importances_'):
            importances = inner_model.feature_importances_
            idx = np.argsort(importances)[::-1][:15]
            plt.figure(figsize=(6,4))
            sns.barplot(x=importances[idx], y=np.array(feature_names)[idx])
            plt.title(f"Top Feature Importances - {model_name}")
            plt.tight_layout()
            plt.show()
        elif hasattr(inner_model, 'coef_'):
            coefs = inner_model.coef_[0]
            idx = np.argsort(np.abs(coefs))[::-1][:15]
            plt.figure(figsize=(6,4))
            sns.barplot(x=coefs[idx], y=np.array(feature_names)[idx])
            plt.title(f"Top Coefficients - {model_name}")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        text_box.insert(tk.END, f"Feature importance error for {model_name}: {e}\n")

def train_model(model_name, model_instance):
    """Unified pipeline training with probability caching and threshold application."""
    try:
        df_local = load_data()
        df_local = basic_clean(df_local)
        test_size = validation_slider.get() if 'validation_slider' in globals() else 0.2
        X = df_local.drop(columns=[TARGET_COLUMN])
        y = df_local[TARGET_COLUMN].values
        X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=SEED
        )
        # Oversample training only
        X_train_np, y_train_np = X_train_local.values, y_train_local
        X_bal, y_bal = oversample_training(X_train_np, y_train_np)
        X_bal_df = pd.DataFrame(X_bal, columns=X_train_local.columns)

        # Build pipeline (for NN wrapper, transform first then fit wrapper on transformed data)
        if isinstance(model_instance, SimpleNNWrapper):
            fitted_prep = preprocessor.fit(X_bal_df)
            X_ready = fitted_prep.transform(X_bal_df)
            model_instance.fit(X_ready, y_bal)
            predictor = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model_instance)
            ])
        else:
            predictor = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model_instance)
            ])
            predictor.fit(X_bal_df, y_bal)

        # Probabilities / predictions
        try:
            y_proba = predictor.predict_proba(X_test_local)[:,1]
        except Exception:
            y_proba = None
        if y_proba is not None:
            y_pred = (y_proba >= CURRENT_THRESHOLD).astype(int)
        else:
            y_pred = predictor.predict(X_test_local)
        PROBA_CACHE[model_name] = {"y_true": y_test_local, "y_proba": y_proba}

        # Metrics & plots
        cm = confusion_matrix(y_test_local, y_pred)
        show_confusion_matrix(model_name, cm)
        report_dict = classification_report(y_test_local, y_pred, output_dict=True)
        show_classification_report(model_name, report_dict)
        if y_proba is not None:
            plot_curves(model_name, y_test_local, y_proba)
        # Feature importance / coefficients
        try:
            ohe = predictor.named_steps['preprocessor'].named_transformers_['cat']
            feature_names = NUMERICAL_COLUMNS + list(ohe.get_feature_names_out(CATEGORICAL_COLUMNS))
            inner = predictor.named_steps['model']
            plot_feature_importance(model_name, getattr(inner, 'model', inner), feature_names)
        except Exception:
            pass

        text_box.insert(tk.END, f"\n{model_name} (threshold {CURRENT_THRESHOLD:.2f}) trained.\n")
        text_box.insert(tk.END, f"Confusion Matrix:\n{cm}\n")
        text_box.insert(tk.END, f"Classification Report:\n{classification_report(y_test_local, y_pred)}\n")
        text_box.see(tk.END)
        TRAINED_MODELS[model_name] = predictor
    except Exception as e:
        text_box.insert(tk.END, f"Error training {model_name}: {e}\n")
        text_box.see(tk.END)

def recompute_threshold_metrics():
    if not PROBA_CACHE:
        text_box.insert(tk.END, "No probability caches. Train a model with predict_proba first.\n")
        return
    text_box.insert(tk.END, f"\nRecomputing metrics at threshold {CURRENT_THRESHOLD:.2f}\n")
    for name, cache in PROBA_CACHE.items():
        y_proba = cache['y_proba']
        y_true = cache['y_true']
        if y_proba is None:
            continue
        y_pred = (y_proba >= CURRENT_THRESHOLD).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        rep = classification_report(y_true, y_pred)
        text_box.insert(tk.END, f"Model: {name}\nCM:\n{cm}\n{rep}\n")
    text_box.see(tk.END)

def save_model():
    """Save all trained models to disk."""
    try:
        for name, model in TRAINED_MODELS.items():
            # Save each model as a joblib file
            joblib.dump(model, f"{name}_model.joblib")
        text_box.insert(tk.END, "Models saved successfully.\n")
    except Exception as e:
        text_box.insert(tk.END, f"Error saving models: {e}\n")

def load_model():
    """Load models from disk and update the TRAINED_MODELS dictionary."""
    try:
        # Clear the existing models
        TRAINED_MODELS.clear()

        # List all joblib files in the current directory
        for file in os.listdir("."):
            if file.endswith("_model.joblib"):
                # Load each model and add it to the dictionary
                model_name = file[:-10]  # Remove "_model" from the filename
                model = joblib.load(file)
                TRAINED_MODELS[model_name] = model

        text_box.insert(tk.END, "Models loaded successfully.\n")
    except Exception as e:
        text_box.insert(tk.END, f"Error loading models: {e}\n")

def show_summary():
    """Show a summary of all trained models."""
    try:
        if not TRAINED_MODELS:
            text_box.insert(tk.END, "No models trained yet.\n")
            return

        for name, model in TRAINED_MODELS.items():
            text_box.insert(tk.END, f"Model: {name}\n")
            text_box.insert(tk.END, f"Parameters: {model.get_params()}\n")
            text_box.insert(tk.END, "\n")

        text_box.insert(tk.END, "End of summary.\n")
    except Exception as e:
        text_box.insert(tk.END, f"Error displaying summary: {e}\n")

def adjust_threshold(value):
    """Update global CURRENT_THRESHOLD and optionally auto-recompute cached metrics."""
    global CURRENT_THRESHOLD
    try:
        CURRENT_THRESHOLD = float(value)
        text_box.insert(tk.END, f"Threshold set to {CURRENT_THRESHOLD:.2f}. Use 'Recompute Threshold Metrics' to refresh.")
        text_box.insert(tk.END, "\n")
        text_box.see(tk.END)
    except Exception as e:
        text_box.insert(tk.END, f"Error adjusting threshold: {e}\n")

if __name__ == "__main__":
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Customer Churn Preprocessing and Modeling")

    # Create a frame for sliders (left side)
    slider_frame = tk.Frame(root)
    slider_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor="n")

    # Validation split slider
    validation_label = tk.Label(slider_frame, text="Validation Split Fraction:")
    validation_label.pack(anchor="w", pady=5)
    validation_slider = tk.Scale(slider_frame, from_=0.1, to=0.5, resolution=0.05, orient=tk.HORIZONTAL)
    validation_slider.set(0.2)
    validation_slider.pack(anchor="w", pady=5)

    # n_estimators slider
    n_estimators_label = tk.Label(slider_frame, text="Number of Estimators (RF):")
    n_estimators_label.pack(anchor="w", pady=5)
    n_estimators_slider = tk.Scale(slider_frame, from_=10, to=200, resolution=10, orient=tk.HORIZONTAL)
    n_estimators_slider.set(100)
    n_estimators_slider.pack(anchor="w", pady=5)

    # Epochs slider
    epochs_label = tk.Label(slider_frame, text="Number of Epochs (NN):")
    epochs_label.pack(anchor="w", pady=5)
    epochs_slider = tk.Scale(slider_frame, from_=10, to=100, resolution=10, orient=tk.HORIZONTAL)
    epochs_slider.set(50)
    epochs_slider.pack(anchor="w", pady=5)

    # K slider
    k_label = tk.Label(slider_frame, text="Number of Neighbors (KNN):")
    k_label.pack(anchor="w", pady=5)
    k_slider = tk.Scale(slider_frame, from_=1, to=50, resolution=1, orient=tk.HORIZONTAL)
    k_slider.set(5)
    k_slider.pack(anchor="w", pady=5)

    # Threshold slider
    threshold_label = tk.Label(slider_frame, text="Classification Threshold:")
    threshold_label.pack(anchor='w', pady=5)
    threshold_slider = tk.Scale(slider_frame, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL, command=adjust_threshold)
    threshold_slider.set(0.5)
    threshold_slider.pack(anchor='w', pady=5)

    # Create a frame for buttons (right side)
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.RIGHT, padx=10, pady=10, anchor="n")

    # Preprocess button
    preprocess_button = tk.Button(button_frame, text="Preprocess and Split Data", command=preprocess_and_split)
    preprocess_button.pack(fill="x", pady=5)

    # Grouped pipeline training buttons
    pipeline_frame = ttk.LabelFrame(button_frame, text="Pipeline Models")
    pipeline_frame.pack(fill='x', pady=8)

    pipeline_rf_btn = tk.Button(pipeline_frame, text="RandomForest", command=lambda: train_model('RandomForest', RandomForestClassifier(random_state=SEED)))
    pipeline_rf_btn.pack(fill='x', pady=2)
    pipeline_lr_btn = tk.Button(pipeline_frame, text="LogisticReg", command=lambda: train_model('LogisticRegression', LogisticRegression(max_iter=1000, random_state=SEED)))
    pipeline_lr_btn.pack(fill='x', pady=2)
    pipeline_knn_btn = tk.Button(pipeline_frame, text="KNN", command=lambda: train_model('KNN', KNeighborsClassifier(n_neighbors=k_slider.get())))
    pipeline_knn_btn.pack(fill='x', pady=2)
    pipeline_nn_btn = tk.Button(pipeline_frame, text="NeuralNet", command=lambda: train_model('NeuralNet', SimpleNNWrapper(lambda: epochs_slider.get(), seed=SEED)))
    pipeline_nn_btn.pack(fill='x', pady=2)

    def train_all():
        train_model('RandomForest', RandomForestClassifier(random_state=SEED))
        train_model('LogisticRegression', LogisticRegression(max_iter=1000, random_state=SEED))
        train_model('KNN', KNeighborsClassifier(n_neighbors=k_slider.get()))
        train_model('NeuralNet', SimpleNNWrapper(lambda: epochs_slider.get(), seed=SEED))
    train_all_btn = tk.Button(pipeline_frame, text="Train All", command=train_all)
    train_all_btn.pack(fill='x', pady=4)

    # Save / Load buttons
    save_btn = tk.Button(button_frame, text="Save Models", command=save_model)
    save_btn.pack(fill='x', pady=3)
    load_btn = tk.Button(button_frame, text="Load Models", command=load_model)
    load_btn.pack(fill='x', pady=3)

    # Summary button
    summary_btn = tk.Button(button_frame, text="Show Summary", command=show_summary)
    summary_btn.pack(fill='x', pady=3)
    # Recompute threshold metrics button
    recompute_btn = tk.Button(button_frame, text="Recompute Threshold Metrics", command=recompute_threshold_metrics)
    recompute_btn.pack(fill='x', pady=3)
    # Create a text box to display output (below sliders and buttons)
    text_box = scrolledtext.ScrolledText(root, width=100, height=20)
    text_box.pack(pady=10, fill="both", expand=True)

    # Now that text_box exists, print ready message
    text_box.insert(tk.END, "Ready. Use pipeline buttons for improved workflow.\n")
    text_box.see(tk.END)

    # Run the Tkinter event loop
    root.mainloop()