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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Constants
DATA_PATH = "/Users/jacksonhannan/Desktop/Python Projects/Customer Churn ML Project/WA_Fn-UseC_-Telco-Customer-Churn.csv"
CATEGORICAL_COLUMNS = [
    "Partner", "Dependents", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
]
NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COLUMN = "Churn"

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

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train a Random Forest model and display metrics as graphics."""
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, predictions)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_display.plot()
        plt.title("Random Forest Confusion Matrix")
        plt.show()

        # Generate classification report as a plot
        report = classification_report(y_test, predictions, output_dict=True)
        report_data = []
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                report_data.append([label] + list(metrics.values()))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        ax.axis("tight")
        ax.table(cellText=report_data,
                 colLabels=["Class"] + list(next(iter(report.values())).keys()),
                 loc="center")
        ax.set_title("Random Forest Classification Report")
        plt.show()

        # Append metrics to the text box
        text_box.insert(tk.END, "\nRandom Forest Model Trained Successfully!\n")
        text_box.insert(tk.END, "Confusion Matrix:\n")
        text_box.insert(tk.END, f"{cm}\n\n")
        text_box.insert(tk.END, "Classification Report:\n")
        text_box.insert(tk.END, f"{classification_report(y_test, predictions)}\n")
    except Exception as e:
        text_box.insert(tk.END, f"Error: {e}\n")

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train a Logistic Regression model and display metrics."""
    try:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, predictions)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_display.plot()
        plt.title("Logistic Regression Confusion Matrix")
        plt.show()

        # Generate classification report as a plot
        report = classification_report(y_test, predictions, output_dict=True)
        report_data = []
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                report_data.append([label] + list(metrics.values()))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        ax.axis("tight")
        ax.table(cellText=report_data,
                 colLabels=["Class"] + list(next(iter(report.values())).keys()),
                 loc="center")
        ax.set_title("Logistic Regression Classification Report")
        plt.show()

        # Append metrics to the text box
        text_box.insert(tk.END, "\nLogistic Regression Model Trained Successfully!\n")
        text_box.insert(tk.END, "Confusion Matrix:\n")
        text_box.insert(tk.END, f"{cm}\n\n")
        text_box.insert(tk.END, "Classification Report:\n")
        text_box.insert(tk.END, f"{classification_report(y_test, predictions)}\n")
    except Exception as e:
        text_box.insert(tk.END, f"Error: {e}\n")

def train_neural_network(X_train, X_test, y_train, y_test, epochs):
    """Train a neural network model and display metrics."""
    try:
        # Define the model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        predictions = (model.predict(X_test) > 0.5).astype(int).flatten()

        # Generate confusion matrix
        cm = confusion_matrix(y_test, predictions)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_display.plot()
        plt.title("Neural Network Confusion Matrix")
        plt.show()

        # Generate classification report as a plot
        report = classification_report(y_test, predictions, output_dict=True)
        report_data = []
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                report_data.append([label] + list(metrics.values()))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        ax.axis("tight")
        ax.table(cellText=report_data,
                 colLabels=["Class"] + list(next(iter(report.values())).keys()),
                 loc="center")
        ax.set_title("Neural Network Classification Report")
        plt.show()

        # Append metrics to the text box
        text_box.insert(tk.END, "\nNeural Network Model Trained Successfully!\n")
        text_box.insert(tk.END, f"Accuracy: {accuracy:.4f}\n")
        text_box.insert(tk.END, "Confusion Matrix:\n")
        text_box.insert(tk.END, f"{cm}\n\n")
        text_box.insert(tk.END, "Classification Report:\n")
        text_box.insert(tk.END, f"{classification_report(y_test, predictions)}\n")
    except Exception as e:
        text_box.insert(tk.END, f"Error: {e}\n")

def train_knn_model(X_train, X_test, y_train, y_test, k):
    """Train a KNN model and display metrics."""
    try:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, predictions)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_display.plot()
        plt.title("KNN Confusion Matrix")
        plt.show()

        # Generate classification report as a plot
        report = classification_report(y_test, predictions, output_dict=True)
        report_data = []
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                report_data.append([label] + list(metrics.values()))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        ax.axis("tight")
        ax.table(cellText=report_data,
                 colLabels=["Class"] + list(next(iter(report.values())).keys()),
                 loc="center")
        ax.set_title("KNN Classification Report")
        plt.show()

        # Append metrics to the text box
        text_box.insert(tk.END, "\nKNN Model Trained Successfully!\n")
        text_box.insert(tk.END, f"K: {k}\n")
        text_box.insert(tk.END, "Confusion Matrix:\n")
        text_box.insert(tk.END, f"{cm}\n\n")
        text_box.insert(tk.END, "Classification Report:\n")
        text_box.insert(tk.END, f"{classification_report(y_test, predictions)}\n")
    except Exception as e:
        text_box.insert(tk.END, f"Error: {e}\n")

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

def train_rf_model():
    try:
        n_estimators = n_estimators_slider.get()
        train_random_forest(X_train, X_test, y_train, y_test)
    except Exception as e:
        text_box.delete(1.0, tk.END)
        text_box.insert(tk.END, f"Error: {e}\n")

def train_lr_model():
    try:
        train_logistic_regression(X_train, X_test, y_train, y_test)
    except Exception as e:
        text_box.delete(1.0, tk.END)
        text_box.insert(tk.END, f"Error: {e}\n")

def train_knn():
    try:
        k = k_slider.get()
        train_knn_model(X_train, X_test, y_train, y_test, k)
    except Exception as e:
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

    # Create a frame for buttons (right side)
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.RIGHT, padx=10, pady=10, anchor="n")

    # Preprocess button
    preprocess_button = tk.Button(button_frame, text="Preprocess and Split Data", command=preprocess_and_split)
    preprocess_button.pack(fill="x", pady=5)

    # Train Random Forest button
    train_rf_button = tk.Button(button_frame, text="Train Random Forest", command=train_rf_model)
    train_rf_button.pack(fill="x", pady=5)

    # Train Logistic Regression button
    train_lr_button = tk.Button(button_frame, text="Train Logistic Regression", command=train_lr_model)
    train_lr_button.pack(fill="x", pady=5)

    # Train Neural Network button
    def train_nn_model():
        try:
            epochs = epochs_slider.get()
            train_neural_network(X_train, X_test, y_train, y_test, epochs)
        except Exception as e:
            text_box.delete(1.0, tk.END)
            text_box.insert(tk.END, f"Error: {e}\n")

    train_nn_button = tk.Button(button_frame, text="Train Neural Network", command=train_nn_model)
    train_nn_button.pack(fill="x", pady=5)

    # Train KNN button
    train_knn_button = tk.Button(button_frame, text="Train KNN", command=train_knn)
    train_knn_button.pack(fill="x", pady=5)

    # Add a button to run regressions and display graphics
    regression_button = tk.Button(button_frame, text="Run Feature Regressions", command=run_regressions_and_display)
    regression_button.pack(fill="x", pady=5)

    # Create a text box to display output (below sliders and buttons)
    text_box = scrolledtext.ScrolledText(root, width=100, height=20)
    text_box.pack(pady=10, fill="both", expand=True)

    # Run the Tkinter event loop
    root.mainloop()