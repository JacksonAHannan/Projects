import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

def load_data():
    # Load the dataset
    df = pd.read_csv('/Users/jacksonhannan/Downloads/Breast Cancer Machine Learning Analysis/Breast_cancer_dataset.csv')

    # Encode diagnosis
    df['diagnosis_encoded'] = df['diagnosis'].map({'B': 0, 'M': 1})

    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('diagnosis_encoded', errors='ignore')

    return df, numerical_cols

def regression_analysis_on_scaled_data(df, numerical_cols, target_col='diagnosis_encoded'):
    """
    Runs regression analysis on scaled numerical data and returns variables most correlated with the target.
    Args:
        df (pd.DataFrame): DataFrame containing features and target.
        numerical_cols (list): List of numerical feature columns.
        target_col (str): Name of the encoded diagnosis column.
    Returns:
        pd.Series: Sorted absolute correlations between each variable and the target.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numerical_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=numerical_cols)
    scaled_df[target_col] = df[target_col].values
    correlations = scaled_df[numerical_cols].corrwith(scaled_df[target_col]).abs().sort_values(ascending=False)
    print("Most correlated variables with diagnosis (on scaled data):")
    print(correlations)
    return correlations

def plot_top3_3d_scatter(df, correlations, target_col='diagnosis_encoded'):
    """
    Plots a 3D scatterplot of the top three most correlated variables to the encoded diagnosis.
    Points are colored blue for 0 and red for 1.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        correlations (pd.Series): Correlations sorted in descending order.
        target_col (str): Name of the encoded diagnosis column.
    """
    top3 = correlations.index[:3]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = df[target_col].map({0: 'blue', 1: 'red'})
    scatter = ax.scatter(df[top3[0]], df[top3[1]], df[top3[2]],
                        c=colors, alpha=0.7, edgecolor='k')
    ax.set_xlabel(top3[0])
    ax.set_ylabel(top3[1])
    ax.set_zlabel(top3[2])
    ax.set_title('3D Scatterplot of Top 3 Correlated Features')
    # Custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Diagnosis 0', markerfacecolor='blue', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='Diagnosis 1', markerfacecolor='red', markersize=10)]
    ax.legend(handles=legend_elements, title="Diagnosis Encoded")
    plt.show()
    input("Press Enter to exit plot window...")

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save model and scaler
    # joblib.dump(model, 'trained_model.joblib')
    # joblib.dump(scaler, 'scaler.joblib')
    print("Model and scaler saved as 'trained_model.joblib' and 'scaler.joblib'.")

    return model, scaler

def predict_new_data(model, scaler, feature_cols, csv_path=None, df=None):
    """
    Predicts diagnosis for new undiagnosed data using the trained model and scaler.
    Args:
        model: Trained sklearn model.
        scaler: Fitted sklearn scaler.
        feature_cols: List of feature columns used for training.
        csv_path: Path to new data CSV (optional).
        df: DataFrame of new data (optional).
    Returns:
        np.ndarray: Predicted diagnosis (0=Benign, 1=Malignant)
    """
    if csv_path:
        new_df = pd.read_csv(csv_path)
    elif df is not None:
        new_df = df.copy()
    else:
        raise ValueError("Provide either csv_path or df for new data.")
    X_new = new_df[feature_cols]
    X_new_scaled = scaler.transform(X_new)
    preds = model.predict(X_new_scaled)
    new_df['predicted_diagnosis'] = preds
    print("Predictions for new data:")
    print(new_df[[*feature_cols, 'predicted_diagnosis']])
    return preds



if __name__ == "__main__":
    df, numerical_cols = load_data()
    
    # Run regression analysis on scaled data
    correlations = regression_analysis_on_scaled_data(df, numerical_cols)

    # Train the model using the top correlated features
    top_features = correlations.index[:3]
    X = df[top_features]
    y = df['diagnosis_encoded']
    model, scaler = train_model(X, y)
    print("Model training complete.")

    # Example: Predict on new undiagnosed data (uncomment and set path to use)
    # loaded_model = joblib.load('trained_model.joblib')
    # loaded_scaler = joblib.load('scaler.joblib')
    # predict_new_data(loaded_model, loaded_scaler, list(top_features), csv_path='path_to_new_data.csv')

    # Plot the top 3 correlated variables in a 3D scatter plot
    plot_top3_3d_scatter(df, correlations)
