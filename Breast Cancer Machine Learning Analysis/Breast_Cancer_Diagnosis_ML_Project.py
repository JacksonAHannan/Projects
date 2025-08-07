import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    # Load the dataset
    df = pd.read_csv('/Users/jacksonhannan/Downloads/Breast Cancer Dataset/Breast_cancer_dataset.csv')

    # Encode diagnosis
    df['diagnosis_encoded'] = df['diagnosis'].map({'B': 0, 'M': 1})

    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('diagnosis_encoded', errors='ignore')

    # Correlations
    correlations = df[numerical_cols].corrwith(df['diagnosis_encoded']).abs().sort_values(ascending=False)

    # Ask user for number of top features
    while True:
        try:
            n_features = int(input("Select number of top features to use (3, 5, 7, 9): "))
            if n_features in [3, 5, 7, 9]:
                break
            else:
                print("Please enter 3, 5, 7, or 9.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    selected_cols = correlations.index[:n_features]
    X = df[selected_cols]
    y = df['diagnosis_encoded']

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy with top {n_features} features: {acc:.4f}")

if __name__ == "__main__":
    main()