from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from sklearn.model_selection import train_test_split

# fetch dataset 
def load_mushroom_dataset():
	"""
	Fetches the UCI Mushroom dataset and returns features, targets, metadata, and variable info.
	Returns:
		X (pd.DataFrame): Features
		y (pd.DataFrame): Targets
		metadata (dict): Dataset metadata
		variables (dict): Variable information
	"""
	mushroom = fetch_ucirepo(id=73)
	X = mushroom.data.features
	y = mushroom.data.targets
	metadata = mushroom.metadata
	variables = mushroom.variables
	return X, y, metadata, variables

# Function to encode all categorical and binary features in a DataFrame
def encode_features(df):
	"""
	Encodes all categorical and binary features in the given DataFrame using one-hot encoding.
	Args:
		df (pd.DataFrame): Input DataFrame with categorical/binary features.
	Returns:
		pd.DataFrame: Encoded DataFrame with all features numeric.
	"""
	df_encoded = df.copy()
	for col in df_encoded.columns:
		le = LabelEncoder()
		df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
	return df_encoded

def mushroom_nn_model(input_shape):
    """
    Creates a simple neural network model for mushroom classification.
    Args:
        input_shape (int): Number of input features.
    Returns:
        tf.keras.Model: Compiled neural network model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main execution:
if __name__ == "__main__":
	X, y, metadata, variables = load_mushroom_dataset()
	print(metadata)
	print(variables)
	# Encode features and target
	X_encoded = encode_features(X)
	y_encoded = LabelEncoder().fit_transform(y.iloc[:,0])  # Assuming single target column
	print(X_encoded.head())
	# Train/test split
	X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
	# Build and train model
	input_shape = X_encoded.shape[1]
	model = mushroom_nn_model(input_shape)
	model.summary()
	model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
	# Predict
	y_pred_prob = model.predict(X_test)
	y_pred = (y_pred_prob > 0.5).astype(int)
	# Classification report
	print("\nClassification Report:")
	print(classification_report(y_test, y_pred))
	print("Confusion Matrix:")
	print(confusion_matrix(y_test, y_pred))
	print("Accuracy Score:")
	print(accuracy_score(y_test, y_pred))
