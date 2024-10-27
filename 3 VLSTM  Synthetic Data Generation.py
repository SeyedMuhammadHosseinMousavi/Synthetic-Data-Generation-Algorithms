%reset -f
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Start time for the overall process
start_time = time.time()

#---------------------------------------------
# Load data
data = pd.read_csv('Physiological Emotion Eecognition EEG Data.csv')
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values
#---------------------------------------------

# Normalize features
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))

# Separate 20% of data from each class for synthetic generation and model training
train_features = []
train_labels = []
remaining_features = []
remaining_labels = []

unique_labels = np.unique(labels_encoded)
for label in unique_labels:
    label_data = features_normalized[labels_encoded == label]
    label_targets = labels_encoded[labels_encoded == label]
    train_x, remaining_x, train_y, remaining_y = train_test_split(label_data, label_targets, test_size=0.8, random_state=42)
    train_features.append(train_x)
    train_labels.append(train_y)
    remaining_features.append(remaining_x)
    remaining_labels.append(remaining_y)

train_features = np.vstack(train_features)
train_labels = np.concatenate(train_labels)
remaining_features = np.vstack(remaining_features)
remaining_labels = np.concatenate(remaining_labels)

# Reshape features for LSTM model
train_features = train_features.reshape((train_features.shape[0], train_features.shape[1], 1))
remaining_features = remaining_features.reshape((remaining_features.shape[0], remaining_features.shape[1], 1))

# Parameters
#---------------------------------------------
input_dim = train_features.shape[1]
epochs = 100
batch_size = 32
#---------------------------------------------

# Build the LSTM model with variational dropout
def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    # Add dropout to LSTM layers
    x = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    x = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
    output = Dense(input_shape[0], activation='linear')(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_lstm_model(train_features.shape[1:])

# Train the model on the remaining 80% data
model.fit(train_features, train_features, epochs=epochs, batch_size=batch_size, verbose=1)

# Generate synthetic data with corresponding labels
def generate_synthetic_data_with_labels(model, data, labels, num_samples):
    sampled_indices = np.random.choice(np.arange(len(data)), size=num_samples, replace=True)
    sampled_data = data[sampled_indices]
    sampled_labels = labels[sampled_indices]
    
    # Predict synthetic data using the model
    synthetic_data = model.predict(sampled_data)

    # Generate noise to add to the synthetic data
    noise = np.random.normal(0, 0.1, synthetic_data.shape)

    # Add noise to the synthetic data
    synthetic_data_noisy = synthetic_data + noise

    return synthetic_data_noisy, sampled_labels

# Usage-------------------------------
num_samples = 1500
synthetic_data, synthetic_labels = generate_synthetic_data_with_labels(model, remaining_features, remaining_labels, num_samples)

# Saving to CSV------------------------
synthetic_df = pd.DataFrame(synthetic_data, columns=[f'Feature_{i+1}' for i in range(synthetic_data.shape[1])])
synthetic_df['Label'] = synthetic_labels
synthetic_df.to_csv('LSTM Synthetic_Data.csv', index=False)

# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

# XGB Classifier----------------------
from xgboost import XGBClassifier

# Flatten the synthetic data if it's 3D (if the last dimension is features)
if len(synthetic_data.shape) == 3:
    synthetic_data = synthetic_data.reshape(synthetic_data.shape[0], -1)

# Flatten train_features if it's still 3D, for consistency with XGBoost input requirements
train_features = train_features.reshape(train_features.shape[0], -1)
remaining_features = remaining_features.reshape(remaining_features.shape[0], -1)

# Combine remaining original and synthetic data, ensuring all are 2-dimensional
combined_features = np.vstack((remaining_features, synthetic_data))
combined_labels = np.concatenate((remaining_labels, synthetic_labels))

# Function to run experiments, ensuring XGBoost receives the correct input shape
def run_experiments(data, labels, n_runs=5):
    accuracies = []
    all_confusion_matrices = []
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred) * 100)  # Convert accuracy to percentage before appending

        # Store confusion matrices for each run
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
        all_confusion_matrices.append(cm)
        
        # Store last run's y_test and y_pred for final reporting
        if run == n_runs - 1:
            final_y_test = y_test
            final_y_pred = y_pred

    # Average confusion matrix
    avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    
    # Print the confusion matrix
    print("Average Confusion Matrix:\n", avg_confusion_matrix)

    # Ensure target names are strings
    target_names = label_encoder.classes_.astype(str)

    # Classification report for the last run
    print("\nClassification Report:\n", classification_report(final_y_test, final_y_pred, target_names=target_names))

    mean_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies) / 100  # Return standard deviation as a proportion
    return mean_accuracy, std_deviation

# Running experiments on synthetic data
synthetic_accuracy, synthetic_std = run_experiments(synthetic_data, synthetic_labels)
print(f"Synthetic Data - Average Accuracy: {synthetic_accuracy:.2f}%, Std Dev: {synthetic_std:.4f}")

# Running experiments on combined data
combined_accuracy, combined_std = run_experiments(combined_features, combined_labels)
print(f"Combined Data - Average Accuracy: {combined_accuracy:.2f}%, Std Dev: {combined_std:.4f}")

# MSE--------------------------------------
from sklearn.metrics import mean_squared_error
if len(synthetic_data.shape) == 3:
    synthetic_data = synthetic_data.reshape(synthetic_data.shape[0], -1)
# Ensure synthetic data and original data (features_normalized) are the same size
min_size = min(features_normalized.shape[0], synthetic_data.shape[0])
# Truncate both datasets to the minimum size for a fair comparison
original_truncated = features_normalized[:min_size]
synthetic_truncated = synthetic_data[:min_size]
# Calculate the MSE between the truncated datasets
mse_value = mean_squared_error(original_truncated, synthetic_truncated)
print(f"Mean Squared Error between Original and Synthetic Data: {mse_value:.4f}")



# ---------------------------------------------
# SHAP (Original and Synthetic)
# ---------------------------------------------
# Train XGBClassifier models on original and synthetic data if they haven't been trained before
original_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
original_model.fit(train_features, train_labels)

synthetic_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
synthetic_model.fit(synthetic_data, synthetic_labels)

# SHAP analysis function for a trained model and dataset
def shap_analysis(data, model, title=""):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    # Handle multiclass data by selecting SHAP values for the first class (adjust as needed)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Plot SHAP summary plot
    plt.figure()  # Create a new figure for each SHAP plot
    shap.summary_plot(shap_values, data, feature_names=[f"Feature_{i}" for i in range(data.shape[1])], show=True)
    plt.title(f"SHAP Summary Plot for {title} Data")
    plt.show()

    # Print the top 10 important features by SHAP value
    shap_importance = np.abs(shap_values).mean(axis=0)
    important_features = sorted(
        [(f"Feature_{i}", shap_importance[i]) for i in range(data.shape[1])],
        key=lambda x: x[1],
        reverse=True
    )
    print(f"\nTop important features for {title} data:")
    for feature, importance in important_features[:10]:  # Display top 10 features
        print(f"{feature}: {importance:.4f}")

# Run SHAP analysis on original and synthetic data
print("SHAP Analysis for Original Data")
shap_analysis(train_features, original_model, title="Original")

print("\nSHAP Analysis for Synthetic Data")
shap_analysis(synthetic_data, synthetic_model, title="Synthetic")

# ---------------------------------------------
# Save SHAP summary plots as high-resolution JPG and PDF
def save_shap_plot(data, model, title=""):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    
    # Handle multiclass by using the first class 
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Create SHAP summary plot and save
    plt.figure()  # Start a new figure
    shap.summary_plot(shap_values, data, feature_names=[f"Feature_{i}" for i in range(data.shape[1])], show=False)
    
    # Save as high-resolution JPG and PDF
    plt.title(f"SHAP Summary Plot for {title} Data")
    plt.savefig(f"LSTM_{title}_SHAP_Summary_Plot.jpg", format="jpg", dpi=300, bbox_inches="tight")
    plt.savefig(f"LSTM_{title}_SHAP_Summary_Plot.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to prevent display or memory issues

# Save SHAP plots for both original and synthetic data
save_shap_plot(train_features, original_model, title="Original")
save_shap_plot(synthetic_data, synthetic_model, title="Synthetic")

# ---------------------------------------------
# Calculate and print similarity percentage of top features
def calculate_feature_similarity(original_features, synthetic_features, top_n=10):
    # Extract the top N features based on rank
    original_top_features = [feature for feature, _ in original_features[:top_n]]
    synthetic_top_features = [feature for feature, _ in synthetic_features[:top_n]]
    
    # Find the intersection of top features
    common_features = set(original_top_features).intersection(synthetic_top_features)
    similarity_percentage = (len(common_features) / top_n) * 100

    print(f"\nSimilarity of top {top_n} important features between original and synthetic data: {similarity_percentage:.2f}%")
    print("Common important features:", common_features)

# Capture top important features for both original and synthetic data
def get_top_important_features(data, model, top_n=10):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    
    if isinstance(shap_values, list):  # Handle multiclass
        shap_values = shap_values[0]

    shap_importance = np.abs(shap_values).mean(axis=0)
    important_features = sorted(
        [(f"Feature_{i}", shap_importance[i]) for i in range(data.shape[1])],
        key=lambda x: x[1],
        reverse=True
    )
    return important_features[:top_n]

# Get the top features for both datasets
original_top_features = get_top_important_features(train_features, original_model)
synthetic_top_features = get_top_important_features(synthetic_data, synthetic_model)

# Calculate and print the similarity percentage
calculate_feature_similarity(original_top_features, synthetic_top_features, top_n=10)



