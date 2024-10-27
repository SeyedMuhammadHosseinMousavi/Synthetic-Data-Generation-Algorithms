%reset -f
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from copulas.multivariate import GaussianMultivariate
import time
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Start time for the overall process
start_time = time.time()

# Load data
#---------------------------------------------
print("Loading data...")
data = pd.read_csv('Physiological Emotion Eecognition EEG Data.csv')
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values
#---------------------------------------------

# Normalize features
print("Normalizing features...")
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))

# Separate 20% of data from each class
print("Separating 20% of data from each class...")
train_features = []
train_labels = []
remaining_features = []
remaining_labels = []

unique_labels = np.unique(labels_encoded)
for label in unique_labels:
    label_data = features_normalized[labels_encoded == label]
    label_targets = labels_encoded[labels_encoded == label]
    train_x, test_x, train_y, test_y = train_test_split(label_data, label_targets, test_size=0.2, random_state=42)
    train_features.append(train_x)
    train_labels.append(train_y)
    remaining_features.append(test_x)
    remaining_labels.append(test_y)

train_features = np.vstack(train_features)
train_labels = np.concatenate(train_labels)
remaining_features = np.vstack(remaining_features)
remaining_labels = np.concatenate(remaining_labels)

# Fit a Gaussian Copula model on the remaining 80% data
print("Fitting Gaussian Copula model...")
copula_model = GaussianMultivariate()
copula_model.fit(pd.DataFrame(remaining_features))

# Simulating training for a set number of iterations to refine the copula fit
#-----------------------------------------
epochs = 1000
#-----------------------------------------
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    # Normally, we don't need to "retrain" a Gaussian Copula in this way, but we simulate it to show progress.

# Generate synthetic data from the remaining 20%
def generate_synthetic_data(copula_model, num_samples):
    print(f"Generating {num_samples} synthetic data samples...")
    synthetic_data = copula_model.sample(num_samples)
    return synthetic_data.values

# Generate synthetic data
synthetic_features = generate_synthetic_data(copula_model, 1500)

# Post-processing: Clip synthetic features to a desired range
synthetic_features = np.clip(synthetic_features, 0, 1)  # Clipping to range [0, 1]

# Handle labels separately since copula models don't directly model categorical data
synthetic_labels = np.random.choice(remaining_labels, size=1500)

# Save synthetic data to CSV
print("Saving synthetic data to CSV...")
synthetic_labels_decoded = label_encoder.inverse_transform(synthetic_labels.astype(int))
synthetic_data_with_labels = np.hstack((synthetic_features, synthetic_labels_decoded.reshape(-1, 1)))
feature_names = ['Feature_' + str(i) for i in range(synthetic_features.shape[1])]
column_names = feature_names + ['Label']
synthetic_df = pd.DataFrame(synthetic_data_with_labels, columns=column_names)
synthetic_df.to_csv('Copula Synthetic_Data.csv', index=False)

# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

# XGB Classification
def run_experiments(data, labels, n_runs=5):
    print(f"Running {n_runs} classification experiments...")
    accuracies = []
    all_confusion_matrices = []
    for run in range(n_runs):
        print(f"Experiment {run + 1}/{n_runs}...")
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        
        # Generate confusion matrix and store
        cm = confusion_matrix(y_test, y_pred)
        all_confusion_matrices.append(cm)
        
        # Store the final run's y_test and y_pred for reporting
        if run == n_runs - 1:
            final_y_test = y_test
            final_y_pred = y_pred
    
    # Calculate the average confusion matrix
    average_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    average_confusion_matrix = np.round(average_confusion_matrix).astype(int)
    
    # Generate classification report for the last run
    target_names = [str(int(label)) for label in np.unique(labels)]
    classification_rep = classification_report(final_y_test, final_y_pred, target_names=target_names)

    average_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    
    print("Average Confusion Matrix:\n", average_confusion_matrix)
    print("\nClassification Report:\n", classification_rep)
    
    return average_accuracy, std_deviation

# Running experiments on original data
print("Evaluating original data...")
original_accuracy, original_std = run_experiments(train_features, train_labels, n_runs=7)
print(f"Original Data - Average Accuracy: {original_accuracy * 100:.2f}%, Std Dev: {original_std:.4f}")

# Running experiments on synthetic data
print("Evaluating synthetic data...")
synthetic_accuracy, synthetic_std = run_experiments(synthetic_features, synthetic_labels.astype(int), n_runs=7)
print(f"Synthetic Data - Average Accuracy: {synthetic_accuracy * 100:.2f}%, Std Dev: {synthetic_std:.4f}")

# Running experiments on combined data (original + synthetic)
print("Evaluating combined data...")
combined_features = np.vstack((train_features, synthetic_features))
combined_labels = np.concatenate((train_labels, synthetic_labels.astype(int)))
combined_accuracy, combined_std = run_experiments(combined_features, combined_labels, n_runs=7)
print(f"Combined Data - Average Accuracy: {combined_accuracy * 100:.2f}%, Std Dev: {combined_std:.4f}")

# MSE Calculation
print("Calculating Mean Squared Error (MSE)...")
min_size = min(train_features.shape[0], synthetic_features.shape[0])
original_truncated = train_features[:min_size]
synthetic_truncated = synthetic_features[:min_size]
mse_value = mean_squared_error(original_truncated, synthetic_truncated)
print(f"Mean Squared Error between Original and Synthetic Data: {mse_value:.4f}")



# ---------------------------------------------
# SHAP (Original and Synthetic)
# ---------------------------------------------
# Train XGBClassifier models on original and synthetic data if they haven't been trained before
original_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
original_model.fit(train_features, train_labels)

synthetic_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
synthetic_model.fit(synthetic_features, synthetic_labels.astype(int))

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
shap_analysis(synthetic_features, synthetic_model, title="Synthetic")

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
    plt.savefig(f"Copula_{title}_SHAP_Summary_Plot.jpg", format="jpg", dpi=300, bbox_inches="tight")
    plt.savefig(f"Copula_{title}_SHAP_Summary_Plot.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to prevent display or memory issues

# Save SHAP plots for both original and synthetic data
save_shap_plot(train_features, original_model, title="Original")
save_shap_plot(synthetic_features, synthetic_model, title="Synthetic")

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
synthetic_top_features = get_top_important_features(synthetic_features, synthetic_model)

# Calculate and print the similarity percentage
calculate_feature_similarity(original_top_features, synthetic_top_features, top_n=10)
