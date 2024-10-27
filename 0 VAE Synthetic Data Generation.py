 %reset -f
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
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

# Prepare labels for training
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)
num_classes = labels_categorical.shape[1]

# Split data into 20% for VAE training and 80% for later use
sng_train_data = []
sng_train_labels = []
remaining_data = []
remaining_labels = []

unique_labels = np.unique(labels_encoded)
for label in unique_labels:
    label_data = features_normalized[labels_encoded == label]
    label_target = labels_categorical[labels_encoded == label]
    sng_x, remaining_x, sng_y, remaining_y = train_test_split(label_data, label_target, test_size=0.8, random_state=42)
    sng_train_data.append(sng_x)
    sng_train_labels.append(sng_y)
    remaining_data.append(remaining_x)
    remaining_labels.append(remaining_y)

sng_train_data = np.vstack(sng_train_data)
sng_train_labels = np.vstack(sng_train_labels)
remaining_data = np.vstack(remaining_data)
remaining_labels = np.vstack(remaining_labels)

# VAE Components
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_cvae_encoder(latent_dim, features_shape, num_classes):
    feature_input = Input(shape=(features_shape,))
    label_input = Input(shape=(num_classes,))
    x = Concatenate()([feature_input, label_input])
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    return Model([feature_input, label_input], [z_mean, z_log_var, z]), feature_input, label_input

def build_cvae_decoder(latent_dim, features_shape, num_classes):
    latent_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(num_classes,))
    x = Concatenate()([latent_input, label_input])
    x = Dense(512, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(features_shape, activation='tanh')(x)
    return Model([latent_input, label_input], outputs)

# Start models
latent_dim = 1000
encoder, feature_input, label_input = build_cvae_encoder(latent_dim, sng_train_data.shape[1], num_classes)
decoder = build_cvae_decoder(latent_dim, sng_train_data.shape[1], num_classes)

# Connect the encoder and decoder
z_mean, z_log_var, z = encoder([feature_input, label_input])
outputs = decoder([z, label_input])
vae = Model([feature_input, label_input], outputs)

# Define VAE loss
reconstruction_loss = binary_crossentropy(feature_input, outputs) * sng_train_data.shape[1]
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1) * -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(0.0002, 0.5))

# Training the CVAE
def train_cvae(epochs, batch_size=32):
    vae.fit([sng_train_data, sng_train_labels], sng_train_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1)

#----------------------------------
# Train the CVAE
train_cvae(epochs=10, batch_size=32)
#----------------------------------

# Generate synthetic data
num_samples = 1500
#----------------------------------

z_sample = np.random.normal(0, 0.1, (num_samples, latent_dim))
random_label_indices = np.random.randint(0, num_classes, num_samples)
synthetic_labels = to_categorical(random_label_indices, num_classes)
synthetic_samples = decoder.predict([z_sample, synthetic_labels])

# Convert labels back to original encoding if necessary for saving or further processing
synthetic_labels_decoded = label_encoder.inverse_transform(random_label_indices)
synthetic_labels_encoded = label_encoder.transform(synthetic_labels_decoded)
synthetic_data_with_labels = np.hstack((synthetic_samples, synthetic_labels_encoded.reshape(-1, 1)))

# Save to CSV (optional step)
synthetic_df = pd.DataFrame(synthetic_data_with_labels, columns=[f'Feature_{i}' for i in range(synthetic_samples.shape[1])] + ['Label'])
synthetic_df.to_csv('VAE Synthetic_Data.csv', index=False)

# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

# XGB Classification
def run_experiments(data, labels, n_runs=5):
    accuracies = []
    all_confusion_matrices = []
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=None)
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        
        # Store confusion matrices for each run
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
        all_confusion_matrices.append(cm)
        
        # Store last run's y_test and y_pred for final reporting
        if run == n_runs - 1:
            final_y_test = y_test
            final_y_pred = y_pred

    mean_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    
    # Average confusion matrix
    avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    
    # Print the confusion matrix
    print("Average Confusion Matrix:\n", avg_confusion_matrix)

    # Ensure target names are strings
    target_names = label_encoder.classes_.astype(str)

    # Classification report for the last run
    print("\nClassification Report:\n", classification_report(final_y_test, final_y_pred, target_names=target_names))
    
    return mean_accuracy, std_deviation

# Performance on original data
original_mean_acc, original_std_dev = run_experiments(features_normalized, labels_encoded)
print(f"Original Data - Average Accuracy: {original_mean_acc * 100:.2f}%, Std Dev: {original_std_dev:.2f}")

# Performance on synthetic data
synthetic_mean_acc, synthetic_std_dev = run_experiments(synthetic_samples, synthetic_labels_encoded)
print(f"Synthetic Data - Average Accuracy: {synthetic_mean_acc * 100:.2f}%, Std Dev: {synthetic_std_dev:.2f}")

# Performance on combined data (80% remaining original + synthetic)
combined_features = np.vstack((remaining_data, synthetic_samples))
combined_labels = np.concatenate((label_encoder.inverse_transform(np.argmax(remaining_labels, axis=1)), synthetic_labels_decoded))
combined_labels_encoded = label_encoder.transform(combined_labels)

combined_mean_acc, combined_std_dev = run_experiments(combined_features, combined_labels_encoded)
print(f"Combined Data - Average Accuracy: {combined_mean_acc * 100:.2f}%, Std Dev: {combined_std_dev:.2f}")

# MSE Calculation
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def calculate_average_mse(original_data, synthetic_data):
    min_samples = min(original_data.shape[0], synthetic_data.shape[0])
    original_data = original_data[:min_samples]
    synthetic_data = synthetic_data[:min_samples]
    if original_data.shape[1] != synthetic_data.shape[1]:
        raise ValueError("The number of features in original and synthetic data must be the same to calculate MSE.")
    
    original_data_normalized = normalize_data(original_data)
    synthetic_data_normalized = normalize_data(synthetic_data)
    
    mse_values = np.mean((original_data_normalized - synthetic_data_normalized) ** 2, axis=1)
    average_mse = np.mean(mse_values)
    return average_mse

average_mse = calculate_average_mse(features_normalized, synthetic_samples)
print("Average MSE between normalized original and synthetic datasets:", average_mse)

# ---------------------------------------------
# SHAP (Original and Synthetic)
# ---------------------------------------------
# Train XGBClassifier models on original and synthetic data if they haven't been trained before
original_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
original_model.fit(features_normalized, labels_encoded)

synthetic_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
synthetic_model.fit(synthetic_samples, synthetic_labels_encoded)

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
shap_analysis(features_normalized, original_model, title="Original")

print("\nSHAP Analysis for Synthetic Data")
shap_analysis(synthetic_samples, synthetic_model, title="Synthetic")

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
    plt.savefig(f"VAE {title}_SHAP_Summary_Plot.jpg", format="jpg", dpi=300, bbox_inches="tight")
    plt.savefig(f"VAE {title}_SHAP_Summary_Plot.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to prevent display or memory issues

# Save SHAP plots for both original and synthetic data
save_shap_plot(features_normalized, original_model, title="Original")
save_shap_plot(synthetic_samples, synthetic_model, title="Synthetic")

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
original_top_features = get_top_important_features(features_normalized, original_model)
synthetic_top_features = get_top_important_features(synthetic_samples, synthetic_model)

# Calculate and print the similarity percentage
calculate_feature_similarity(original_top_features, synthetic_top_features, top_n=10)
