# Synthetic-Data-Generation-Algorithms
Synthetic Data Generation Algorithms (VAE-GAN-Diffusion Model-LSTM-Copula)

# Synthetic Data Generation for Physiological Emotion Recognition EEG Data
This repository demonstrates various synthetic data generation techniques using **Variational Autoencoders (VAE)**, **Generative Adversarial Networks (GAN)**, **Diffusion Model**, **LSTM-based model**, and **Copula model**. The synthetic data generation is performed on a dataset of **Physiological Emotion Recognition EEG Data** involving **4 different emotions**.

# Brainwave EEG Dataset
This project utilizes small part of the [Brainwave EEG Dataset](https://ieee-dataport.org/documents/brainwave-eeg-dataset) available on IEEE DataPort. This dataset contains EEG signals associated with various emotional states, providing data for emotion recognition tasks.
- **Data Type**: EEG signals collected from participants across different emotional states.

## Usage
Download the dataset directly from the IEEE DataPort link provided and place it in your working directory to run the synthetic data generation and analysis scripts in this repository.

## Citation
Please cite the dataset according to the guidelines on the IEEE DataPort page if used in any published work.


## Overview

Emotion recognition through physiological signals is an essential task in affective computing. In this project, we utilize multiple machine learning and deep learning models to generate synthetic data that mimics the original dataset. The generated data can enhance training, testing, or validation by providing additional samples for model generalization and robustness.

The dataset used is **Physiological Emotion Recognition EEG Data**, containing 4 classes representing distinct emotional states. Data preprocessing includes a **split of 20% of each class** for synthetic data generation and **80% for training the generation models**. This structure is applied across each method.

The synthetic data generation models explored in this project are:

1. **Variational Autoencoder (VAE)**
2. **Generative Adversarial Network (GAN)**
3. **Diffusion Model**
4. **LSTM Model**
5. **Copula Model**

Each model is evaluated based on similarity between synthetic and original data using SHAP feature importance analysis and classification performance.

---

## Data Preprocessing

1. **Data**: The dataset used is **Physiological Emotion Recognition EEG Data**, which includes physiological measurements associated with 4 different emotions.
2. **Splitting**: The dataset is split by class into **20% for synthetic data generation** and **80% for training** the model.
3. **Normalization**: The features are normalized using **MinMaxScaler** to fit within the range [0,1], which is especially useful for neural networks and Copula modeling.

---

## Synthetic Data Generation Models

### Variational Autoencoder (VAE)
The VAE model is an unsupervised generative model that learns to generate synthetic samples by compressing input data into a latent space and then reconstructing it. Here, the VAE is trained on the 80% training data and used to generate synthetic EEG data for emotion recognition.

- **Architecture**: The VAE consists of an encoder and a decoder network with dense layers. The encoder compresses the input into a latent space, while the decoder reconstructs the data from the latent space.
- **Data Split**: 80% of each class is used for training the VAE, and the remaining 20% is held out for testing and synthetic generation.
- **Synthetic Data**: 2000 samples are generated, mimicking the structure and characteristics of the original data.

### Generative Adversarial Network (GAN)
The GAN model consists of a generator and a discriminator network trained in an adversarial setup. The generator tries to create synthetic data resembling the original, while the discriminator attempts to distinguish real from synthetic data. 

- **Architecture**: The generator network learns to generate data samples, while the discriminator attempts to classify samples as real or synthetic.
- **Data Split**: The GAN is trained on 80% of the original dataset, with 20% reserved for testing and synthetic generation.
- **Synthetic Data**: 2000 samples are generated by feeding random noise into the trained generator, creating realistic data.

### Diffusion Model
Diffusion models are probabilistic models that generate data by iteratively denoising a noisy signal. In this project, a diffusion-based generative model is used to learn the data structure and produce synthetic EEG samples by reversing the noise.

- **Architecture**: A diffusion process is applied iteratively to generate synthetic samples by removing noise over each step.
- **Data Split**: 80% of the data is used for training the diffusion model, and the remaining 20% is held out for synthetic data generation.
- **Synthetic Data**: 500 samples per class are generated, simulating the data’s distribution and patterns.

### LSTM-based Model
The LSTM-based generative model leverages sequential data patterns using Long Short-Term Memory (LSTM) layers. It learns temporal dependencies in EEG data and generates synthetic samples by predicting next steps in the sequence.

- **Architecture**: The LSTM model consists of stacked LSTM layers with variational dropout to capture sequence dependencies in the EEG data.
- **Data Split**: 80% of the data is used for model training, while 20% is held out for generating synthetic sequences.
- **Synthetic Data**: 1500 samples are generated, each capturing sequential dependencies learned by the LSTM.

### Copula Model
The Copula model, specifically a Gaussian Multivariate Copula, is a statistical model used to learn dependencies between variables. It generates synthetic data by capturing the joint distribution of features and sampling from it.

- **Architecture**: A Gaussian copula model is fitted to the data, capturing the statistical dependencies between variables.
- **Data Split**: The copula model is trained on 80% of the data, with 20% held out for synthetic sample generation.
- **Synthetic Data**: 1500 samples are generated, capturing similar feature dependencies as the original data.

---

## SHAP Analysis for Feature Importance

To evaluate the similarity between original and synthetic data, we employ **SHAP (SHapley Additive exPlanations)** for feature importance analysis. This is done for each model, where:

- SHAP feature importance values are calculated for both original and synthetic datasets.
- **Top 10 important features** are extracted and compared between the original and synthetic data for each model.
- **Similarity Analysis**: We calculate the percentage of similarity between the top 10 features in original and synthetic data.

---

## Evaluation

### Classification with XGBoost
Each synthetic dataset is evaluated using an **XGBoost classifier** trained on the generated data. The classifier’s accuracy and confusion matrix are measured for each of the following:

1. **Original Data**: 80% of the original data used for training the XGBoost classifier.
2. **Synthetic Data**: XGBoost classifier is trained on the generated synthetic data.
3. **Combined Data**: Training on a combination of original and synthetic data.

### Mean Squared Error (MSE)
To quantitatively evaluate the similarity between synthetic and original data, we calculate the **Mean Squared Error (MSE)** between normalized original and synthetic datasets.

### Similarity of Important Features
The top features from SHAP analysis are compared across original and synthetic datasets. The similarity percentage is calculated for the top 10 features, indicating how well the synthetic data retains important feature characteristics of the original data.

---

## Results

| Model               | Synthetic Samples Generated | SHAP Feature Similarity (%) | XGBoost Accuracy (Synthetic) | XGBoost Accuracy (Combined) |
|---------------------|----------------------------|-----------------------------|------------------------------|-----------------------------|
| VAE                 | 2000                       | 80%                         | 50.13%                            | 69.56%                           |
| GAN                 | 2000                       | 85%                         | -                            | -                           |
| Diffusion Model     | 500 per class              | 80%                         | -                            | -                           |
| LSTM                | 1500                       | 81%                         | -                            | -                           |
| Copula              | 1500                       | 80%                         | -                            | -                           |

**Note**: Values in the table are placeholders; update them with actual evaluation metrics after running experiments.

---

## Files

- **`vae_synthetic_data.py`**: Script for VAE-based synthetic data generation and evaluation.
- **`gan_synthetic_data.py`**: Script for GAN-based synthetic data generation and evaluation.
- **`diffusion_synthetic_data.py`**: Script for diffusion model synthetic data generation and evaluation.
- **`lstm_synthetic_data.py`**: Script for LSTM-based synthetic data generation and evaluation.
- **`copula_synthetic_data.py`**: Script for Copula-based synthetic data generation and evaluation.
- **`evaluation.py`**: Script for running SHAP, MSE, and classification evaluations on generated datasets.

---

## Getting Started

### Requirements
- Python 3.8+
- Packages: `numpy`, `pandas`, `sklearn`, `tensorflow`, `xgboost`, `shap`, `copulas`, `matplotlib`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
