# Hyperspectral Image Classification

This repository contains a machine learning pipeline for pixel-wise classification of hyperspectral images.

The project evaluates both "fast" tree-based models and "heavy" neural networks, utilizing Principal Component Analysis (PCA) for dimensionality reduction. 
It also incorporates custom **spatial feature engineering** to capture local neighborhood information (mean and standard deviation) to improve classification accuracy.

## Features
* **Modular Architecture**: Clean separation of configuration, data loading, feature engineering, modeling, and evaluation.
* **Spectral Analysis**: Uses LightGBM, XGBoost, CatBoost, and MLP Classifiers on PCA-reduced spectral bands.
* **Spatial Analysis**: Extracts 3x3 local neighborhood features from PCA maps and trains a dedicated LightGBM spatial model.
* **Robust Evaluation**: 5-fold Stratified Cross-Validation to ensure reliable performance metrics.
* **Automated Best-Model Selection**: Dynamically picks the best-performing model to predict the full image map.

## Results
<img width="1059" height="743" alt="hyperSpectral" src="https://github.com/user-attachments/assets/9971f4fd-a43b-4c0a-94c5-366ee7680caa" />

## Project Structure

```text
hyperspectral_project/
│
├── data/                            # Directory for raw datasets
│   └── HyperspectralTask.mat        
│
├── config.py                        # Global variables, hyperparameters, and paths
├── data_loader.py                   # Data ingestion and preprocessing
├── features.py                      # Spatial feature extraction logic
├── models.py                        # Pipeline and model definitions
├── evaluation.py                    # Cross-validation and scoring functions
├── visualization.py                 # Exporting predictions and plotting
├── main.py                          # Main execution script
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
