import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import config
from data_loader import load_and_preprocess_data  # Updated import
from evaluation import cv_score_pipeline
from models import get_fast_models, get_neural_net, get_spatial_lgbm
from features import make_spatial_features_from_pca
from visualization import save_and_plot_results

warnings.filterwarnings("ignore")

def main():
    print("=" * 80)
    print("HYPERSPECTRAL IMAGE CLASSIFICATION (CPU, with NN + Spatial)")
    print("=" * 80)

    # Load Data
    Data, Labels, X_all, y_all, mask, X_labeled, y_labeled_0, num_classes, H, W, B = \
        load_and_preprocess_data(config.MAT_PATH)
    
    print(f"Data shape: {Data.shape} | Labels shape: {Labels.shape} | Bands: {B}")
    print("Unique labels:", np.unique(Labels))
    print(f"\nLabeled samples: {X_labeled.shape[0]} | Classes: {num_classes}")

    fast_results = []
    heavy_results = []

    # Fast Models
    print("\n" + "=" * 80)
    print("MODEL SELECTION – FAST MODELS (5-fold CV)")
    print("=" * 80)
    
    models_fast = get_fast_models(config.N_PCA_SPECTRAL, num_classes, config.RANDOM_STATE, config.CV)
    
    for name, pipe, k in models_fast:
        print(f"\nEvaluating {name}...")
        mean_acc, std_acc = cv_score_pipeline(pipe, X_labeled, y_labeled_0, n_splits=k, name=name, random_state=config.RANDOM_STATE)
        fast_results.append((name, mean_acc, std_acc, pipe))
        print(f"  {name} | CV acc: {mean_acc:.4f} +/- {std_acc:.4f}")

    # Heavy Models
    print("\n" + "=" * 80)
    print("MODEL SELECTION – HEAVY MODELS (5-fold CV)")
    print("=" * 80)

    nn_pca = get_neural_net(config.N_PCA_SPECTRAL, config.RANDOM_STATE)
    mean_acc, std_acc = cv_score_pipeline(nn_pca, X_labeled, y_labeled_0, n_splits=config.CV, name="NeuralNet_PCA", random_state=config.RANDOM_STATE)
    heavy_results.append(("NeuralNet_PCA", mean_acc, std_acc, nn_pca))
    print(f"  NeuralNet_PCA | CV acc: {mean_acc:.4f} +/- {std_acc:.4f}")

    # Spatial Features & Model
    print("\nEvaluating LightGBM_Spatial...")
    spatial_prep = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=config.N_PCA_SPATIAL, random_state=config.RANDOM_STATE)),
    ])
    spatial_prep.fit(X_labeled)

    X_spatial_all = make_spatial_features_from_pca(Data, spatial_prep, window=config.WINDOW)    # Make spatial features for all pixels
    X_spatial_labeled = X_spatial_all[mask]                                                     # Keep only the labeled rows (pixels)

    spatial_model = get_spatial_lgbm(num_classes, config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=config.CV, shuffle=True, random_state=config.RANDOM_STATE)
    scores = []

    for fold, (tr, va) in enumerate(cv.split(X_spatial_labeled, y_labeled_0), 1):
        spatial_model.fit(X_spatial_labeled[tr], y_labeled_0[tr])
        pred = spatial_model.predict(X_spatial_labeled[va])
        acc = accuracy_score(y_labeled_0[va], pred)
        scores.append(acc)
        print(f"  LightGBM_Spatial | Fold {fold}: acc={acc:.4f}")

    mean_acc, std_acc = float(np.mean(scores)), float(np.std(scores))
    heavy_results.append(("LightGBM_Spatial", mean_acc, std_acc, (spatial_prep, spatial_model)))
    print(f"  LightGBM_Spatial | CV acc: {mean_acc:.4f} +/- {std_acc:.4f}")

    # Pick Best Model
    all_results = fast_results + heavy_results
    all_results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 80)
    print("OVERALL RESULTS (sorted)")
    print("=" * 80)
    for name, mean_acc, std_acc, _ in all_results:
        print(f"{name:18s} CV acc: {mean_acc:.4f} +/- {std_acc:.4f}")

    best_name, best_mean, best_std, best_obj = all_results[0]
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_name}")
    print(f"CV acc: {best_mean:.4f} +/- {best_std:.4f}")
    print("=" * 80)

    # Train Best Model & Predict
    print("\nTraining best model on all labeled and predicting full image")
    if best_name.endswith("_PCA"):
        best_pipeline = best_obj
        best_pipeline.fit(X_labeled, y_labeled_0)
        pred_all_0 = best_pipeline.predict(X_all).astype(int)
    elif best_name == "LightGBM_Spatial":
        spatial_prep, spatial_model = best_obj
        spatial_prep.fit(X_labeled)     # Train on all labeled
        X_spatial_all = make_spatial_features_from_pca(Data, spatial_prep, window=config.WINDOW)
        X_spatial_labeled = X_spatial_all[mask]
        spatial_model.fit(X_spatial_labeled, y_labeled_0)
        pred_all_0 = spatial_model.predict(X_spatial_all).astype(int)
    else:
        raise RuntimeError("Unknown best model type.")

    pred_all_1 = pred_all_0 + 1     # Back to 1..9 and reshape
    pred_map = pred_all_1.reshape(H, W)

    print("Pred map shape:", pred_map.shape)
    print("Unique predicted labels:", np.unique(pred_map))

    # Visualization
    save_and_plot_results(pred_map, Labels, best_name, num_classes)

if __name__ == "__main__":
    main()