import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_and_plot_results(pred_map, Labels, best_name, num_classes):
    out_file = "test_predictions.csv"
    pd.DataFrame(pred_map).to_csv(out_file, index=False, header=False)

    sub = pd.read_csv(out_file, header=None)
    print("\nSaved:", out_file)
    print("SUB shape:", sub.shape)
    print("Value range:", sub.min().min(), sub.max().max(), f"(should be 1..{num_classes})")

    print("\nPlotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    labels_vis = np.where(Labels == 0, np.nan, Labels)
    axes[0].imshow(labels_vis, cmap='jet')
    axes[0].set_title("Ground Truth (Labeled Pixels)")
    axes[0].axis('off')

    axes[1].imshow(pred_map, cmap='jet')
    axes[1].set_title(f"Model Prediction ({best_name})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()