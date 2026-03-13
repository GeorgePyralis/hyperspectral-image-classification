import numpy as np
import h5py

def load_and_preprocess_data(mat_path):
    # Load Data
    with h5py.File(mat_path, "r") as f:
        Data = np.array(f["Data"])
        Labels = np.array(f["Labels"])

    Data = np.transpose(Data, (2, 1, 0))  # (610, 340, 103)
    Labels = np.transpose(Labels, (1, 0)) # (610, 340)

    H, W, B = Data.shape                  
    
    # Flatten Data
    X_all = Data.reshape(-1, B)           # (H*W, B) Each row refers to 1 pixel with 103 bands
    y_all = Labels.reshape(-1)            # Label for every pixel

    mask = y_all > 0                      # Keep only the labeled pixels 
    X_labeled = X_all[mask] 
    y_labeled = y_all[mask].astype(int)   # 1..9
    y_labeled_0 = y_labeled - 1           # 0..8

    num_classes = int(y_labeled.max())    # =9
    
    return Data, Labels, X_all, y_all, mask, X_labeled, y_labeled_0, num_classes, H, W, B