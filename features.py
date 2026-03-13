import numpy as np
from scipy.ndimage import uniform_filter

# Using the hyperspectral cube and a fitted PCA pipeline the function creates the following spatial features: center, local mean, local std.

def make_spatial_features_from_pca(Data_3d, pca_transformer, window=3):
    H, W, B = Data_3d.shape 
    X_flat = Data_3d.reshape(-1, B)           # Flatten image (H*W, B)
    PCs = pca_transformer.transform(X_flat)   # PCA transform (H*W, K)
    K = PCs.shape[1]
    PC_maps = PCs.reshape(H, W, K)            # Image Format again (H, W, K)

    # local mean and mean of squares (for variance/std)
    local_mean = np.zeros_like(PC_maps, dtype=np.float32)
    local_mean_sq = np.zeros_like(PC_maps, dtype=np.float32)

    for k in range(K):                                                      #for each channel k
        comp = PC_maps[:, :, k]                                             #2D image (H, W) for this specific PCA channel
        m = uniform_filter(comp, size=window, mode="reflect")               #local 3*3 mean. Mirror the values when on edges
        msq = uniform_filter(comp * comp, size=window, mode="reflect")      #local mean of comp^2
        local_mean[:, :, k] = m                                             #save each channel's results in the array
        local_mean_sq[:, :, k] = msq

    local_var = np.maximum(local_mean_sq - local_mean * local_mean, 1e-8)   #Variance of each pixel in its 3*3 window
    local_std = np.sqrt(local_var)                                          #standard deviation of each pixel in its 3*3 window

    feats = np.concatenate([PC_maps, local_mean, local_std], axis=2)        # Spatial Features Array: pixel's PCA value for each k + local_mean + local_std (H, W, 3K)

    return feats.reshape(-1, 3 * K)                                         # flatten (H*W, 3K)