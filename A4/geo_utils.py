"""
Satellite Image Processing and DOFA Inference Pipeline

This module contains functions for:
- Loading and preprocessing .tif satellite images.
- Creating normalization parameters for different satellite types.
- Running inference using the DOFA (Dynamic One-For-All) model and optionally visualizing the output.
"""

import torch
import rasterio
import numpy as np
from torchvision import transforms
from torchgeo.models import DOFA
import os
from typing import Tuple, List, Optional
import warnings
import time
import matplotlib.pyplot as plt

DEFAULT_WAVELENGTHS = [0.64, 0.56, 0.48]  # RGB band wavelengths in micrometers

def load_tif_as_tensor(
    path: str,
    size: Tuple[int, int] = (224, 224),
    bands: List[int] = [1, 2, 3],
    band_order: str = 'RGB',
    normalization_params: Optional[dict] = None,
    handle_nodata: bool = True,
    nodata_fill: float = 0.0,
    data_range: Optional[Tuple[float, float]] = None,
    clip_percentiles: Optional[Tuple[float, float]] = None
) -> torch.Tensor:
    """
    Loads a .tif image, processes and normalizes it, and returns a PyTorch tensor.

    Args:
        path (str): Path to the .tif image.
        size (tuple): Target (height, width) for resizing.
        bands (list): List of band indices to extract.
        band_order (str): Order of bands ('RGB', 'BGR', or 'custom').
        normalization_params (dict): Mean and std for normalization.
        handle_nodata (bool): Whether to handle nodata values.
        nodata_fill (float): Value to fill nodata with.
        data_range (tuple): Optional fixed min/max for scaling.
        clip_percentiles (tuple): Percentiles to clip outliers.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, H, W).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(path) as src:
        if max(bands) > src.count:
            raise ValueError(f"File only has {src.count} bands.")
        img = src.read(bands).astype(np.float32)
        dtype = src.dtypes[0]
        nodata_value = src.nodata
        print(f"Original data type: {dtype}")
        print(f"Image shape: {img.shape}")
        print(f"Data range: {img.min()} to {img.max()}")

    if handle_nodata and nodata_value is not None:
        img[img == nodata_value] = nodata_fill
        print(f"Handled nodata value: {nodata_value}")

    if data_range:
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
    elif dtype == 'uint8':
        img = img / 255.0
    elif dtype == 'uint16':
        img = img / 65535.0
    elif img.max() > 1:
        img = img / img.max()

    if clip_percentiles:
        low = np.percentile(img, clip_percentiles[0])
        high = np.percentile(img, clip_percentiles[1])
        img = np.clip(img, low, high)
        img = (img - low) / (high - low)
        print(f"Clipped to {clip_percentiles} percentiles and renormalized")

    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)

    if band_order == 'BGR' and img.shape[2] >= 3:
        img = img[:, :, [2, 1, 0]]

    if normalization_params is None:
        normalization_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(**normalization_params)
    ])

    tensor = transform(img).unsqueeze(0)
    print(f"Final tensor shape: {tensor.shape}")
    return tensor

def create_satellite_normalization_params(imagery_type: str = 'sentinel2') -> dict:
    """
    Returns mean and std values for different satellite imagery types.

    Args:
        imagery_type (str): Type of satellite ('sentinel2', 'landsat', or 'imagenet').

    Returns:
        dict: Normalization parameters with 'mean' and 'std'.
    """
    if imagery_type.lower() == 'sentinel2':
        return {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    elif imagery_type.lower() == 'landsat':
        return {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}
    else:
        return {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

def run_dofa_inference(
    image_tensor: torch.Tensor,
    wavelengths: List[float] = DEFAULT_WAVELENGTHS,
    visualize: bool = True
) -> torch.Tensor:
    """
    Loads the DOFA model, runs inference, and optionally visualizes the output.

    Args:
        image_tensor (torch.Tensor): Input tensor of shape (1, 3, H, W).
        wavelengths (list): RGB wavelengths in micrometers.
        visualize (bool): Whether to plot the 45D feature vector output.

    Returns:
        torch.Tensor: Output feature vector of shape (1, 45).
    """
    model = DOFA(img_size=224, patch_size=16)
    model.eval()

    with torch.no_grad():
        start = time.time()
        output = model(image_tensor, wavelengths=wavelengths)
        duration = time.time() - start
        print(f"Inference completed in {duration:.2f} seconds")
        print(f"Output shape: {output.shape}")

    if visualize:
        # Plot 45D feature vector as a line graph
        vec = output.squeeze().cpu().numpy()
        plt.figure(figsize=(10, 3))
        plt.plot(vec, marker='o')
        plt.title("DOFA 45-Dimensional Feature Vector")
        plt.xlabel("Feature Index")
        plt.ylabel("Activation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return output