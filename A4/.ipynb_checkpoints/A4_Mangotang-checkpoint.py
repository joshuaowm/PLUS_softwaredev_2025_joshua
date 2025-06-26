# dofa_pipeline.py

import os
import warnings
from typing import Tuple, List, Optional

import numpy as np
import torch
import rasterio
from torchvision import transforms
from torchgeo.models import DOFA


# ---- Configuration ----
DEFAULT_WAVELENGTHS = [0.64, 0.56, 0.48]  # R, G, B in micrometers

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
    """Load a GeoTIFF and return a normalized torch tensor."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not path.lower().endswith(('.tif', '.tiff')):
        warnings.warn(f"File extension may not be TIFF: {path}")

    try:
        with rasterio.open(path) as src:
            if max(bands) > src.count:
                raise ValueError(f"Requested band {max(bands)} but file has {src.count} bands")
            img = src.read(bands).astype(np.float32)
            nodata_value = src.nodata
            dtype = src.dtypes[0]
            print(f"Original dtype: {dtype}, Shape: {img.shape}, Range: {img.min()} to {img.max()}")
    except Exception as e:
        raise RuntimeError(f"Error reading image: {e}")

    # Handle nodata
    if handle_nodata and nodata_value is not None:
        img[img == nodata_value] = nodata_fill
        print(f"Replaced nodata ({nodata_value}) with {nodata_fill}")

    # Scale based on type or provided range
    if data_range:
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
        img = np.clip(img, 0, 1)
        print(f"Scaled using provided range: {data_range}")
    elif dtype == 'uint8':
        img /= 255.0
        print("Scaled 8-bit data")
    elif dtype == 'uint16':
        img /= 65535.0
        print("Scaled 16-bit data")
    elif img.max() > 1:
        img /= img.max()
        print("Auto-scaled to [0, 1]")

    # Optional clipping
    if clip_percentiles:
        low, high = np.percentile(img, clip_percentiles)
        img = np.clip(img, low, high)
        img = (img - low) / (high - low)
        print(f"Clipped to {clip_percentiles} percentiles")

    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

    # Band order
    if band_order == 'BGR':
        img = img[:, :, [2, 1, 0]]
        print("Reordered bands: BGR → RGB")

    # Normalization
    norm = normalization_params or create_satellite_normalization_params()
    print(f"Normalization: mean={norm['mean']}, std={norm['std']}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm['mean'], std=norm['std'])
    ])

    tensor = transform(img).unsqueeze(0)  # Add batch dim
    print(f"Final tensor shape: {tensor.shape}")
    return tensor

def create_satellite_normalization_params(imagery_type: str = 'sentinel2') -> dict:
    """Return normalization stats based on imagery type."""
    if imagery_type.lower() == 'sentinel2':
        return {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    elif imagery_type.lower() == 'landsat':
        return {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}
    else:
        return {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

def run_dofa_inference(
    image_tensor: torch.Tensor,
    wavelengths: List[float] = DEFAULT_WAVELENGTHS,
    device: Optional[str] = None,
    visualize: bool = False
    ) -> torch.Tensor:
    """Run DOFA model on preprocessed tensor."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    try:
        model = DOFA(img_size=224, patch_size=16).to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize DOFA model: {e}")

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        try:
            output = model(image_tensor, wavelengths=wavelengths)
            print(f"Inference complete. Output shape: {output.shape}")

            if visualize:
                import matplotlib.pyplot as plt
                plt.plot(output.squeeze().cpu().numpy())
                plt.title("DOFA Output Feature Vector")
                plt.xlabel("Feature Index")
                plt.ylabel("Activation")
                plt.grid(True)
                plt.show()

            return output

        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
