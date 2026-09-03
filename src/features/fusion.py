import os
import numpy as np
from PIL import Image

def perform_spectral_fusion(image_path, alpha=0.6):
    """
    Performs pixel-level fusion on a 3-channel composite image (IR, WV, VIS).
    - Channel 0 (R): Infrared (convective cloud tops)
    - Channel 1 (G): Water Vapor (upper atmospheric moisture)
    - Channel 2 (B): Visible (fine cloud morphology)
    
    Returns a fused false-color composite image optimized for human visual analysis:
    - Fuses thermal gradients (IR) with structural morphology (VIS) using weighted alpha blending.
    - Highlights high-convective regions in vibrant colors (false-color LUT) and background moisture in dark tones.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Load composite image
    img = Image.open(image_path).convert("RGB")
    data = np.array(img, dtype=np.float32)

    ir = data[..., 0]   # Infrared band
    wv = data[..., 1]   # Water Vapor band
    vis = data[..., 2]  # Visible band

    # Create a custom False-Color Enhancement (similar to Dvorak BD curve)
    # Extremely cold cloud tops (high values in our normalized IR channel) get highlighted
    fused_r = ir * alpha + vis * (1 - alpha)
    fused_g = wv * 0.5 + vis * 0.5
    fused_b = np.where(ir > 200, 255.0 - ir, vis * alpha)  # Enhance eyewall cloud tops

    # Assemble fused channels
    fused_data = np.zeros_like(data, dtype=np.uint8)
    fused_data[..., 0] = np.clip(fused_r, 0, 255).astype(np.uint8)
    fused_data[..., 1] = np.clip(fused_g, 0, 255).astype(np.uint8)
    fused_data[..., 2] = np.clip(fused_b, 0, 255).astype(np.uint8)

    return Image.fromarray(fused_data)
