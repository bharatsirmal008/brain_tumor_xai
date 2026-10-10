"""Experimental three-view tumor screening before tumor classification."""

from pathlib import Path

import numpy as np
import streamlit as st
import torch

from src.stage1_three_image_binary import Stage1ThreeImageBinary


@st.cache_resource
def load_binary_screening_model():
    path = Path(__file__).resolve().parents[1] / "checkpoints/stage1_three_image_full_v1/best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["experiment"] != "STAGE1_THREE_IMAGE_FULL_V1":
        raise ValueError("Unexpected screening checkpoint experiment")
    threshold = float(checkpoint["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Screening threshold must be within [0, 1]")
    model = Stage1ThreeImageBinary()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, threshold


def run_binary_screening(processed_views):
    images = np.stack([
        np.asarray(processed_views[name], dtype=np.float32)
        for name in ("axial", "coronal", "sagittal")
    ])
    if images.shape != (3, 224, 224):
        raise ValueError(f"Invalid screening input: {images.shape}")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("MRI pixels must be finite and within [0, 1]")
    model, threshold = load_binary_screening_model()
    tensor = torch.from_numpy(np.ascontiguousarray(images)).unsqueeze(0)
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(tensor)).item())
    if not np.isfinite(probability):
        raise ValueError("Screening returned a non-finite score")
    return {
        "tumor_probability": probability,
        "threshold": threshold,
        "screen_positive": probability >= threshold,
    }
