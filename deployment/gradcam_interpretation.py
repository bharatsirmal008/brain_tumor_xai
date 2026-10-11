"""Display-only summaries of normalized Grad-CAM maps; no anatomical inference."""
import numpy as np


def describe_gradcam(image, cam, view, predicted_class):
    image = np.asarray(image, dtype=float)
    cam = np.asarray(cam, dtype=float)
    prefix = f"**{view.title()} view:** "
    if image.ndim != 2 or cam.shape != image.shape:
        return prefix + "An attention summary is unavailable because the image and heatmap dimensions differ."
    support = np.isfinite(image) & (image > 1e-5)
    if not support.any() or not np.isfinite(cam[support]).all():
        return prefix + "There is not enough valid image or heatmap data to summarize attention."
    values = np.clip(cam, 0, 1)
    visible = values[support]
    if np.ptp(visible) <= 1e-6:
        return prefix + "The heatmap has no distinct attention hotspot within the non-black image area; a strongest region cannot be identified reliably."
    # Locate the centroid of maximum-valued pixels; ties do not select an arbitrary corner.
    rows, cols = np.where(support & (values == visible.max()))
    horizontal = ("left", "center", "right")[min(2, int(cols.mean() / image.shape[1] * 3))]
    vertical = ("upper", "middle", "lower")[min(2, int(rows.mean() / image.shape[0] * 3))]
    high = 100 * np.mean(visible >= 2 / 3)
    moderate = 100 * np.mean((visible >= 1 / 3) & (visible < 2 / 3))
    low = 100 * np.mean(visible < 1 / 3)
    return (prefix + f"The peak AI attention is located in the **{horizontal}-{vertical} image region** "
            f"for the predicted **{predicted_class.title()}** class. Warmer colors show stronger relative attention; "
            "cooler colors show lower attention. Within the non-black MRI image area, "
            f"**{high:.1f}%** has high activation, **{moderate:.1f}%** moderate activation, and "
            f"**{low:.1f}%** low activation. This describes image coordinates and model attention, "
            "not confirmed anatomical localization or a tumor boundary.")
