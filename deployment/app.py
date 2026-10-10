from pathlib import Path
import sys
from io import BytesIO
import tempfile
import os

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    gaussian_filter,
    label
)


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT)
    )


from deployment.final_deployment import (
    load_final_model,
    preprocess_raw_upload,
    predict_patient,
    CLASS_NAMES,
)

from deployment.input_guard import (
    inspect_raw_image,
    detect_duplicate_views,
    check_orientation_filename,
    validate_age,
    severe_image_problem,
    image_quality_warning,
)

from deployment.report_generator import build_patient_report
from deployment.screening import run_binary_screening


# ================================================================================================================
# PAGE
# ================================================================================================================

st.set_page_config(
    page_title="Brain Tumor XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(
    """
<style>
:root {
    --ink: #102a43;
    --muted: #627d98;
    --navy: #0b2741;
    --navy-2: #123e5f;
    --teal: #00a6a6;
    --teal-dark: #007f82;
    --cyan: #35d0ba;
    --surface: rgba(255, 255, 255, 0.88);
    --surface-solid: #ffffff;
    --line: #d9e7ef;
    --soft: #eef7fa;
    --success: #0b8f6a;
    --shadow: 0 18px 50px rgba(16, 42, 67, 0.10);
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    color: var(--ink);
    color-scheme: light;
    background:
        radial-gradient(circle at 10% 0%, rgba(53, 208, 186, .14), transparent 28rem),
        radial-gradient(circle at 92% 12%, rgba(46, 134, 193, .12), transparent 30rem),
        linear-gradient(180deg, #f7fbfd 0%, #eef6fa 100%);
}

[data-testid="stHeader"] {
    background: rgba(247, 251, 253, .78) !important;
    backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(217, 231, 239, .72);
}

[data-testid="stToolbar"] button,
[data-testid="stToolbar"] span {
    color: var(--navy) !important;
}

.block-container {
    max-width: 1240px;
    padding-top: 1.3rem;
    padding-bottom: 4rem;
}

.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 2.25rem 2.4rem 2rem;
    margin-bottom: 1.2rem;
    color: white;
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 28px;
    background: linear-gradient(125deg, #09243c 0%, #0b4962 58%, #008d8e 100%);
    box-shadow: 0 24px 65px rgba(10, 55, 80, .22);
    animation: rise-in .55s ease-out both;
}

.hero-shell::after {
    content: "";
    position: absolute;
    width: 290px;
    height: 290px;
    right: -75px;
    top: -115px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(91, 236, 210, .38), rgba(91, 236, 210, 0) 68%);
    pointer-events: none;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    padding: .38rem .7rem;
    margin-bottom: .85rem;
    border: 1px solid rgba(255,255,255,.24);
    border-radius: 999px;
    background: rgba(255,255,255,.10);
    font-size: .73rem;
    font-weight: 750;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.hero-title {
    max-width: 760px;
    font-size: clamp(2rem, 4vw, 3.35rem);
    line-height: 1.02;
    font-weight: 800;
    letter-spacing: -.055em;
    margin: 0 0 .75rem;
}

.hero-subtitle {
    max-width: 720px;
    margin: 0;
    color: rgba(255,255,255,.78);
    font-size: 1.02rem;
    line-height: 1.65;
}

.workflow-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .8rem;
    margin: 0 0 1.2rem;
}

.workflow-step {
    display: flex;
    align-items: center;
    gap: .75rem;
    padding: .85rem 1rem;
    border: 1px solid rgba(185, 211, 224, .78);
    border-radius: 16px;
    background: rgba(255,255,255,.68);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 24px rgba(16,42,67,.05);
}

.workflow-step span {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    flex: 0 0 30px;
    border-radius: 10px;
    color: white;
    background: linear-gradient(135deg, var(--teal), var(--cyan));
    font-size: .78rem;
    font-weight: 800;
}

.workflow-step div {
    color: var(--ink);
    font-size: .88rem;
    font-weight: 720;
}

.section {
    display: flex;
    align-items: center;
    gap: .65rem;
    color: var(--navy);
    font-size: 1.35rem;
    font-weight: 780;
    letter-spacing: -.025em;
    margin-top: 2rem;
    margin-bottom: .8rem;
}

.section::before {
    content: "";
    width: 9px;
    height: 26px;
    border-radius: 99px;
    background: linear-gradient(180deg, var(--cyan), var(--teal-dark));
    box-shadow: 0 0 0 5px rgba(0,166,166,.08);
}

p, label {
    color: #48657d;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #56738a !important;
    opacity: 1 !important;
    font-weight: 520;
    line-height: 1.55;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stNumberInput"] label p,
[data-testid="stSelectbox"] label p,
[data-testid="stFileUploader"] label p {
    color: var(--navy) !important;
    opacity: 1 !important;
    font-weight: 720 !important;
}

[data-testid="stFileUploader"] {
    padding: .35rem;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: var(--surface);
    box-shadow: 0 10px 30px rgba(16,42,67,.06);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}

[data-testid="stFileUploader"]:hover {
    transform: translateY(-2px);
    border-color: rgba(0,166,166,.55);
    box-shadow: 0 16px 36px rgba(16,42,67,.10);
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 128px;
    border: 1.5px dashed #9fc7d5 !important;
    border-radius: 14px !important;
    background: linear-gradient(145deg, #f7fcfd, #edf8fa) !important;
}

[data-testid="stFileUploaderDropzone"] button {
    color: var(--teal-dark) !important;
    border-color: #9bcfd2 !important;
    background: white !important;
    box-shadow: 0 6px 15px rgba(0,127,130,.10) !important;
}

[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzone"] button svg {
    color: var(--teal-dark) !important;
    fill: var(--teal-dark) !important;
    opacity: 1 !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    color: white !important;
    border-color: var(--teal-dark) !important;
    background: var(--teal-dark) !important;
}

[data-testid="stFileUploaderDropzone"] button:hover p,
[data-testid="stFileUploaderDropzone"] button:hover span,
[data-testid="stFileUploaderDropzone"] button:hover svg {
    color: white !important;
    fill: white !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] > div > span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #45657b !important;
    opacity: 1 !important;
}

div[data-baseweb="input"], div[data-baseweb="select"] > div {
    min-height: 48px;
    border-color: var(--line) !important;
    border-radius: 13px !important;
    color: var(--ink) !important;
    background-color: rgba(255,255,255,.96) !important;
    box-shadow: 0 5px 18px rgba(16,42,67,.045);
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] span,
div[data-baseweb="select"] svg {
    color: var(--ink) !important;
    fill: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}

[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] [role="combobox"] {
    color: var(--ink) !important;
    background: #ffffff !important;
    -webkit-text-fill-color: var(--ink) !important;
}

[data-testid="stNumberInput"] div[data-baseweb="input"]:hover,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: #79bec4 !important;
    box-shadow: 0 7px 20px rgba(0,127,130,.10) !important;
}

[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: var(--teal) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0,166,166,.16) !important;
}

div[data-baseweb="input"] button {
    color: var(--teal-dark) !important;
    background: transparent !important;
}

div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(0,166,166,.12) !important;
}

[data-testid="stCheckbox"] {
    padding: .62rem .78rem;
    margin-bottom: .38rem;
    border: 1px solid var(--line);
    border-radius: 13px;
    background: rgba(255,255,255,.68);
    transition: background .2s ease, border-color .2s ease;
}

[data-testid="stCheckbox"]:hover {
    border-color: #9bcfd2;
    background: rgba(255,255,255,.95);
}

[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] span {
    color: #294b63 !important;
    opacity: 1 !important;
}

.result-card {
    min-height: 132px;
    border: 1px solid rgba(130, 190, 205, .45);
    border-radius: 22px;
    padding: 23px 24px;
    color: white;
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 100%);
    box-shadow: 0 16px 38px rgba(11,39,65,.16);
    animation: rise-in .45s ease-out both;
}

.result-label {
    color: rgba(255,255,255,.65);
    font-size: .76rem;
    font-weight: 750;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.result-value {
    color: white;
    font-size: clamp(1.55rem, 3vw, 2.15rem);
    line-height: 1.18;
    font-weight: 790;
    margin-top: .5rem;
}

.guard-pass, .guard-info {
    border-radius: 15px;
    padding: 13px 15px;
    line-height: 1.55;
}

.guard-pass {
    color: #05664f;
    background: #e9f8f2;
    border: 1px solid #a9decf;
}

.guard-info {
    color: #164e63;
    background: #edf8fb;
    border: 1px solid #b9dfe8;
}

[data-testid="stAlert"] {
    border-radius: 15px;
    border-width: 1px;
}

.stButton > button, [data-testid="stDownloadButton"] > button {
    min-height: 50px;
    border: 0 !important;
    border-radius: 14px !important;
    color: white !important;
    font-weight: 760 !important;
    letter-spacing: .01em;
    background: linear-gradient(110deg, #008d8e, #00b89c) !important;
    box-shadow: 0 12px 28px rgba(0,141,142,.23) !important;
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
}

.stButton > button p,
.stButton > button span,
[data-testid="stDownloadButton"] > button p,
[data-testid="stDownloadButton"] > button span {
    color: white !important;
    opacity: 1 !important;
}

.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-2px);
    filter: saturate(1.08) brightness(1.03);
    box-shadow: 0 16px 34px rgba(0,141,142,.30) !important;
}

.stButton > button:hover p,
.stButton > button:hover span,
[data-testid="stDownloadButton"] > button:hover p,
[data-testid="stDownloadButton"] > button:hover span,
.stButton > button:focus p,
[data-testid="stDownloadButton"] > button:focus p {
    color: white !important;
    opacity: 1 !important;
}

.stButton > button:disabled {
    color: #375b70 !important;
    border: 1px solid #b9cdd8 !important;
    background: #dce9ee !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

.stButton > button:disabled p,
.stButton > button:disabled span {
    color: #375b70 !important;
    opacity: 1 !important;
}

[data-testid="stMetric"] {
    min-height: 108px;
    padding: 16px 17px;
    border: 1px solid var(--line);
    border-radius: 17px;
    background: var(--surface);
    box-shadow: 0 10px 28px rgba(16,42,67,.06);
}

[data-testid="stMetricLabel"] p {
    color: var(--muted);
    font-size: .78rem;
    font-weight: 730;
    letter-spacing: .04em;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    color: var(--navy);
    font-weight: 780;
}

[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--teal-dark), var(--cyan)) !important;
}

[data-testid="stImage"] img {
    border: none !important;
    outline: none !important;
    border-radius: 17px;
    box-shadow: 0 14px 34px rgba(16,42,67,.12);
}

[data-testid="stSlider"] [role="slider"] {
    background: var(--teal) !important;
    border-color: white !important;
    box-shadow: 0 0 0 4px rgba(0,166,166,.15);
}

hr {
    border-color: var(--line) !important;
}

.footer {
    text-align: center;
    color: #48657d;
    font-size: .78rem;
    font-weight: 560;
    padding-top: 1.2rem;
}

@keyframes rise-in {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 760px) {
    .block-container { padding: .8rem .8rem 3rem; }
    .hero-shell { padding: 1.55rem 1.35rem; border-radius: 22px; }
    .workflow-strip { grid-template-columns: 1fr; }
    .workflow-step { padding: .7rem .8rem; }
    .section { margin-top: 1.55rem; }

    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: .85rem !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 112px;
    }

    .result-card {
        min-height: auto;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
        scroll-behavior: auto !important;
    }
}

</style>
""",
    unsafe_allow_html=True
)


st.markdown(
    """
<section class="hero-shell">
    <div class="hero-badge">✦ Explainable medical AI</div>
    <div class="hero-title">Brain Tumor XAI</div>
    <p class="hero-subtitle">
        Analyze three MRI perspectives with clinical context, transparent model attention,
        and a patient-friendly printable report.
    </p>
</section>
<div class="workflow-strip">
    <div class="workflow-step"><span>01</span><div>Upload three MRI views</div></div>
    <div class="workflow-step"><span>02</span><div>Verify patient information</div></div>
    <div class="workflow-step"><span>03</span><div>Review prediction and XAI</div></div>
</div>
""",
    unsafe_allow_html=True
)


# ================================================================================================================
# MODEL
# ================================================================================================================

@st.cache_resource
def get_model():

    return load_final_model()


with st.spinner(
    "Loading AI model..."
):

    model, checkpoint, device = get_model()


# ================================================================================================================
# HELPERS
# ================================================================================================================

def preprocess_uploaded(
    uploaded_file
):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp.name

    try:

        processed = preprocess_raw_upload(
            temp_path
        )

    finally:

        try:
            os.remove(
                temp_path
            )

        except Exception:
            pass

    return processed


def largest_component(mask):

    labeled, count = label(
        mask
    )

    if count == 0:
        return mask

    sizes = np.bincount(
        labeled.ravel()
    )

    sizes[0] = 0

    largest = int(
        sizes.argmax()
    )

    return (
        labeled == largest
    )



def anatomy_alpha_mask(image):
    """
    DISPLAY ONLY.

    Keeps the previous smooth Grad-CAM appearance,
    while preventing heatmap from appearing over pure black padding.

    Raw Grad-CAM values and prediction are NOT changed.
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # Anything with actual MRI intensity is considered image support.
    mask = (
        image > 1e-5
    ).astype(
        np.float32
    )

    # Soft edge so heatmap does not look sharply clipped.
    mask = gaussian_filter(
        mask,
        sigma=1.8
    )

    if mask.max() > 0:

        mask = (
            mask /
            mask.max()
        )

    return np.clip(
        mask,
        0.0,
        1.0
    )


def make_gradcam_overlay(
    image,
    raw_cam,
    overlay_opacity=0.55
):
    """
    Previous-style smooth Grad-CAM visualization.

    IMPORTANT:
    - raw CAM location unchanged
    - no CAM renormalization after anatomy mask
    - no tumor-region forcing
    - only black padding is made transparent
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    cam = np.asarray(
        raw_cam,
        dtype=np.float32
    )

    cam = np.clip(
        cam,
        0.0,
        1.0
    )

    anatomy = anatomy_alpha_mask(
        image
    )


    # Previous visual style:
    # broad smooth heatmap remains visible.
    visible_cam = np.ma.masked_where(
        anatomy < 0.03,
        cam
    )


    # Alpha depends mostly on anatomy,
    # NOT on CAM strength.
    #
    # This restores the full blue-green-yellow-red map
    # instead of showing only tiny red patches.
    overlay_opacity = float(
        np.clip(
            overlay_opacity,
            0.0,
            0.85
        )
    )

    alpha_map = (
        overlay_opacity
        *
        anatomy
    )


    fig, ax = plt.subplots(
        figsize=(
            6.4,
            6.4
        ),
        dpi=180
    )

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")


    # MRI base
    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="lanczos"
    )


    # Smooth full Grad-CAM
    ax.imshow(
        visible_cam,
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=alpha_map,
        interpolation="lanczos"
    )


    # Strong-attention contour
    contour_cam = np.ma.masked_where(
        anatomy < 0.08,
        cam
    )

    if overlay_opacity >= 0.10 and np.any(
        (
            cam >= 0.60
        )
        &
        (
            anatomy >= 0.08
        )
    ):

        try:

            ax.contour(
                contour_cam,
                levels=[
                    0.60
                ],
                linewidths=1.15
            )

        except Exception:
            pass


    ax.set_xlim(
        0,
        223
    )

    ax.set_ylim(
        223,
        0
    )

    ax.axis(
        "off"
    )

    fig.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    return fig


def figure_to_borderless_png(fig):
    """Convert a Grad-CAM figure to a sharp PNG without a surrounding frame."""
    output = BytesIO()
    fig.savefig(
        output,
        format="png",
        dpi=180,
        facecolor="black",
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )
    output.seek(0)
    return output.getvalue()


@st.fragment
def render_gradcam_comparison(processed_views, gradcam):
    """Interactive display-only comparison that does not rerun inference."""
    overlay_percent = st.slider(
        "Original MRI ↔ Grad-CAM overlay",
        min_value=0,
        max_value=85,
        value=55,
        step=5,
        help=(
            "Move to 0% to see the original processed MRI. Move right to increase "
            "the visibility of the Grad-CAM attention colors."
        ),
        key="gradcam_overlay_percent",
    )
    st.caption(
        f"Current view: {'original MRI only' if overlay_percent == 0 else f'{overlay_percent}% attention overlay'}. "
        "This control changes only the display, not the prediction or Grad-CAM values."
    )

    attention_cols = st.columns(3, gap="medium")
    for column, view_name in zip(
        attention_cols,
        ("axial", "coronal", "sagittal")
    ):
        with column:
            fig = make_gradcam_overlay(
                processed_views[view_name],
                gradcam[view_name],
                overlay_opacity=overlay_percent / 100.0
            )
            overlay_png = figure_to_borderless_png(fig)
            plt.close(fig)
            st.image(overlay_png, width="stretch")
            st.caption(
                f"{view_name.title()} {'MRI' if overlay_percent == 0 else 'attention'}"
            )


def make_mri_figure(image):
    """Render a processed MRI slice without altering inference data."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="bicubic")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def make_gradcam_heatmap(raw_cam):
    """Render the normalized Grad-CAM values as a standalone heatmap."""
    cam = np.clip(np.asarray(raw_cam, dtype=np.float32), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    heatmap = ax.imshow(cam, cmap="jet", vmin=0, vmax=1, interpolation="bicubic")
    ax.axis("off")
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.subplots_adjust(left=0, right=0.9, top=1, bottom=0)
    return fig


def gradcam_summary(image, raw_cam):
    """Calculate display-space CAM statistics inside the visible MRI support."""
    cam = np.clip(np.asarray(raw_cam, dtype=np.float32), 0.0, 1.0)
    support = anatomy_alpha_mask(image) >= 0.08
    if not np.any(support):
        support = np.ones_like(cam, dtype=bool)

    supported_cam = np.where(support, cam, -1.0)
    y, x = np.unravel_index(np.argmax(supported_cam), supported_cam.shape)
    height, width = cam.shape

    horizontal = "left" if x < width / 3 else "right" if x >= 2 * width / 3 else "center"
    vertical = "upper" if y < height / 3 else "lower" if y >= 2 * height / 3 else "middle"
    values = cam[support]

    return {
        "location": f"{horizontal}-{vertical}",
        "high": float(np.mean(values >= 0.70) * 100.0),
        "moderate": float(np.mean((values >= 0.40) & (values < 0.70)) * 100.0),
        "low": float(np.mean((values > 0.0) & (values < 0.40)) * 100.0),
    }


def shapley_explanation(feature_name, value, predicted_name):
    magnitude = abs(float(value))
    strength = "very small" if magnitude <= 0.01 else "small" if magnitude < 0.05 else "noticeable"

    if value > 0.01:
        return (
            f"{feature_name} supported the {predicted_name} prediction. The positive influence "
            f"means this feature increased the model's support for {predicted_name} relative "
            f"to the clinical baseline. Its magnitude was {strength}."
        )
    if value < -0.01:
        return (
            f"{feature_name} opposed the {predicted_name} prediction. The negative influence "
            f"means this feature reduced the model's support for {predicted_name} relative "
            f"to the clinical baseline. Its magnitude was {strength}."
        )
    return (
        f"{feature_name} had a very small influence on the {predicted_name} prediction; "
        "its Shapley value was close to zero."
    )


# ================================================================================================================
# HUMAN-FRIENDLY CLINICAL INFLUENCE GRAPH
# ================================================================================================================

def make_clinical_influence_chart(
    age_value,
    sex_value
):

    """
    Visualization only.

    Positive value:
        supports predicted class

    Negative value:
        opposes predicted class

    Values are model-score contributions,
    NOT percentages.
    """

    labels = [
        "Age",
        "Sex"
    ]

    values = np.array(
        [
            age_value,
            sex_value
        ],
        dtype=np.float32
    )


    max_abs = max(
        float(
            np.max(
                np.abs(
                    values
                )
            )
        ),
        0.01
    )


    fig, ax = plt.subplots(
        figsize=(8.5, 3.2),
        dpi=160
    )


    y = np.arange(
        len(
            labels
        )
    )


    bar_colors = [
        "#2E8B57" if value > 0.01 else "#C94C4C" if value < -0.01 else "#7A8A99"
        for value in values
    ]

    bars = ax.barh(
        y,
        values,
        color=bar_colors,
        height=0.48
    )


    ax.axvline(
        0,
        linewidth=1.2
    )


    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        labels
    )


    ax.set_xlim(
        -max_abs * 1.35,
        max_abs * 1.35
    )


    ax.set_xlabel(
        "Influence on predicted result"
    )


    ax.set_title(
        "Clinical Feature Influence"
    )


    ax.grid(
        axis="x",
        alpha=0.15
    )


    for bar, value in zip(
        bars,
        values
    ):

        x = float(value)
        is_large_enough = abs(x) >= max_abs * 0.28

        if is_large_enough:
            label_x = x / 2
            label_color = "white"
            horizontal_alignment = "center"
        else:
            label_x = (
                max_abs * 0.10
                if x >= 0
                else -max_abs * 0.10
            )
            label_color = "#1C2630"
            horizontal_alignment = "left" if x >= 0 else "right"

        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=10,
            fontweight="semibold",
            color=label_color
        )


    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


    ax.text(
        0.5,
        -0.27,
        "Green = supports prediction    |    Red = opposes prediction    |    Gray = minimal influence",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#596773"
    )

    fig.tight_layout(pad=1.4)

    return fig


def influence_label(
    value
):

    if value > 0.01:

        return (
            "Supports the prediction"
        )

    if value < -0.01:

        return (
            "Opposes the prediction"
        )

    return (
        "Very small / nearly neutral influence"
    )




# ================================================================================================================
# INPUTS
# ================================================================================================================

st.markdown(
    '<div class="section">MRI Views</div>',
    unsafe_allow_html=True
)

st.caption(
    "Upload one clear, correctly oriented image for each anatomical plane. "
    "Supported formats: PNG, JPG, BMP, and TIFF."
)


u1, u2, u3 = st.columns(
    3,
    gap="medium"
)


with u1:

    axial_file = st.file_uploader(
        "Axial MRI",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff"
        ],
        key="axial"
    )


with u2:

    coronal_file = st.file_uploader(
        "Coronal MRI",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff"
        ],
        key="coronal"
    )


with u3:

    sagittal_file = st.file_uploader(
        "Sagittal MRI",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff"
        ],
        key="sagittal"
    )


st.markdown(
    '<div class="section">Clinical Information</div>',
    unsafe_allow_html=True
)

st.caption(
    "Clinical context is combined with the three MRI views during inference."
)


c1, c2 = st.columns(
    2
)


with c1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=110,
        value=59,
        step=1
    )


with c2:

    sex = st.selectbox(
        "Sex",
        [
            "female",
            "male"
        ]
    )


# ================================================================================================================
# ORIGINAL MRI PREVIEW
# ================================================================================================================

all_files = all(
    file is not None
    for file in [
        axial_file,
        coronal_file,
        sagittal_file
    ]
)


if all_files:

    st.markdown(
        '<div class="section">Uploaded MRI Preview</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Verify the original uploaded images before analysis."
    )


    preview1, preview2, preview3 = st.columns(
        3,
        gap="medium"
    )


    with preview1:

        st.image(
            axial_file,
            caption="Axial MRI",
            width="stretch"
        )


    with preview2:

        st.image(
            coronal_file,
            caption="Coronal MRI",
            width="stretch"
        )


    with preview3:

        st.image(
            sagittal_file,
            caption="Sagittal MRI",
            width="stretch"
        )




# ================================================================================================================
# CONFIRMATION
# ================================================================================================================

st.markdown(
    '<div class="section">Input Verification</div>',
    unsafe_allow_html=True
)

st.caption(
    "Complete these checks to confirm that all inputs belong to the same study."
)


same_patient_confirm = st.checkbox(
    "I confirm that axial, coronal and sagittal MRI views belong to the same patient and study."
)


orientation_confirm = st.checkbox(
    "I confirm that each MRI has been uploaded into the correct Axial / Coronal / Sagittal slot."
)


clinical_confirm = st.checkbox(
    "I confirm that Age and Sex belong to the same patient as the uploaded MRI views."
)


# ================================================================================================================
# PRE-VALIDATION
# ================================================================================================================

files = {
    "axial":
        axial_file,

    "coronal":
        coronal_file,

    "sagittal":
        sagittal_file,
}


all_files = all(
    file is not None
    for file in files.values()
)


hard_errors = []
warnings = []

processed_views = None


if all_files:

    raw_info = {}


    for view_name, file in files.items():

        try:

            info = inspect_raw_image(
                file
            )

            raw_info[
                view_name
            ] = info


            severe = severe_image_problem(
                info
            )

            if severe:

                hard_errors.append(
                    f"{view_name.title()}: {severe}"
                )


            warning = image_quality_warning(
                info
            )

            if warning:

                warnings.append(
                    f"{view_name.title()}: {warning}"
                )


        except Exception as e:

            hard_errors.append(
                f"{view_name.title()} image could not be read: {e}"
            )


        orientation = check_orientation_filename(
            file.name,
            view_name
        )


        if orientation[
            "status"
        ] == "mismatch":

            hard_errors.append(
                orientation[
                    "message"
                ]
            )


    if not hard_errors:

        try:

            processed_views = {
                view_name:
                    preprocess_uploaded(
                        file
                    )

                for view_name, file in files.items()
            }


            duplicates, duplicate_metrics = (
                detect_duplicate_views(
                    processed_views[
                        "axial"
                    ],
                    processed_views[
                        "coronal"
                    ],
                    processed_views[
                        "sagittal"
                    ]
                )
            )


            if duplicates:

                hard_errors.append(
                    "Near-duplicate MRI views detected: "
                    +
                    ", ".join(
                        duplicates
                    )
                    +
                    ". Each slot should contain its correct anatomical view."
                )


        except Exception as e:

            hard_errors.append(
                f"MRI preprocessing failed: {e}"
            )


age_check = validate_age(
    age
)


if age_check[
    "block"
]:

    hard_errors.append(
        age_check[
            "message"
        ]
    )


elif age_check[
    "warning"
]:

    warnings.append(
        age_check[
            "message"
        ]
    )


# ================================================================================================================
# INPUT STATUS
# ================================================================================================================

if all_files:

    if hard_errors:

        for message in hard_errors:

            st.error(
                message
            )


    else:

        st.markdown(
            """
<div class="guard-pass">
✅ Uploaded files passed the automatic technical checks.
</div>
""",
            unsafe_allow_html=True
        )


        st.caption(
            "Automatic checks cannot prove that three ordinary PNG/JPG images belong to the same patient. "
            "That is why patient/study confirmation is required."
        )


        if warnings:

            for message in warnings:

                st.warning(
                    message
                )


# ================================================================================================================
# READY STATE
# ================================================================================================================

ready = (
    all_files
    and
    not hard_errors
    and
    same_patient_confirm
    and
    orientation_confirm
    and
    clinical_confirm
)


if all_files and not ready and not hard_errors:

    st.info(
        "Complete the three verification confirmations to enable analysis."
    )


# ================================================================================================================
# ANALYZE BUTTON
# ================================================================================================================

analyze = st.button(
    "🔬 Analyze MRI",
    type="primary",
    width="stretch",
    disabled=not ready
)


if analyze:

    try:

        with st.status(
            "Analyzing patient...",
            expanded=True
        ) as status:


            st.write(
                "Input validation passed."
            )

            st.write(
                "Running experimental Tumor / No-Tumor screening..."
            )

            screening = run_binary_screening(processed_views)
            st.write("Stage-1 Tumor score:", f"{screening['tumor_probability']:.2%}")
            st.caption(
                "Experimental screening only. Not clinically validated; "
                "the training datasets contain source-related confounding."
            )
            if not screening["screen_positive"]:
                status.update(label="Screening complete", state="complete", expanded=True)
                st.info(
                    "Research screening result: NO TUMOR pattern detected. "
                    "The four-class model and Grad-CAM were not run."
                )
                st.warning("This does not rule out a tumor. Clinical assessment is required.")
                st.stop()

            st.write("Tumor screen positive: running classification and explanations.")

            result = predict_patient(
                model=model,
                device=device,
                axial_image=processed_views[
                    "axial"
                ],
                coronal_image=processed_views[
                    "coronal"
                ],
                sagittal_image=processed_views[
                    "sagittal"
                ],
                age=float(age),
                sex=sex,
                generate_gradcam=True,
                generate_shapley=True
            )


            st.write(
                "Generating explanations..."
            )


            status.update(
                label="Analysis complete",
                state="complete",
                expanded=False
            )


        predicted_name = result["predicted_class"].title()
        confidence = float(result["confidence"])
        confidence_percent = confidence * 100.0

        # 1. PATIENT INPUT SUMMARY
        st.markdown(
            '<div class="section">Patient Input Summary</div>',
            unsafe_allow_html=True
        )
        summary_cols = st.columns(3)
        summary_cols[0].metric("Age", f"{float(age):.0f} years")
        summary_cols[1].metric("Sex", str(sex).title())
        summary_cols[2].metric("MRI views", "3 validated views")
        st.caption("Inputs used: axial, coronal and sagittal MRI views together with Age and Sex.")

        # 2. AI PREDICTION
        st.markdown(
            '<div class="section">AI Prediction</div>',
            unsafe_allow_html=True
        )


        r1, r2 = st.columns(
            2
        )


        with r1:

            st.markdown(
                f"""
<div class="result-card">
<div class="result-label">Predicted Tumor Type</div>
<div class="result-value">{predicted_name}</div>
</div>
""",
                unsafe_allow_html=True
            )


        with r2:

            st.markdown(
                f"""
<div class="result-card">
<div class="result-label">Prediction Confidence</div>
<div class="result-value">{result["confidence"] * 100:.2f}%</div>
</div>
""",
                unsafe_allow_html=True
            )


        # 3. CLASS PROBABILITIES
        st.markdown(
            '<div class="section">Class Probabilities</div>',
            unsafe_allow_html=True
        )


        prob_cols = st.columns(
            4
        )


        for index, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = float(
                result[
                    "probabilities"
                ][
                    class_name
                ]
            )


            with prob_cols[
                index
            ]:

                st.metric(
                    class_name.title(),
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    probability
                )


        # 4. GRAD-CAM ANALYSIS
        st.markdown(
            '<div class="section">Grad-CAM Analysis</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Grad-CAM does not show the exact tumor boundary and is not a segmentation map. "
            "It shows which regions influenced the model's classification decision."
        )

        cam_summaries = {}
        for view_name in ("axial", "coronal", "sagittal"):
            image = processed_views[view_name]
            cam = result["gradcam"][view_name]
            cam_summaries[view_name] = gradcam_summary(image, cam)

        render_gradcam_comparison(
            processed_views,
            result["gradcam"]
        )

        # 6. GRAD-CAM COLOR LEGEND
        st.markdown(
            '<div class="section">Grad-CAM Color Legend</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            """
<div class="guard-info">
🔴 <b>Red / dark red:</b> highest model attention; the region contributed strongly to the predicted class.<br>
🟠 <b>Orange / yellow:</b> high model attention.<br>
🟢 <b>Green:</b> moderate model attention.<br>
🔵 <b>Blue:</b> low model attention.<br>
⚫ <b>Dark / no heat:</b> very low model attention.<br><br>
The model focused strongly on red and yellow regions. These colors do not confirm tumor tissue.
</div>
""",
            unsafe_allow_html=True
        )

        # 7. PATIENT-SPECIFIC GRAD-CAM INTERPRETATION
        st.markdown(
            '<div class="section">Patient-Specific Grad-CAM Interpretation</div>',
            unsafe_allow_html=True
        )
        for view_name, summary in cam_summaries.items():
            st.markdown(
                f"**{view_name.title()} view:** The strongest AI attention is located in the "
                f"**{summary['location']} image region**. Red and yellow regions contributed most "
                f"strongly to the predicted **{predicted_name}** class; green regions contributed "
                f"moderately and blue regions had comparatively low influence. Within the visible "
                f"MRI area, **{summary['high']:.1f}%** had high activation, "
                f"**{summary['moderate']:.1f}%** moderate activation and "
                f"**{summary['low']:.1f}%** low activation. This is an image-coordinate, "
                "model-attention description, not confirmed anatomical localization."
            )

        # CLINICAL INFLUENCE
        st.markdown(
            '<div class="section">Clinical Influence</div>',
            unsafe_allow_html=True
        )
        age_value = float(result["clinical_shapley"]["age_shap_logit"])
        sex_value = float(result["clinical_shapley"]["sex_shap_logit"])
        combined_value = age_value + sex_value

        def simple_influence_status(value):
            if value > 0.01:
                return "Supports prediction"
            if value < -0.01:
                return "Opposes prediction"
            return "Very small influence"

        influence_col1, influence_col2 = st.columns(2, gap="medium")
        with influence_col1:
            st.metric("Age Influence", simple_influence_status(age_value))
            st.caption(f"Shapley contribution: {age_value:+.4f}")
        with influence_col2:
            st.metric("Sex Influence", simple_influence_status(sex_value))
            st.caption(f"Shapley contribution: {sex_value:+.4f}")

        # 10–11. FEATURE-SPECIFIC SHAPLEY EXPLANATIONS
        st.markdown("#### Age SHAP/Shapley explanation")
        st.write(shapley_explanation("Age", age_value, predicted_name))
        st.markdown("#### Sex SHAP/Shapley explanation")
        st.write(shapley_explanation("Sex", sex_value, predicted_name))
        st.caption(
            "A positive value supports the current predicted class relative to the clinical baseline; "
            "a negative value opposes it. These values explain model behavior and do not establish a diagnosis."
        )

        # 12. COMBINED CLINICAL INTERPRETATION
        st.markdown(
            '<div class="section">Combined Clinical Interpretation</div>',
            unsafe_allow_html=True
        )
        age_direction = "supported" if age_value > 0.01 else "opposed" if age_value < -0.01 else "had minimal effect on"
        sex_direction = "supported" if sex_value > 0.01 else "opposed" if sex_value < -0.01 else "had minimal effect on"
        combined_description = (
            "provided additional support" if combined_value > 0.01
            else "slightly reduced support" if combined_value < -0.01
            else "had a relatively small net effect"
        )
        st.write(
            f"Age {age_direction} the predicted {predicted_name} class, while Sex {sex_direction} it. "
            f"Together, the clinical information {combined_description}. The final prediction remained "
            "primarily driven by MRI imaging features."
        )
        fig = make_clinical_influence_chart(age_value, sex_value)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        # 13. OVERALL AI INTERPRETATION
        st.markdown(
            '<div class="section">Overall AI Interpretation</div>',
            unsafe_allow_html=True
        )
        locations = ", ".join(
            f"{name.title()}: {summary['location']}" for name, summary in cam_summaries.items()
        )
        st.write(
            f"The AI model analyzed all three MRI views together with Age and Sex. It assigned the "
            f"highest probability to **{predicted_name}** with **{confidence_percent:.2f}% confidence**. "
            f"The strongest Grad-CAM attention occurred at these image-coordinate locations: {locations}. "
            f"Age {age_direction} the predicted class and Sex {sex_direction} it. Overall, the model's "
            "decision was primarily driven by MRI features, with clinical information providing a secondary influence."
        )

        # 14. PATIENT-FRIENDLY EXPLANATION
        st.markdown(
            '<div class="section">What This Result Means</div>',
            unsafe_allow_html=True
        )
        st.write(
            f"Your three MRI views were analyzed together with your Age and Sex information. The AI model "
            f"found patterns most similar to the **{predicted_name}** class. Red and yellow areas show where "
            "the AI focused most strongly. These highlighted areas are not confirmed tumor boundaries and "
            "should be reviewed by a qualified medical specialist."
        )

        # 15. SUGGESTED NEXT STEPS
        st.markdown(
            '<div class="section">Suggested Next Steps</div>',
            unsafe_allow_html=True
        )
        st.warning("This AI output is an assistive screening/research result and not a final diagnosis.")
        st.markdown(
            """
- Show the complete original MRI study to a qualified radiologist; prefer original DICOM/NIfTI scans over screenshots.
- Compare the study with previous MRI examinations when available.
- Ask the clinician or radiologist to evaluate lesion location, size, enhancement pattern, edema and surrounding structures.
- Additional contrast imaging or follow-up should be decided only by the treating clinician.
- Histopathology, where clinically required, remains the definitive method for many tumor diagnoses.
- Do not start, stop or change treatment based only on this AI dashboard.
"""
        )

        # 16. MEDICAL DISCLAIMER
        st.markdown(
            '<div class="section">Medical Disclaimer</div>',
            unsafe_allow_html=True
        )
        st.error(
            "This system is developed for research and AI-assisted decision support. It is not a standalone "
            "diagnostic tool and does not replace evaluation by a qualified radiologist, neurologist, "
            "neurosurgeon, oncologist or other medical professional."
        )

        # 17. PRINTABLE REPORT
        st.markdown(
            '<div class="section">Printable Patient Report</div>',
            unsafe_allow_html=True
        )

        report_pdf = build_patient_report(
            age=age,
            sex=sex,
            predicted_name=predicted_name,
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            processed_views=processed_views,
            gradcam=result["gradcam"],
            cam_summaries=cam_summaries,
            age_shapley=age_value,
            sex_shapley=sex_value,
        )

        st.download_button(
            "Download / Print PDF Report",
            data=report_pdf,
            file_name="brain_tumor_ai_assessment_report.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
        )

        st.caption(
            "Open the downloaded PDF and use your device's Print command. "
            "The report includes natural-language explanations for patients and clinicians."
        )



    except Exception as error:

        st.error(
            "Analysis could not be completed."
        )

        with st.expander(
            "Technical details"
        ):

            st.exception(
                error
            )


st.divider()

st.markdown(
    """
<div class="footer">
Brain Tumor XAI • Research prototype — not intended for clinical diagnosis
</div>
""",
    unsafe_allow_html=True
)




