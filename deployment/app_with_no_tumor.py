from datetime import datetime, timezone
from io import BytesIO

import hashlib
from html import escape


from pathlib import Path
import sys
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


from deployment.research_report import (
    SHAP_GUIDE,
    clinical_sentence, interpretation, next_steps, build_research_report, REPORT_VERSION,
)

# ================================================================================================================
# PAGE
# ================================================================================================================

st.set_page_config(
    page_title="Brain Tumor XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Styles are bundled locally for offline and Community Cloud compatibility.
st.markdown("<style>" + (Path(__file__).parent / "assets/dashboard.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
st.markdown("""
<header class="app-header"><div class="brand"><span class="brain-mark" aria-hidden="true">🧠</span><div>
<h1>Brain Tumor XAI</h1><p>Multi-angle MRI classification with image and clinical explanations</p></div></div></header>
""", unsafe_allow_html=True)

# ================================================================================================================
# MODEL
# ================================================================================================================

@st.cache_resource
def get_model():

    return load_final_model()


# ============================================================
# NEW: THREE-IMAGE TUMOR / NO-TUMOR RESEARCH GATE
# ============================================================

@st.cache_resource
def load_binary_screening_model():
    import torch
    from src.stage1_three_image_binary import Stage1ThreeImageBinary

    path = (
        ROOT / "checkpoints/stage1_three_image_v3_corrected/best.pt"
    )

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    assert checkpoint.get("version") == "stage1_v3_corrected", "Unexpected Stage-1 checkpoint version"

    screening_model = Stage1ThreeImageBinary()
    screening_model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    screening_model.eval()

    return screening_model, float(checkpoint["threshold"])


def run_binary_screening(processed_views):
    import torch

    order = ("axial", "coronal", "sagittal")

    three_images = np.stack([
        np.asarray(processed_views[name], dtype=np.float32)
        for name in order
    ])

    if three_images.shape != (3, 224, 224):
        raise ValueError(
            f"Invalid screening input: {three_images.shape}"
        )

    if not np.isfinite(three_images).all():
        raise ValueError("MRI contains non-finite pixels")

    if three_images.min() < 0 or three_images.max() > 1:
        raise ValueError("MRI values must be within [0,1]")

    screening_model, threshold = load_binary_screening_model()

    tensor = torch.from_numpy(
        np.ascontiguousarray(three_images)
    ).unsqueeze(0)

    with torch.inference_mode():
        logit = screening_model(tensor)
        probability = float(torch.sigmoid(logit).item())

    return {
        "tumor_probability": probability,
        "threshold": threshold,
        "screen_positive": probability >= threshold,
    }


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
    raw_cam
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
    alpha_map = (
        0.55
        *
        anatomy
    )


    fig, ax = plt.subplots(
        figsize=(
            5.2,
            5.2
        )
    )


    # MRI base
    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="bicubic"
    )


    # Smooth full Grad-CAM
    ax.imshow(
        visible_cam,
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=alpha_map,
        interpolation="bicubic"
    )


    # Strong-attention contour
    contour_cam = np.ma.masked_where(
        anatomy < 0.08,
        cam
    )

    if np.any(
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
        figsize=(8, 2.8)
    )


    y = np.arange(
        len(
            labels
        )
    )


    bars = ax.barh(
        y,
        values
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

        direction = (
            "supports"
            if value > 0
            else
            "opposes"
            if value < 0
            else
            "neutral"
        )

        x = float(
            value
        )

        offset = (
            max_abs * 0.04
        )

        ax.text(
            x + (
                offset
                if x >= 0
                else -offset
            ),
            bar.get_y()
            +
            bar.get_height() / 2,
            f"{value:+.4f}  ({direction})",
            va="center",
            ha=(
                "left"
                if x >= 0
                else "right"
            ),
            fontsize=10
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


    fig.tight_layout()

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




# UI orchestration only: all inference and input checks use the original functions.
VIEW_NAMES = ("axial", "coronal", "sagittal")
STEPS = ("Input Validation", "Stage-1 Screening", "Stage-2 Classification", "Explainability")


def section(number, title, subtitle):
    st.markdown(f'<div class="section-heading"><span>{number}</span><h3>{escape(title)}</h3></div>', unsafe_allow_html=True)
    st.caption(subtitle)


def probability_bar(label, value, leading=False):
    value = float(value)
    st.markdown(
        f'<div class="probability {"leading" if leading else ""}"><div><span>{escape(label)}</span>'
        f'<strong>{value:.2%}</strong></div><div class="probability-track" role="meter" '
        f'aria-label="{escape(label)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{value * 100:.2f}">'
        f'<span style="width:{value * 100:.4f}%"></span></div></div>', unsafe_allow_html=True,
    )


def workflow(states):
    rows = ''.join(
        f'<div class="workflow-step"><span class="step-number">{i + 1:02d}</span>'
        f'<span>{name}</span><strong class="state-{state.lower().replace(" ", "-")}">{state}</strong></div>'
        for i, (name, state) in enumerate(zip(STEPS, states))
    )
    return f'<div class="workflow">{rows}</div>'


left = st.container()
right = st.container()
with left:
    with st.container(border=False, key="upload_panel"):
        section("01", "Upload MRI Views", "Upload three MRI images from the same patient and study.")
        files = {}
        for column, view, glyph in zip(st.columns(3), VIEW_NAMES, ("⊙", "◫", "◧")):
            with column:
                st.markdown(f'<div class="upload-label"><span>{glyph}</span> {view.title()} MRI</div>', unsafe_allow_html=True)
                files[view] = st.file_uploader(
                    f"{view.title()} MRI", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                    key=view, label_visibility="collapsed",
                )
                uploaded = files[view]
                if uploaded is not None:
                    st.caption(f"Uploaded · {uploaded.size / 1024:.1f} KB")
                    st.caption(uploaded.name)
                else:
                    st.caption(f"Required · {view.title()} view")
        st.caption("PNG / JPG / JPEG · Existing BMP / TIFF support retained. Use × to remove or browse to replace.")

    with st.container(border=False, key="clinical_panel"):
        section("02", "Clinical Information", "Provide the clinical inputs used by the research model.")
        age_column, sex_column = st.columns(2)
        with age_column:
            age = st.number_input("Age (years)", min_value=1, max_value=110, value=59, step=1)
        with sex_column:
            sex = st.selectbox("Sex", ["female", "male"], format_func=str.title)
        st.caption("ⓘ Age and Sex contribute to the multimodal model and its interpretation.")

    with st.container(border=False, key="preview_panel"):
        section("03", "Uploaded MRI Preview", "Verify the original uploaded images before analysis.")
        for column, view in zip(st.columns(3), VIEW_NAMES):
            with column:
                uploaded = files[view]
                if uploaded is None:
                    st.markdown('<div class="image-empty"><span>＋</span>Awaiting MRI</div>', unsafe_allow_html=True)
                    st.caption(f"{view.title()} MRI")
                else:
                    try:
                        st.image(uploaded.getvalue(), caption=f"{view.title()} MRI", use_container_width=True)
                    except Exception:
                        st.error(f"{view.title()} preview unavailable. Replace this image.")

    verification = st.container(border=False, key="verification_panel")
    with verification:
        section("04", "Input Verification", "Confirm the information before running research analysis.")
        # Changing any source input clears attestations and results for the previous study.
        fingerprint = hashlib.sha256()
        for view in VIEW_NAMES:
            uploaded = files[view]
            fingerprint.update(view.encode())
            if uploaded is not None:
                fingerprint.update(uploaded.name.encode())
                fingerprint.update(uploaded.getbuffer())
        fingerprint.update(f"{age}:{sex}".encode())
        input_key = fingerprint.hexdigest()
        if st.session_state.get("study_input_key") != input_key:
            st.session_state.study_input_key = input_key
            st.session_state.pop("study_analysis", None)
            for key in ("confirm_patient", "confirm_orientation", "confirm_clinical"):
                st.session_state[key] = False
        same_patient_confirm = st.checkbox("I confirm that the Axial, Coronal, and Sagittal MRI views belong to the same patient and study.", key="confirm_patient")
        orientation_confirm = st.checkbox("I confirm that each MRI was uploaded into its correct view slot.", key="confirm_orientation")
        clinical_confirm = st.checkbox("I confirm that Age and Sex belong to the same patient as the uploaded MRI views.", key="confirm_clinical")

all_files = all(file is not None for file in files.values())
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



with left:
    with verification:
        for message in hard_errors:
            st.error(message)
        for message in warnings:
            st.warning(message)
        if all_files and not hard_errors:
            st.success("Uploaded files passed the automatic technical checks.")
        elif not all_files:
            st.info("Required views: " + ", ".join(view.title() for view in VIEW_NAMES if files[view] is None))
        st.caption("Automatic checks cannot prove that three ordinary PNG/JPG files belong to the same patient. Your confirmations are required.")
        ready = all_files and not hard_errors and same_patient_confirm and orientation_confirm and clinical_confirm
        if all_files and not hard_errors and not ready:
            st.caption("Complete all three confirmations to enable analysis.")
        analyze = st.button("🔬 Analyze MRI", type="primary", use_container_width=True, disabled=not ready)



with right:
    st.markdown('<div class="output-divider"><span>ANALYSIS &amp; OUTPUT</span></div>', unsafe_allow_html=True)
    with st.container(border=False, key="status_panel"):
        section("05", "Analysis Status", "Follow the actual research workflow.")
        status_slot = st.empty()
        saved = st.session_state.get("study_analysis")
        states = saved["states"][:] if saved else ["Failed" if hard_errors else "Not Started", "Not Started", "Not Started", "Not Started"]
        status_slot.markdown(workflow(states), unsafe_allow_html=True)

        if analyze:
            # Replace the old result before execution so failures cannot expose stale predictions.
            saved = {
                "states": ["Completed", "Running", "Not Started", "Not Started"],
                "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "age": int(age), "sex": sex, "warnings": list(warnings),
            }
            st.session_state.study_analysis = saved
            states = saved["states"]
            status_slot.markdown(workflow(states), unsafe_allow_html=True)
            active_stage = 1
            try:
                with st.spinner("Running Stage-1 screening…"):
                    screening = run_binary_screening(processed_views)
                saved["screening"] = screening
                states[1] = "Completed"
                if not screening["screen_positive"]:
                    states[2:] = ["Skipped", "Skipped"]
                else:
                    states[2] = "Running"
                    active_stage = 2
                    status_slot.markdown(workflow(states), unsafe_allow_html=True)
                    with st.spinner("Loading model and classifying the three MRI views…"):
                        model, checkpoint, device = get_model()
                        prediction_inputs = dict(
                            model=model, device=device,
                            axial_image=processed_views["axial"], coronal_image=processed_views["coronal"],
                            sagittal_image=processed_views["sagittal"], age=float(age), sex=sex,
                        )
                        # The existing entry point supports inference without optional explanations.
                        # Retain this real classification if the subsequent XAI call fails.
                        saved["result"] = predict_patient(**prediction_inputs, generate_gradcam=False, generate_shapley=False)
                    states[2] = "Completed"
                    states[3] = "Running"
                    active_stage = 3
                    status_slot.markdown(workflow(states), unsafe_allow_html=True)
                    with st.spinner("Generating three 2D Grad-CAM views and clinical contributions…"):
                        explained = predict_patient(**prediction_inputs, generate_gradcam=True, generate_shapley=True)
                        saved["result"] = explained
                    states[3] = "Completed"
            except Exception:
                states[active_stage] = "Failed"
                for i in range(active_stage + 1, 4):
                    states[i] = "Skipped"
                # No exception dumps: paths or uploaded patient information may be present.
                saved["error"] = (
                    "Explainability generation failed. The completed classification is shown below; explanations are unavailable."
                    if active_stage == 3 else
                    "Model inference failed. Check checkpoint availability and the server environment, then retry analysis."
                )
            status_slot.markdown(workflow(states), unsafe_allow_html=True)

        if saved and saved.get("error"):
            st.error(saved["error"])
        if saved is None:
            st.markdown('<div class="results-empty"><span class="empty-orbit">◎</span><h3>Ready to analyze</h3><p>Upload three MRI views and complete input verification.</p><small>Real model outputs will appear here.</small></div>', unsafe_allow_html=True)

    if saved and "screening" in saved:
        screening = saved["screening"]
        positive = screening["screen_positive"]
        with st.container(border=True):
            st.markdown('<div class="eyebrow">STAGE-1 · SCREENING COMPLETE</div>', unsafe_allow_html=True)
            if positive:
                st.markdown(
                    '<div class="negative-result-card positive-result-card">'
                    '<div class="negative-result-heading"><span class="negative-check" aria-hidden="true">!</span>'
                    '<div><h3>Tumor Detected</h3><span>Screen Positive · Experimental Stage-1 screening</span></div></div>'
                    '<p>Tumor suspected by the experimental Stage-1 screening model in the uploaded MRI views.</p>'
                    '<div class="negative-result-note"><span aria-hidden="true">ⓘ</span>'
                    ' This AI-assisted research result should be reviewed by a doctor. A positive screen is not a confirmed diagnosis; the system is not clinically validated.</div>'
                    '</div>', unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="negative-result-card">'
                    '<div class="negative-result-heading"><span class="negative-check" aria-hidden="true">✓</span>'
                    '<div><h3>No Tumor Detected</h3><span>Screen Negative · Experimental Stage-1 screening</span></div></div>'
                    '<p>No tumor detected by the experimental Stage-1 screening model in the uploaded MRI views.</p>'
                    '<div class="negative-result-note"><span aria-hidden="true">ⓘ</span>'
                    ' This AI-assisted research result should be reviewed by a doctor. A negative screen does not clinically rule out a tumor.</div>'
                    '</div>', unsafe_allow_html=True,
                )

        if positive and "result" in saved:
            result = saved["result"]
            with st.container(border=True):
                section("06", "Tumor Classification", "Experimental classification within the four tumor classes.")
                name_column, score_column = st.columns(2)
                with name_column:
                    st.metric("Predicted Tumor Type", result["predicted_class"].title())
                with score_column:
                    st.metric("Prediction Confidence", f"{result['confidence']:.2%}")
                st.markdown("#### Class Probabilities")
                probability_cards = []
                for class_name in CLASS_NAMES:
                    probability = float(result["probabilities"][class_name])
                    leading = " is-leading" if class_name == result["predicted_class"] else ""
                    probability_cards.append(
                        f'<div class="class-probability{leading}"><div class="class-probability-card">'
                        f'<span>{escape(class_name.title())}</span><strong>{probability:.2%}</strong></div>'
                        f'<div class="class-probability-track" role="meter" aria-label="{escape(class_name.title())} probability" '
                        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{probability * 100:.2f}">'
                        f'<span style="width:{probability * 100:.4f}%"></span></div></div>'
                    )
                st.markdown('<div class="class-probability-grid">' + ''.join(probability_cards) + '</div>', unsafe_allow_html=True)

            if result.get("gradcam"):
                with st.container(border=True, key="gradcam_panel"):
                    section("07", "Explainability | Gradcam Visualization", "Visual interpretation of the model’s image attention.")
                    for column, view in zip(st.columns(3), VIEW_NAMES):
                        with column:
                            if view in result["gradcam"]:
                                fig = make_gradcam_overlay(processed_views[view], result["gradcam"][view])
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                                st.caption(f"{view.title()} attention")
                            else:
                                st.info(f"{view.title()} attention unavailable")
                    st.markdown('<div class="cam-scale"></div><div class="scale-labels"><span>Lower influence</span><span>Higher influence</span></div>', unsafe_allow_html=True)
                    st.markdown(
                        "**How to understand this heatmap**\n\n"
                        "The heatmap shows **where the AI looked while making its prediction**.\n\n"
                        "🔴 **Red:** strongest influence on the prediction  \n"
                        "🟡 **Yellow:** strong influence  \n"
                        "🟢 **Green:** moderate influence  \n"
                        "🔵 **Blue:** lower influence"
                    )

            if result.get("clinical_shapley"):
                with st.container(border=True, key="clinical_xai_panel"):
                    section("08", "Clinical SHAP Contributions", "Contribution to the predicted tumor-class logit.")
                    clinical = result["clinical_shapley"]
                    age_value = float(clinical["age_shap_logit"])
                    sex_value = float(clinical["sex_shap_logit"])
                    for column, label_text, input_text, contribution in zip(
                        st.columns(2), ("Age", "Sex"), (f"{age} years", sex.title()), (age_value, sex_value)
                    ):
                        with column:
                            st.metric(f"{label_text} · {input_text}", f"{contribution:+.4f}")
                            st.caption("Supports class score" if contribution > 0 else "Reduces class score" if contribution < 0 else "Neutral contribution")
                    st.markdown("#### Clinical Impact on Prediction")
                    with plt.rc_context({"text.color": "#202020", "axes.labelcolor": "#202020", "xtick.color": "#202020", "ytick.color": "#202020", "axes.edgecolor": "#202020", "font.size": 12}):
                        fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="white")
                        ax.set_facecolor("white")
                        values = [age_value, sex_value]
                        bars = ax.barh(["Age", "Sex"], values, color="#217FB5", height=.42)
                        max_abs = max(abs(age_value), abs(sex_value), .01)
                        ax.set_xlim(-max_abs * 1.45, max_abs * 1.45)
                        ax.axvline(0, color="#478FBE", linewidth=1.2)
                        ax.grid(axis="x", alpha=.12)
                        ax.set_axisbelow(True)
                        ax.set_xlabel("Effect on predicted tumor class (logit contribution)")
                        for bar, value in zip(bars, values):
                            ax.text(
                                value + (max_abs * .07 if value >= 0 else -max_abs * .07),
                                bar.get_y() + bar.get_height() / 2,
                                f"{value:+.4f}", va="center", ha="left" if value >= 0 else "right",
                                fontsize=12, color="#202020",
                            )
                        ax.spines[["top", "right"]].set_visible(False)
                        fig.tight_layout(pad=1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    st.markdown("#### What do these bars mean for this result?")
                    st.write(clinical_sentence("Age", age_value, result["predicted_class"]))
                    st.write(clinical_sentence("Sex", sex_value, result["predicted_class"]))
                    st.write(SHAP_GUIDE)


        with st.container(border=True):
            st.markdown("### Understanding your result")
            report_result = saved.get("result") if positive else None
            st.write(interpretation(screening, report_result))
            st.markdown("#### Suggested next steps")
            for step in next_steps(screening, report_result, saved.get("warnings", [])):
                st.write("• " + step)

        with st.container(border=True):
            st.markdown("### Download your research report")
            st.caption("Includes original MRI views, verified inputs, screening details, available explanations, and suggested next steps. Generated in memory for this analysis.")
            if saved.get("pdf_version") != REPORT_VERSION:
                saved.pop("pdf_bytes", None)
            if "analyzed_at" not in saved:
                st.info("Run analysis again to create a report with verified analysis details.")
            elif st.button("Prepare PDF report", use_container_width=True):
                try:
                    with st.spinner("Preparing your research report…"):
                        overlays = {}
                        for view, cam in (report_result or {}).get("gradcam", {}).items():
                            fig = make_gradcam_overlay(processed_views[view], cam)
                            try:
                                buffer = BytesIO()
                                fig.savefig(buffer, format="png", dpi=150, facecolor="black")
                                overlays[view] = buffer.getvalue()
                            finally:
                                plt.close(fig)
                        saved["pdf_bytes"] = build_research_report(
                            saved=saved,
                            original_images={view: upload.getvalue() for view, upload in files.items()},
                            overlay_images=overlays,
                        )
                        saved["pdf_version"] = REPORT_VERSION
                except Exception:
                    st.error("The PDF could not be prepared. Your analysis is still available; please retry.")
            if saved.get("pdf_bytes"):
                st.download_button("Download PDF report", saved["pdf_bytes"],
                    file_name="brain_mri_research_report.pdf", mime="application/pdf", use_container_width=True)
                st.caption("The report contains supplied clinical information and MRI images. Share it only with intended recipients.")

st.markdown(
    '<section class="medical-disclaimer" aria-labelledby="medical-disclaimer-title">'
    '<h2 id="medical-disclaimer-title">Medical Disclaimer</h2>'
    '<div class="medical-disclaimer-card">'
    'This system is developed for research and AI-assisted decision support. It is not a standalone '
    'diagnostic tool and does not replace evaluation by a qualified radiologist, neurologist, neurosurgeon, '
    'oncologist or other medical professional.'
    '</div></section>', unsafe_allow_html=True,
)
st.markdown('<div class="footer">Brain Tumor XAI · Research prototype · Not intended for clinical diagnosis</div>', unsafe_allow_html=True)
