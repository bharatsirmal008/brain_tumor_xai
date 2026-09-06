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

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero-title {
    font-size: 2.55rem;
    font-weight: 760;
    letter-spacing: -0.04rem;
    margin-bottom: 0.15rem;
}

.hero-subtitle {
    opacity: 0.68;
    margin-bottom: 2rem;
}

.section {
    font-size: 1.35rem;
    font-weight: 720;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
}

.result-card {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 18px;
    padding: 22px;
    background: rgba(128,128,128,.045);
}

.result-label {
    opacity: .64;
    font-size: .88rem;
}

.result-value {
    font-size: 2.0rem;
    font-weight: 760;
    margin-top: .25rem;
}

.guard-pass {
    border-radius: 12px;
    padding: 10px 14px;
    background: rgba(50,180,100,.10);
    border: 1px solid rgba(50,180,100,.22);
}

.guard-info {
    border-radius: 12px;
    padding: 10px 14px;
    background: rgba(60,130,220,.09);
    border: 1px solid rgba(60,130,220,.20);
}

.footer {
    text-align:center;
    opacity:.50;
    font-size:.78rem;
    padding-top:1rem;
}

.stButton > button {
    border-radius: 12px;
    min-height: 48px;
    font-weight: 650;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.18);
    padding: 15px;
    border-radius: 14px;
}

</style>
""",
    unsafe_allow_html=True
)


st.markdown(
    '<div class="hero-title">🧠 Brain Tumor XAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Multi-angle MRI classification with image and clinical explanations'
    '</div>',
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




# ================================================================================================================
# INPUTS
# ================================================================================================================

st.markdown(
    '<div class="section">MRI Views</div>',
    unsafe_allow_html=True
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
            use_container_width=True
        )


    with preview2:

        st.image(
            coronal_file,
            caption="Coronal MRI",
            use_container_width=True
        )


    with preview3:

        st.image(
            sagittal_file,
            caption="Sagittal MRI",
            use_container_width=True
        )




# ================================================================================================================
# CONFIRMATION
# ================================================================================================================

st.markdown(
    '<div class="section">Input Verification</div>',
    unsafe_allow_html=True
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
    use_container_width=True,
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
                "Running final classification model..."
            )


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


        # ========================================================================================================
        # PREDICTION
        # ========================================================================================================

        st.markdown(
            '<div class="section">Prediction Result</div>',
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
<div class="result-value">{result["predicted_class"].title()}</div>
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


        # ========================================================================================================
        # PROBABILITIES
        # ========================================================================================================

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


        # ========================================================================================================
        # GRAD-CAM
        # ========================================================================================================

        st.markdown(
            '<div class="section">AI Attention Map</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            """
<div class="guard-info">
<b>How to understand this heatmap</b><br><br>

The heatmap shows <b>where the AI looked while making its prediction</b>.

<br>🔴 <b>Red:</b> strongest influence on the prediction
<br>🟡 <b>Yellow:</b> strong influence
<br>🟢 <b>Green:</b> moderate influence
<br>🔵 <b>Blue:</b> lower influence

<br><br>
A red or yellow area may overlap an important tumor-related region,
but it does <b>not</b> mean that every red pixel is definitely tumor.
</div>
""",
            unsafe_allow_html=True
        )


        st.caption(
            "Grad-CAM explains model attention. It does not draw the exact tumor boundary."
        )


        g1, g2, g3 = st.columns(
            3
        )


        items = [
            (
                g1,
                "Axial",
                processed_views["axial"],
                result["gradcam"]["axial"]
            ),
            (
                g2,
                "Coronal",
                processed_views["coronal"],
                result["gradcam"]["coronal"]
            ),
            (
                g3,
                "Sagittal",
                processed_views["sagittal"],
                result["gradcam"]["sagittal"]
            ),
        ]


        for column, title, image, cam in items:

            with column:

                fig = make_gradcam_overlay(
                    image,
                    cam
                )

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(
                    fig
                )

                st.caption(
                    f"{title} attention"
                )


        # ========================================================================================================
        # CLINICAL EXPLANATION
        # ========================================================================================================

        st.markdown(
            '<div class="section">Clinical Information Influence</div>',
            unsafe_allow_html=True
        )


        age_value = float(
            result[
                "clinical_shapley"
            ][
                "age_shap_logit"
            ]
        )

        sex_value = float(
            result[
                "clinical_shapley"
            ][
                "sex_shap_logit"
            ]
        )

        combined_value = (
            age_value
            +
            sex_value
        )


        # ========================================================================================================
        # 1. KEEP THE PREVIOUS AGE / SEX INFLUENCE CARDS
        # ========================================================================================================

        def simple_influence_status(
            value
        ):

            if value > 0.01:
                return "Supports prediction"

            elif value < -0.01:
                return "Opposes prediction"

            else:
                return "Very small influence"


        influence_col1, influence_col2 = st.columns(
            2,
            gap="medium"
        )


        with influence_col1:

            st.metric(
                "Age Influence",
                simple_influence_status(
                    age_value
                )
            )

            st.caption(
                f"Clinical contribution: {age_value:+.4f}"
            )


        with influence_col2:

            st.metric(
                "Sex Influence",
                simple_influence_status(
                    sex_value
                )
            )

            st.caption(
                f"Clinical contribution: {sex_value:+.4f}"
            )


        # ========================================================================================================
        # 2. GRAPHICAL VIEW
        # ========================================================================================================

        st.markdown(
            "#### Clinical Impact on Prediction"
        )


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
            figsize=(
                8,
                2.8
            )
        )


        positions = np.arange(
            len(
                labels
            )
        )


        bars = ax.barh(
            positions,
            values,
            height=0.48
        )


        # Neutral reference
        ax.axvline(
            0,
            linewidth=1.2,
            alpha=0.8
        )


        ax.set_yticks(
            positions
        )

        ax.set_yticklabels(
            labels
        )


        ax.set_xlim(
            -max_abs * 1.45,
            max_abs * 1.45
        )


        ax.set_xlabel(
            "Effect on predicted tumor class"
        )


        # Cleaner labels — no overlapping text
        for bar, value in zip(
            bars,
            values
        ):

            text = f"{value:+.4f}"

            offset = (
                max_abs * 0.07
            )

            if value >= 0:

                x = (
                    float(value)
                    +
                    offset
                )

                align = "left"

            else:

                x = (
                    float(value)
                    -
                    offset
                )

                align = "right"


            ax.text(
                x,
                bar.get_y()
                +
                bar.get_height() / 2,
                text,
                va="center",
                ha=align,
                fontsize=10
            )


        ax.grid(
            axis="x",
            alpha=0.12
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


        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(
            fig
        )


        # ========================================================================================================
        # 3. HUMAN-FRIENDLY DYNAMIC EXPLANATIONS
        # ========================================================================================================

        predicted_name = (
            result[
                "predicted_class"
            ]
            .title()
        )

        confidence_percent = (
            float(
                result[
                    "confidence"
                ]
            )
            *
            100.0
        )


        def feature_sentence(
            feature_name,
            value
        ):

            if value > 0.01:

                return (
                    f"**{feature_name}:** This information "
                    f"supported the **{predicted_name}** prediction."
                )

            elif value < -0.01:

                return (
                    f"**{feature_name}:** This information reduced "
                    f"support for the **{predicted_name}** prediction."
                )

            else:

                return (
                    f"**{feature_name}:** This information had very little "
                    f"effect on the **{predicted_name}** prediction."
                )


        if combined_value > 0.01:

            combined_sentence = (
                f"Taken together, Age and Sex gave additional support "
                f"to the **{predicted_name}** prediction."
            )

        elif combined_value < -0.01:

            combined_sentence = (
                f"Taken together, Age and Sex slightly reduced support "
                f"for the **{predicted_name}** prediction."
            )

        else:

            combined_sentence = (
                f"Taken together, Age and Sex had only a small effect "
                f"on the **{predicted_name}** prediction."
            )


        # ========================================================================================================
        # 4. PARAGRAPH EXPLANATION
        # ========================================================================================================

        st.markdown(
            "#### How the clinical information affected this result"
        )


        st.markdown(
            f"""
The model combined the three MRI views with the patient's **Age and Sex**.

{feature_sentence("Age", age_value)}

{feature_sentence("Sex", sex_value)}

{combined_sentence}

The final model prediction was **{predicted_name}** with
**{confidence_percent:.2f}% confidence**. In this model, the MRI images
provide the main diagnostic evidence, while Age and Sex can strengthen,
weaken, or have very little effect on that image-based prediction.
"""
        )


        # ========================================================================================================
        # 5. PATIENT-SPECIFIC POINTS — NOT GRAPH INSTRUCTIONS
        # ========================================================================================================

        st.markdown(
            "#### Clinical contribution summary"
        )


        age_point = (
            "increased support"
            if age_value > 0.01
            else
            "reduced support"
            if age_value < -0.01
            else
            "made almost no difference"
        )


        sex_point = (
            "increased support"
            if sex_value > 0.01
            else
            "reduced support"
            if sex_value < -0.01
            else
            "made almost no difference"
        )


        combined_point = (
            "overall supported the MRI-based decision"
            if combined_value > 0.01
            else
            "overall slightly opposed the MRI-based decision"
            if combined_value < -0.01
            else
            "overall had minimal effect on the MRI-based decision"
        )


        st.markdown(
            f"""
- **Age {age_point}** for this prediction.
- **Sex {sex_point}** for this prediction.
- Together, the clinical information **{combined_point}**.
- The final result still considers **all three MRI views + Age + Sex together**.
"""
        )


        # ========================================================================================================
        # 6. SIMPLE INFORMATION NOTE
        # ========================================================================================================

        st.info(
            "Clinical information does not make a separate diagnosis. "
            "It modifies the evidence coming from the MRI images as part of the final combined prediction."
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




