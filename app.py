# ==========================================
# ♻️ WASTE CLASSIFICATION AI
# STREAMLIT APP
# ==========================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="centered"
)


# ==========================================
# CSS
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# CLASS NAMES
# ==========================================

DEFAULT_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


def load_class_names():

    if os.path.exists("class_names.txt"):

        with open(
            "class_names.txt",
            "r"
        ) as file:

            classes = [
                line.strip()
                for line in file.readlines()
                if line.strip()
            ]

        return classes

    return DEFAULT_CLASSES


class_names = load_class_names()


# ==========================================
# MODEL INFORMATION
# ==========================================

MODEL_NAME = "MobileNetV2"
IMAGE_SIZE = "224 × 224"
NUM_CLASSES = len(class_names)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "waste_classifier.keras"
    )


try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Could not load waste_classifier.keras"
    )

    st.error(
        str(e)
    )

    st.info(
        "Make sure waste_classifier.keras "
        "is in the same folder as app.py."
    )

    st.stop()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">'
    '♻️ Waste Classification AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning based waste classification system'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header(
        "ℹ️ Model Information"
    )

    st.write(
        f"**Model:** {MODEL_NAME}"
    )

    st.write(
        f"**Classes:** {NUM_CLASSES}"
    )

    st.write(
        f"**Image Size:** {IMAGE_SIZE}"
    )

    st.markdown("---")

    st.subheader(
        "♻️ Waste Classes"
    )

    for name in class_names:

        st.write(
            f"• {name.capitalize()}"
        )

    st.markdown("---")

    st.caption(
        "Waste Classification AI Project"
    )


# ==========================================
# FILE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(

    "📤 Upload a waste image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ],

    key="waste_image"
)


# ==========================================
# IMAGE UPLOADED
# ==========================================

if uploaded_file is not None:

    # ======================================
    # OPEN IMAGE
    # ======================================

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ======================================
    # DISPLAY IMAGE
    # ======================================

    st.subheader(
        "🖼️ Uploaded Image"
    )

    st.image(
        original_image,
        caption="Your uploaded waste image",
        width="stretch"
    )


    # ======================================
    # PREPARE IMAGE
    # ======================================

    image = original_image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ======================================
    # PREDICTION
    # ======================================

    with st.spinner(
        "🤖 AI is analyzing the image..."
    ):

        prediction = model.predict(
            image_array,
            verbose=0
        )


    # ======================================
    # CHECK OUTPUT
    # ======================================

    if len(prediction[0]) != len(class_names):

        st.error(
            "❌ Model classes and class_names.txt "
            "do not match."
        )

        st.stop()


    # ======================================
    # PREDICTED INDEX
    # ======================================

    predicted_index = np.argmax(
        prediction[0]
    )


    # ======================================
    # PREDICTED CLASS
    # ======================================

    predicted_class = class_names[
        predicted_index
    ]


    # ======================================
    # CONFIDENCE
    # ======================================

    confidence = (
        float(
            prediction[0][predicted_index]
        ) * 100
    )


    # ======================================
    # TOP PREDICTION
    # ======================================

    st.markdown("---")

    st.subheader(
        "🏆 Top Prediction"
    )

    st.success(
        f"♻️ {predicted_class.capitalize()}"
    )

    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )


    # ======================================
    # CONFIDENCE MESSAGE
    # ======================================

    if confidence >= 80:

        st.success(
            "✅ High confidence prediction"
        )

    elif confidence >= 50:

        st.warning(
            "⚠️ Moderate confidence prediction"
        )

    else:

        st.warning(
            "⚠️ Low confidence prediction"
        )


    # ======================================
    # PROBABILITIES
    # ======================================

    st.markdown("---")

    st.subheader(
        "📊 Class Probabilities"
    )

    probabilities = (
        prediction[0] * 100
    )


    # ======================================
    # SORT
    # ======================================

    sorted_indices = np.argsort(
        probabilities
    )[::-1]


    # ======================================
    # DISPLAY ALL CLASSES
    # ======================================

    for index in sorted_indices:

        class_name = class_names[
            index
        ]

        probability = float(
            probabilities[index]
        )

        st.write(
            f"**{class_name.capitalize()}** — "
            f"{probability:.2f}%"
        )

        st.progress(
            min(
                int(round(probability)),
                100
            )
        )


    # ======================================
    # TOP 3
    # ======================================

    st.markdown("---")

    st.subheader(
        "🥇 Top 3 Predictions"
    )


    top_3_indices = sorted_indices[:3]


    for rank, index in enumerate(
        top_3_indices,
        start=1
    ):

        class_name = class_names[
            index
        ]

        probability = float(
            probabilities[index]
        )

        st.write(
            f"**{rank}. "
            f"{class_name.capitalize()}**"
        )

        st.progress(
            min(
                int(round(probability)),
                100
            )
        )

        st.caption(
            f"Probability: "
            f"{probability:.2f}%"
        )


    # ======================================
    # CLEAR BUTTON
    # ======================================

    st.markdown("---")

    if st.button(
        "🔄 Clear Image / Upload Another",
        use_container_width=True
    ):

        st.session_state.pop(
            "waste_image",
            None
        )

        st.rerun()


# ==========================================
# NO IMAGE
# ==========================================

else:

    st.info(
        "👆 Upload a waste image to start "
        "classification."
    )

    st.markdown("---")

    st.subheader(
        "♻️ Supported Waste Classes"
    )

    cols = st.columns(3)

    for i, name in enumerate(
        class_names
    ):

        with cols[i % 3]:

            st.write(
                f"**{name.capitalize()}**"
            )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Built with TensorFlow • MobileNetV2 • Streamlit"
)