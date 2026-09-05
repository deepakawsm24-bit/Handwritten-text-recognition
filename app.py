import os
import json
import pickle

import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps

from streamlit_drawable_konva import st_canvas


# ============================================================
# 1. APP CONFIG
# ============================================================

st.set_page_config(
    page_title="CRNN Handwritten Word Recognizer",
    page_icon="✍️",
    layout="wide",
)

IMG_HEIGHT = 64
IMG_WIDTH = 256
IMG_CHANNELS = 1
TIME_STEPS = 32
NUM_CLASSES = 80

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 2. FIND MODEL
# ============================================================

def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


MODEL_PATH = first_existing([
    os.path.join(
        BASE_DIR,
        "CRNN_CTC_Basemodel_prediction.keras"
    ),
    os.path.join(
        BASE_DIR,
        "baseline_model_deployment",
        "CRNN_CTC_Basemodel_prediction.keras"
    ),
])


# ============================================================
# 3. CHARACTER MAPPING
# ============================================================

NUM_TO_CHAR_PATHS = [
    os.path.join(BASE_DIR, "num_to_char.pkl"),
    os.path.join(BASE_DIR, "common", "num_to_char.pkl"),
    os.path.join(
        BASE_DIR,
        "baseline_model_deployment",
        "num_to_char.pkl"
    ),
]

CHAR_MAPPING_JSON = os.path.join(
    BASE_DIR,
    "char_mapping.json"
)


@st.cache_resource
def load_num_to_char():

    for path in NUM_TO_CHAR_PATHS:

        if os.path.exists(path):

            with open(path, "rb") as f:
                mapping = pickle.load(f)

            return {
                int(k): str(v)
                for k, v in mapping.items()
            }

    if os.path.exists(CHAR_MAPPING_JSON):

        with open(
            CHAR_MAPPING_JSON,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if (
            isinstance(data, dict)
            and "num_to_char" in data
        ):
            data = data["num_to_char"]

        if isinstance(data, dict):

            return {
                int(k): str(v)
                for k, v in data.items()
            }

    raise FileNotFoundError(
        "Character mapping not found. "
        "Add char_mapping.json or num_to_char.pkl."
    )


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_prediction_model():

    if MODEL_PATH is None:

        raise FileNotFoundError(
            "CRNN_CTC_Basemodel_prediction.keras "
            "was not found."
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


try:

    num_to_char = load_num_to_char()
    prediction_model = load_prediction_model()

except Exception as e:

    st.error("Model setup failed.")
    st.code(str(e))
    st.stop()


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

def distortion_free_resize(
    image,
    target_size=(IMG_WIDTH, IMG_HEIGHT)
):

    target_width, target_height = target_size

    image = tf.image.resize(
        image,
        size=(
            target_height,
            target_width
        ),
        preserve_aspect_ratio=True,
    )

    current_height = tf.shape(image)[0]
    current_width = tf.shape(image)[1]

    pad_height = (
        target_height - current_height
    )

    pad_width = (
        target_width - current_width
    )

    image = tf.pad(
        image,
        [
            [0, pad_height],
            [0, pad_width],
            [0, 0],
        ],
        constant_values=1.0,
    )

    return image


def crop_to_ink(pil_image):

    gray = ImageOps.grayscale(
        pil_image
    )

    arr = np.asarray(gray)

    # Dark pixels are handwriting
    ink_mask = arr < 200

    coordinates = np.argwhere(
        ink_mask
    )

    if coordinates.size == 0:
        return pil_image

    min_y = coordinates[:, 0].min()
    min_x = coordinates[:, 1].min()

    max_y = coordinates[:, 0].max()
    max_x = coordinates[:, 1].max()

    padding = 10

    min_y = max(
        0,
        min_y - padding
    )

    min_x = max(
        0,
        min_x - padding
    )

    max_y = min(
        arr.shape[0] - 1,
        max_y + padding
    )

    max_x = min(
        arr.shape[1] - 1,
        max_x + padding
    )

    return pil_image.crop(
        (
            min_x,
            min_y,
            max_x + 1,
            max_y + 1
        )
    )


def preprocess_pil_image(
    pil_image,
    auto_crop=True
):

    # Grayscale
    pil_image = pil_image.convert("L")

    # Same crop for Upload and Draw
    if auto_crop:

        pil_image = crop_to_ink(
            pil_image
        )

    arr = np.asarray(
        pil_image,
        dtype=np.uint8
    )

    # Add channel
    image = tf.convert_to_tensor(
        arr[..., None],
        dtype=tf.uint8
    )

    # Normalize to [0,1]
    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    # Resize + white padding
    image = distortion_free_resize(
        image
    )

    # Add batch dimension
    image = tf.expand_dims(
        image,
        axis=0
    )

    return image, pil_image


# ============================================================
# 6. CTC DECODER
# ============================================================

def decode_prediction(predictions):

    input_length = np.full(
        predictions.shape[0],
        predictions.shape[1]
    )

    decoded, _ = tf.keras.backend.ctc_decode(
        predictions,
        input_length=input_length,
        greedy=True,
    )

    decoded = decoded[0].numpy()

    words = []

    for sequence in decoded:

        text = ""

        for token in sequence:

            token = int(token)

            if token == -1:
                continue

            if token in num_to_char:

                text += num_to_char[token]

        words.append(text)

    return words


# ============================================================
# 7. COMMON PREDICTION FUNCTION
# ============================================================

def predict_word(
    pil_image,
    auto_crop=True
):

    # SAME preprocessing
    processed_image, cropped_image = (
        preprocess_pil_image(
            pil_image,
            auto_crop=auto_crop
        )
    )

    # SAME CRNN model
    predictions = prediction_model.predict(
        processed_image,
        verbose=0
    )

    # SAME CTC decoder
    predicted_words = decode_prediction(
        predictions
    )

    if predicted_words:

        predicted_text = predicted_words[0]

    else:

        predicted_text = ""

    return (
        predicted_text,
        cropped_image,
        processed_image
    )


# ============================================================
# 8. MAIN UI
# ============================================================

st.title(
    "✍️ CRNN Handwritten Word Recognizer"
)

st.write(
    "Recognize a single handwritten English "
    "word using a CRNN + CTC model."
)

st.info(
    "For best results, use one handwritten word "
    "per image. Avoid screenshots containing "
    "extra text, borders, or multiple words."
)


# ============================================================
# CREATE TABS — ONLY ONCE
# ============================================================

tab_upload, tab_draw = st.tabs(
    [
        "📤 Upload",
        "✍️ Draw"
    ]
)


# ============================================================
# 9. UPLOAD TAB
# ============================================================

with tab_upload:

    st.header(
        "Upload a Handwritten Word"
    )

    st.write(
        "Upload an image containing a "
        "single handwritten word."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ],
        help="Use one handwritten word per image.",
        key="upload_file"
    )

    auto_crop_upload = st.checkbox(
        "Automatically crop handwriting",
        value=True,
        key="auto_crop_upload"
    )

    if uploaded_file is not None:

        try:

            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                uploaded_image,
                caption="Uploaded image",
                width=500
            )

            if st.button(
                "🔎 Recognize Word",
                key="recognize_upload",
                type="primary"
            ):

                with st.spinner(
                    "Recognizing..."
                ):

                    predicted, cropped, processed = (
                        predict_word(
                            uploaded_image,
                            auto_crop=auto_crop_upload
                        )
                    )

                st.image(
                    cropped,
                    caption="Image used for recognition",
                    width=500
                )

                st.subheader(
                    "Prediction"
                )

                if predicted:

                    st.success(
                        predicted
                    )

                else:

                    st.warning(
                        "[empty prediction]"
                    )

        except Exception as e:

            st.error(
                "Could not process this image."
            )

            st.exception(e)


# ============================================================
# 10. DRAW TAB
# ============================================================

with tab_draw:

    st.header(
        "Draw a Handwritten Word"
    )

    st.write(
        "Draw one English word inside "
        "the canvas."
    )

    canvas_result = st_canvas(

        fill_color=(
            "rgba(255, 255, 255, 0)"
        ),

        stroke_width=7,

        stroke_color="#000000",

        background_color="#FFFFFF",

        height=220,

        width=700,

        drawing_mode="freedraw",

        key="word_canvas"
    )

    auto_crop_draw = st.checkbox(
        "Automatically crop handwriting",
        value=True,
        key="auto_crop_draw"
    )

    if st.button(
        "🔎 Recognize Word",
        key="recognize_draw",
        type="primary"
    ):

        if canvas_result.image_data is None:

            st.warning(
                "Please draw a word first."
            )

        else:

            rgba = np.asarray(
                canvas_result.image_data
            ).astype(np.uint8)

            draw_image = Image.fromarray(
                rgba[..., :3],
                mode="RGB"
            )

            gray = np.asarray(
                ImageOps.grayscale(
                    draw_image
                )
            )

            if np.all(gray > 245):

                st.warning(
                    "Please draw a word first."
                )

            else:

                with st.spinner(
                    "Recognizing..."
                ):

                    predicted, cropped, processed = (
                        predict_word(
                            draw_image,
                            auto_crop=auto_crop_draw
                        )
                    )

                st.image(
                    cropped,
                    caption="Image used for recognition",
                    width=500
                )

                st.subheader(
                    "Prediction"
                )

                if predicted:

                    st.success(
                        predicted
                    )

                else:

                    st.warning(
                        "[empty prediction]"
                    )


# ============================================================
# 11. MODEL INFORMATION
# ============================================================

with st.expander(
    "Model information"
):

    st.write(
        "Architecture: CRNN + CTC"
    )

    st.write(
        "Input shape: (64, 256, 1)"
    )

    st.write(
        "Output shape: (32, 80)"
    )

    st.write(
        "Decoder: CTC greedy decoding"
    )

    st.write(
        "Vocabulary: 79 characters + 1 CTC blank"
    )

    st.write(
        "Training notebook baseline: "
        "CER 16.78%, WER 39.70%, "
        "Exact Word Accuracy 60.35%."
    )
