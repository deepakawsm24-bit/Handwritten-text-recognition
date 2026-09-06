import os
import json
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

# ============================================================
# CRNN HANDWRITTEN WORD RECOGNIZER
# Upload + Draw
# ============================================================

st.set_page_config(
    page_title="CRNN Handwritten Word Recognizer",
    page_icon="✍️",
    layout="wide",
)

IMG_WIDTH = 256
IMG_HEIGHT = 64
NUM_CLASSES = 80
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. FIND MODEL + CHARACTER MAPPING
# ============================================================

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


MODEL_PATH = first_existing([
    os.path.join(BASE_DIR, "CRNN_CTC_Basemodel_prediction.keras"),
    os.path.join(BASE_DIR, "baseline_model_deployment",
                 "CRNN_CTC_Basemodel_prediction.keras"),
])

MAPPING_PATHS = [
    os.path.join(BASE_DIR, "num_to_char.pkl"),
    os.path.join(BASE_DIR, "common", "num_to_char.pkl"),
    os.path.join(BASE_DIR, "baseline_model_deployment", "num_to_char.pkl"),
]

CHAR_MAPPING_JSON = os.path.join(BASE_DIR, "char_mapping.json")


@st.cache_resource
def load_mapping():
    # Prefer the original training artifact.
    for path in MAPPING_PATHS:
        if os.path.exists(path):
            with open(path, "rb") as f:
                mapping = pickle.load(f)
            return {int(k): str(v) for k, v in mapping.items()}

    # Fallback to repository JSON.
    if os.path.exists(CHAR_MAPPING_JSON):
        with open(CHAR_MAPPING_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "num_to_char" in data:
            data = data["num_to_char"]

        if isinstance(data, dict):
            return {int(k): str(v) for k, v in data.items()}

    raise FileNotFoundError(
        "Character mapping not found. Add num_to_char.pkl or char_mapping.json."
    )


@st.cache_resource
def load_model():
    if MODEL_PATH is None:
        raise FileNotFoundError(
            "CRNN_CTC_Basemodel_prediction.keras was not found."
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


try:
    num_to_char = load_mapping()
    prediction_model = load_model()
except Exception as e:
    st.error("Model setup failed.")
    st.code(str(e))
    st.stop()


# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

def crop_to_ink(pil_image, threshold=250, padding_ratio=0.02):
    """
    Remove outer white margins while keeping a small safety margin.
    Works for uploaded images and drawings.
    """
    pil_image = pil_image.convert("L")
    arr = np.asarray(pil_image)

    # Dark pixels = handwriting.
    mask = arr < threshold

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return pil_image

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad = max(2, int(round(min(arr.shape) * padding_ratio)))

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(arr.shape[1] - 1, x2 + pad)
    y2 = min(arr.shape[0] - 1, y2 + pad)

    return pil_image.crop((x1, y1, x2 + 1, y2 + 1))


def resize_and_pad(image):
    """
    Same deployment geometry as the CRNN training pipeline:
    preserve aspect ratio, resize to fit 256x64,
    then white-pad to exactly 256x64.
    """
    image = tf.image.resize(
        image,
        size=(IMG_HEIGHT, IMG_WIDTH),
        preserve_aspect_ratio=True,
    )

    current_height = tf.shape(image)[0]
    current_width = tf.shape(image)[1]

    pad_height = IMG_HEIGHT - current_height
    pad_width = IMG_WIDTH - current_width

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


def preprocess_pil_image(pil_image, auto_crop=True):
    """
    PIL RGB/L image -> model tensor (1, 64, 256, 1).
    """
    pil_image = pil_image.convert("L")

    if auto_crop:
        pil_image = crop_to_ink(pil_image)

    arr = np.asarray(pil_image, dtype=np.uint8)

    # One grayscale channel.
    image = tf.convert_to_tensor(
        arr[..., None],
        dtype=tf.uint8,
    )

    # [0,255] -> [0,1], white background remains 1.
    image = tf.image.convert_image_dtype(
        image,
        tf.float32,
    )

    image = resize_and_pad(image)

    return tf.expand_dims(image, axis=0), pil_image


# ============================================================
# 3. CTC DECODING
# ============================================================

def decode_prediction(predictions):
    """
    Greedy CTC decoding.
    Model output: (batch, time, classes).
    """
    input_length = np.full(
        predictions.shape[0],
        predictions.shape[1],
        dtype=np.int32,
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

            # -1 = unused position returned by CTC decoder.
            if token == -1:
                continue

            if token in num_to_char:
                text += num_to_char[token]

        words.append(text)

    return words


# ============================================================
# 4. PREDICTION
# ============================================================

def predict_word(pil_image, auto_crop=True):
    processed, cropped = preprocess_pil_image(
        pil_image,
        auto_crop=auto_crop,
    )

    predictions = prediction_model.predict(
        processed,
        verbose=0,
    )

    predicted_words = decode_prediction(predictions)

    predicted = predicted_words[0] if predicted_words else ""

    return predicted, cropped, processed


# ============================================================
# 5. DRAW CANVAS
# ============================================================

def canvas_to_pil(image_data):
    """
    Convert Streamlit canvas RGBA data to a proper white-background
    PIL image.

    This is important because transparent canvas pixels must not be
    interpreted as black/ink during preprocessing.
    """
    rgba = np.asarray(image_data, dtype=np.uint8)

    if rgba.ndim != 3 or rgba.shape[2] < 4:
        return Image.fromarray(rgba[:, :, :3]).convert("RGB")

    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0

    white = np.full_like(rgb, 255.0)

    composited = rgb * alpha + white * (1.0 - alpha)

    return Image.fromarray(
        np.clip(composited, 0, 255).astype(np.uint8)
    ).convert("RGB")


# ============================================================
# 6. UI
# ============================================================

st.title("✍️ CRNN Handwritten Word Recognizer")
st.write(
    "Recognize one handwritten English word using a CRNN + CTC model."
)

st.info(
    "For best results, write or upload one word only. "
    "Keep the letters separated and avoid lines, borders, or extra text."
)

tab_upload, tab_draw = st.tabs(["📤 Upload", "✍️ Draw"])


# ============================================================
# UPLOAD TAB
# ============================================================

with tab_upload:
    st.header("📤 Upload a Handwritten Word")

    uploaded_file = st.file_uploader(
        "Choose a handwritten word image",
        type=["png", "jpg", "jpeg"],
        key="upload_file",
    )

    auto_crop_upload = st.checkbox(
        "Automatically crop handwriting",
        value=True,
        key="auto_crop_upload",
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")

            st.image(
                uploaded_image,
                caption="Uploaded image",
                use_container_width=True,
            )

            if st.button(
                "🔍 Recognize Word",
                key="upload_predict",
                type="primary",
            ):
                with st.spinner("Recognizing handwriting..."):
                    predicted, cropped, processed = predict_word(
                        uploaded_image,
                        auto_crop=auto_crop_upload,
                    )

                if auto_crop_upload:
                    st.image(
                        cropped,
                        caption="Image used for recognition",
                        width=500,
                    )

                st.subheader("Prediction")

                if predicted:
                    st.success(predicted)
                else:
                    st.warning("No prediction was produced.")

        except Exception as e:
            st.error("Could not process the uploaded image.")
            st.exception(e)


# ============================================================
# DRAW TAB
# ============================================================

with tab_draw:
    st.header("✍️ Draw a Handwritten Word")
    st.write(
        "Write one English word on the white canvas, then click "
        "\"Recognize Drawing\"."
    )

    try:
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            background_color="#FFFFFF",
            stroke_color="#000000",
            stroke_width=7,
            drawing_mode="freedraw",
            width=700,
            height=220,
            display_toolbar=True,
            key="handwriting_canvas",
        )

        if st.button(
            "🔍 Recognize Drawing",
            key="draw_predict",
            type="primary",
        ):
            if canvas_result.image_data is None:
                st.warning("Please write a word on the canvas first.")
            else:
                canvas_pil = canvas_to_pil(
                    canvas_result.image_data
                )

                # Check whether actual dark ink exists.
                gray = np.asarray(
                    ImageOps.grayscale(canvas_pil)
                )

                if not np.any(gray < 250):
                    st.warning(
                        "Please write a word on the canvas first."
                    )
                else:
                    # Crop only for prediction; visible canvas is unchanged.
                    cropped_canvas = crop_to_ink(canvas_pil)

                    st.image(
                        cropped_canvas,
                        caption="Image used for recognition",
                        width=600,
                    )

                    with st.spinner("Recognizing handwriting..."):
                        predicted, cropped, processed = predict_word(
                            cropped_canvas,
                            auto_crop=False,
                        )

                    st.subheader("Prediction")

                    if predicted:
                        st.success(predicted)
                    else:
                        st.warning("No prediction was produced.")

    except ImportError:
        st.error(
            "Draw canvas dependency is missing. Install "
            "streamlit-drawable-canvas and restart Streamlit."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model information"):
    st.write("Architecture: CRNN + CTC")
    st.write("Input shape:", prediction_model.input_shape)
    st.write("Output shape:", prediction_model.output_shape)
    st.write("Expected image size: 64 × 256 × 1")
    st.write("Decoder: CTC greedy decoding")
    st.write("Character classes:", NUM_CLASSES)

    st.caption(
        "If the preprocessing is correct but the model still predicts "
        "a different word, the remaining limitation is the trained "
        "model's recognition accuracy/generalization."
    )
