import os
import json
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps


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
# 2. FIND DEPLOYMENT FILES
# ============================================================
def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None

MODEL_PATH = first_existing([
    os.path.join(BASE_DIR, "CRNN_CTC_Basemodel_prediction.keras"),
    os.path.join(BASE_DIR, "baseline_model_deployment", "CRNN_CTC_Basemodel_prediction.keras"),
])

NUM_TO_CHAR_PATHS = [
    os.path.join(BASE_DIR, "num_to_char.pkl"),
    os.path.join(BASE_DIR, "common", "num_to_char.pkl"),
    os.path.join(BASE_DIR, "baseline_model_deployment", "num_to_char.pkl"),
]

CHAR_MAPPING_JSON = os.path.join(BASE_DIR, "char_mapping.json")

# ============================================================
# 3. LOAD CHARACTER MAPPING
# ============================================================
@st.cache_resource
def load_num_to_char():
    # Preferred artifact produced by the training notebook.
    for path in NUM_TO_CHAR_PATHS:
        if os.path.exists(path):
            with open(path, "rb") as f:
                mapping = pickle.load(f)
            return {int(k): str(v) for k, v in mapping.items()}

    # Fallback for the JSON file already present in the repository.
    if os.path.exists(CHAR_MAPPING_JSON):
        with open(CHAR_MAPPING_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support either { "0": " ", ... } or
        # { "num_to_char": {"0": " ", ...} }.
        if isinstance(data, dict) and "num_to_char" in data:
            data = data["num_to_char"]

        if isinstance(data, dict):
            return {int(k): str(v) for k, v in data.items()}

    raise FileNotFoundError(
        "Character mapping not found. Add num_to_char.pkl "
        "from the P21/common folder, or add char_mapping.json."
    )

# ============================================================
# 4. LOAD PREDICTION MODEL
# ============================================================
@st.cache_resource
def load_prediction_model():
    if MODEL_PATH is None:
        raise FileNotFoundError(
            "CRNN_CTC_Basemodel_prediction.keras was not found."
        )

    # This is the prediction-only model saved in notebook Cell 145,
    # so the custom CTC loss layer is not required here.
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
#    Matches the training notebook:
#    grayscale -> float32 [0,1] -> aspect-ratio resize ->
#    white padding -> 64 x 256 x 1
# ============================================================
def distortion_free_resize(image, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    target_width, target_height = target_size

    image = tf.image.resize(
        image,
        size=(target_height, target_width),
        preserve_aspect_ratio=True,
    )

    current_height = tf.shape(image)[0]
    current_width = tf.shape(image)[1]

    pad_height = target_height - current_height
    pad_width = target_width - current_width

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
    """Crop outer white margins around handwriting.

    This is an optional deployment-only convenience for uploaded/drawn
    images. The IAM training images are already word-level crops.
    """
    gray = ImageOps.grayscale(pil_image)
    arr = np.asarray(gray)

    # Ink is darker than the white background.
    mask = arr < 245

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return pil_image

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Small safety margin around the ink.
    pad = max(2, int(round(min(arr.shape) * 0.02)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(arr.shape[1] - 1, x2 + pad)
    y2 = min(arr.shape[0] - 1, y2 + pad)

    return pil_image.crop((x1, y1, x2 + 1, y2 + 1))


def preprocess_pil_image(pil_image, auto_crop=True):
    pil_image = pil_image.convert("L")

    if auto_crop:
        pil_image = crop_to_ink(pil_image)

    # Convert PIL image to TensorFlow image with one channel.
    arr = np.asarray(pil_image, dtype=np.uint8)
    image = tf.convert_to_tensor(arr[..., None], dtype=tf.uint8)

    # Exactly the same normalization used during training.
    image = tf.image.convert_image_dtype(
        image,
        tf.float32,
    )

    image = distortion_free_resize(image)

    # Final shape: (1, 64, 256, 1)
    return tf.expand_dims(image, axis=0), pil_image

# ============================================================
# 6. CTC DECODING
#    Same greedy decoder used in the notebook.
# ============================================================
def decode_prediction(predictions):
    input_length = np.full(
        predictions.shape[0],
        predictions.shape[1],
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

    return predicted_words[0], cropped, processed

# ============================================================
# 7. UI
# ============================================================
st.title("✍️ CRNN Handwritten Word Recognizer")
st.write(
    "Recognize a single handwritten English word using a CRNN + CTC model."
)

st.info(
    "Important: this baseline model was trained on IAM word images. "
    "For best results, upload one tightly cropped handwritten word, "
    "not a screenshot containing labels, borders, lines, or multiple words."
)

tab_upload, tab_draw = st.tabs(["📤 Upload", "✍️ Draw"])

# ------------------------------------------------------------
# UPLOAD TAB
# ------------------------------------------------------------
with tab_upload:
    st.header("Upload a Handwritten Word")
    st.write("Upload an image containing a single handwritten word.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        help="Use one handwritten word per image.",
    )

    auto_crop_upload = st.checkbox(
        "Automatically crop to handwriting",
        value=True,
        key="auto_crop_upload",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True,
            )

            if st.button(
                "🔎 Recognize Word",
                key="recognize_upload",
                type="primary",
            ):
                with st.spinner("Recognizing..."):
                    predicted, cropped, processed = predict_word(
                        image,
                        auto_crop=auto_crop_upload,
                    )

                if auto_crop_upload:
                    st.image(
                        cropped,
                        caption="Image used for recognition after cropping",
                        width=500,
                    )

                st.subheader("Prediction")
                st.success(predicted if predicted else "[empty prediction]")

        except Exception as e:
            st.error("Could not process this image.")
            st.exception(e)

# ------------------------------------------------------------
# DRAW TAB
# ------------------------------------------------------------
with tab_draw:
    st.header("✍️ Draw a Handwritten Word")
    st.write(
        "Draw one English word in a drawing app (Paint/Snipping Tool), "
        "save it as PNG/JPG, then upload it here."
    )

    st.info(
        "The old embedded drawing canvas was removed because its dependency "
        "is no longer maintained. Your existing CRNN + CTC prediction pipeline "
        "is unchanged."
    )

    draw_file = st.file_uploader(
        "Choose your drawing image",
        type=["png", "jpg", "jpeg"],
        key="draw_upload",
        help="Save your handwritten word as a PNG/JPG and upload it here.",
    )

    auto_crop_draw = st.checkbox(
        "Automatically crop handwriting",
        value=True,
        key="auto_crop_draw",
    )

    if draw_file is not None:
        try:
            draw_image = Image.open(draw_file).convert("RGB")

            st.image(
                draw_image,
                caption="Drawing",
                width=700,
            )

            if st.button(
                "🔎 Recognize Word",
                key="recognize_draw",
                type="primary",
            ):
                with st.spinner("Recognizing..."):
                    predicted, cropped, processed = predict_word(
                        draw_image,
                        auto_crop=auto_crop_draw,
                    )

                st.image(
                    cropped,
                    caption="Image used for recognition",
                    width=500,
                )

                st.subheader("Prediction")
                st.success(
                    predicted if predicted else "[empty prediction]"
                )

        except Exception as e:
            st.error("Could not process this drawing.")
            st.exception(e)

# ============================================================
# 8. MODEL INFORMATION
# ============================================================
with st.expander("Model information"):
    st.write("Architecture: CRNN + CTC")
    st.write("Input shape: (64, 256, 1)")
    st.write("Output shape: (32, 80)")
    st.write("Decoder: CTC greedy decoding")
    st.write("Vocabulary: 79 characters + 1 CTC blank")
    st.write(
        "Training notebook baseline: "
        "CER 16.78%, WER 39.70%, Exact Word Accuracy 60.35%."
    )
