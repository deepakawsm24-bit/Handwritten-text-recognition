import os
import json
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

# ============================================================
# CONFIG
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

# The repository shown by you keeps the prediction model and
# char_mapping.json in the project root.
MODEL_PATH = os.path.join(
    BASE_DIR, "CRNN_CTC_Basemodel_prediction.keras"
)

CHAR_MAPPING_JSON = os.path.join(
    BASE_DIR, "char_mapping.json"
)

# ============================================================
# LOAD CHARACTER MAPPING
# ============================================================
@st.cache_resource
def load_num_to_char():
    # Prefer the exact mapping format saved in char_mapping.json.
    with open(CHAR_MAPPING_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "num_to_char" in data:
        data = data["num_to_char"]

    return {int(k): str(v) for k, v in data.items()}

# ============================================================
# LOAD PREDICTION MODEL
# ============================================================
@st.cache_resource
def load_prediction_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

try:
    num_to_char = load_num_to_char()
    prediction_model = load_prediction_model()
except Exception as e:
    st.error("Model setup failed.")
    st.exception(e)
    st.stop()

# ============================================================
# SAME PREPROCESSING AS TRAINING NOTEBOOK
# ============================================================
def distortion_free_resize(image, img_size=(256, 64)):
    target_width, target_height = img_size

    image = tf.image.resize(
        image,
        size=(target_height, target_width),
        preserve_aspect_ratio=True
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
            [0, 0]
        ],
        constant_values=1.0
    )

    return image

def preprocess_image(image):
    # Same order as the training notebook:
    # RGB -> grayscale -> [0,1] -> aspect-ratio resize -> white pad
    image = tf.image.rgb_to_grayscale(image)

    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    image = distortion_free_resize(image)

    return tf.expand_dims(image, axis=0)

# ============================================================
# CROP ONLY FOR DRAW/OPTIONAL UPLOAD
# ============================================================
def crop_to_ink(image):
    grayscale_image = tf.image.rgb_to_grayscale(image)

    ink_mask = grayscale_image[:, :, 0] < 250
    coordinates = tf.where(ink_mask)

    if tf.shape(coordinates)[0] == 0:
        return image

    min_y = tf.reduce_min(coordinates[:, 0])
    min_x = tf.reduce_min(coordinates[:, 1])
    max_y = tf.reduce_max(coordinates[:, 0])
    max_x = tf.reduce_max(coordinates[:, 1])

    return image[
        min_y:max_y + 1,
        min_x:max_x + 1,
        :
    ]

# ============================================================
# CTC DECODING — SAME AS TRAINING NOTEBOOK
# ============================================================
def decode_prediction(predictions):
    input_length = np.full(
        predictions.shape[0],
        predictions.shape[1]
    )

    decoded_predictions, _ = tf.keras.backend.ctc_decode(
        predictions,
        input_length=input_length,
        greedy=True
    )

    decoded_predictions = decoded_predictions[0].numpy()

    decoded_words = []

    for sequence in decoded_predictions:
        predicted_text = ""

        for token in sequence:
            token = int(token)

            if token == -1:
                continue

            if token in num_to_char:
                predicted_text += num_to_char[token]

        decoded_words.append(predicted_text)

    return decoded_words

def predict_word(image):
    processed_image = preprocess_image(image)

    predictions = prediction_model.predict(
        processed_image,
        verbose=0
    )

    return decode_prediction(predictions)[0]

# ============================================================
# UI
# ============================================================
st.title("✍️ CRNN Handwritten Word Recognizer")
st.write(
    "Recognize a single handwritten English word using a CRNN + CTC model."
)

tab_upload, tab_draw = st.tabs(["📤 Upload", "✍️ Draw"])

# ============================================================
# UPLOAD
# ============================================================
with tab_upload:
    st.header("📤 Upload a Handwritten Word")
    st.write("Upload an image containing a single handwritten word.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    # IMPORTANT: staff's original upload behavior was NO crop by default.
    auto_crop = st.checkbox(
        "Automatically crop to ink",
        value=False
    )

    if uploaded_file is not None:
        uploaded_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            uploaded_image,
            caption="Uploaded image",
            use_container_width=True
        )

        image_tensor = tf.convert_to_tensor(
            np.array(uploaded_image),
            dtype=tf.uint8
        )

        if auto_crop:
            image_tensor = crop_to_ink(image_tensor)

            st.image(
                image_tensor.numpy(),
                caption="Auto-cropped image",
                use_container_width=True
            )

        if st.button(
            "🔍 Recognize Word",
            key="upload_predict",
            type="primary"
        ):
            with st.spinner("Recognizing handwriting..."):
                predicted_word = predict_word(image_tensor)

            st.subheader("Prediction")
            st.success(
                predicted_word if predicted_word else "[empty prediction]"
            )

# ============================================================
# DRAW — MATCH STAFF CANVAS SETTINGS
# ============================================================
with tab_draw:
    st.header("✍️ Draw a Handwritten Word")
    st.write("Write one word on the canvas using your mouse.")

    if st_canvas is None:
        st.error(
            "The drawing component is not installed. "
            "Install streamlit-drawable-canvas-fix==0.9.8 "
            "in requirements.txt and redeploy the app."
        )
    else:
        # These settings match the staff member's original app:
        # width=600, height=200, stroke_width=4.
        canvas_result = st_canvas(
            background_color="#FFFFFF",
            stroke_color="#000000",
            stroke_width=4,
            drawing_mode="freedraw",
            width=600,
            height=200,
            display_toolbar=True,
            key="handwriting_canvas"
        )

        if st.button(
            "🔍 Recognize Drawing",
            key="draw_predict",
            type="primary"
        ):
            if canvas_result.image_data is None:
                st.warning("Please write a word on the canvas first.")
            else:
                canvas_image = (
                    canvas_result.image_data.astype(np.uint8)
                )

                has_drawing = np.any(
                    canvas_image[:, :, 3] > 0
                )

                if not has_drawing:
                    st.warning("Please write a word on the canvas first.")
                else:
                    # Keep the staff member's RGBA -> RGB behavior.
                    canvas_rgb = canvas_image[:, :, :3]

                    canvas_tensor = tf.convert_to_tensor(
                        canvas_rgb,
                        dtype=tf.uint8
                    )

                    # Keep the staff member's draw behavior:
                    # crop the handwriting before prediction.
                    cropped_canvas = crop_to_ink(
                        canvas_tensor
                    )

                    st.image(
                        cropped_canvas.numpy(),
                        caption="Image used for prediction",
                        width=500
                    )

                    with st.spinner("Recognizing handwriting..."):
                        predicted_word = predict_word(
                            cropped_canvas
                        )

                    st.subheader("Prediction")
                    st.success(
                        predicted_word if predicted_word else "[empty prediction]"
                    )

# ============================================================
# MODEL INFORMATION
# ============================================================
with st.expander("Model information"):
    st.write("Architecture: CRNN + CTC")
    st.write("Input shape: (64, 256, 1)")
    st.write("Output shape: (32, 80)")
    st.write("Decoder: CTC greedy decoding")
    st.write("Vocabulary: 79 characters + 1 CTC blank")
