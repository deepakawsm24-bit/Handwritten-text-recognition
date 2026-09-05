# ============================================================
# CRNN HANDWRITTEN WORD RECOGNIZER
# ============================================================

# 1. IMPORT REQUIRED LIBRARIES
import os
import json
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from streamlit_drawable_konva import st_canvas


# ============================================================
# 2. DEFINE PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "CRNN_CTC_Basemodel_prediction.keras"
)

CHAR_MAPPING_JSON = os.path.join(
    BASE_DIR,
    "char_mapping.json"
)

NUM_TO_CHAR_PATH = os.path.join(
    BASE_DIR,
    "..",
    "common",
    "num_to_char.pkl"
)


# ============================================================
# 3. LOAD MODEL AND CHARACTER MAPPING
# ============================================================

prediction_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Load character mapping.
# Your repository has char_mapping.json, while the original
# team-member project used common/num_to_char.pkl.
if os.path.exists(NUM_TO_CHAR_PATH):
    with open(NUM_TO_CHAR_PATH, "rb") as f:
        num_to_char = pickle.load(f)

elif os.path.exists(CHAR_MAPPING_JSON):
    with open(CHAR_MAPPING_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "num_to_char" in data:
        data = data["num_to_char"]

    num_to_char = {
        int(k): str(v)
        for k, v in data.items()
    }

else:
    raise FileNotFoundError(
        "Character mapping not found. "
        "Please check char_mapping.json."
    )

print("Prediction model loaded successfully.")
print("Model input shape:", prediction_model.input_shape)
print("Model output shape:", prediction_model.output_shape)
print("Character mapping loaded successfully.")


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def distortion_free_resize(
    image,
    img_size=(256, 64)
):
    """
    Resize while preserving aspect ratio and add white
    padding to reach exactly 256 x 64.
    """

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
    """
    Convert an RGB image into the format expected by the CRNN:
    (1, 64, 256, 1)
    """

    # RGB -> grayscale
    image = tf.image.rgb_to_grayscale(image)

    # [0, 255] -> [0, 1]
    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    # Preserve aspect ratio + white padding
    image = distortion_free_resize(image)

    # Add batch dimension
    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# 5. CROP HANDWRITING TO INK
# ============================================================

def crop_to_ink(image):
    """
    Remove unnecessary white space around handwriting.

    Uses NumPy here because it is more reliable for the
    Streamlit drawing canvas than the previous TensorFlow
    coordinate/padding implementation.
    """

    image_np = image.numpy()

    # Convert RGB to grayscale
    grayscale = np.mean(
        image_np,
        axis=2
    )

    # Only clearly dark pixels are treated as handwriting.
    # This prevents almost-white background pixels from being
    # detected as ink.
    ink_mask = grayscale < 200

    coordinates = np.argwhere(ink_mask)

    # If nothing is detected, return original image.
    if coordinates.size == 0:
        return image

    min_y = coordinates[:, 0].min()
    min_x = coordinates[:, 1].min()
    max_y = coordinates[:, 0].max()
    max_x = coordinates[:, 1].max()

    # Small padding around handwriting
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
        image_np.shape[0] - 1,
        max_y + padding
    )

    max_x = min(
        image_np.shape[1] - 1,
        max_x + padding
    )

    cropped_image = image_np[
        min_y:max_y + 1,
        min_x:max_x + 1,
        :
    ]

    return tf.convert_to_tensor(
        cropped_image,
        dtype=tf.uint8
    )


# ============================================================
# 6. CTC GREEDY DECODING
# ============================================================

def decode_prediction(predictions):

    batch_size = predictions.shape[0]
    time_steps = predictions.shape[1]

    input_length = np.full(
        batch_size,
        time_steps
    )

    decoded_predictions, _ = tf.keras.backend.ctc_decode(
        predictions,
        input_length=input_length,
        greedy=True
    )

    decoded_predictions = (
        decoded_predictions[0].numpy()
    )

    decoded_words = []

    for sequence in decoded_predictions:

        predicted_text = ""

        for token in sequence:

            token = int(token)

            # -1 = unused / blank position
            if token == -1:
                continue

            if token in num_to_char:
                predicted_text += num_to_char[token]

        decoded_words.append(
            predicted_text
        )

    return decoded_words


# ============================================================
# 7. PREDICT A SINGLE IMAGE
# ============================================================

# ------------------------------------------------------------
# UPLOAD TAB
# ------------------------------------------------------------
tab_upload, tab_draw = st.tabs(["📤 Upload", "✍️ Draw"])
with tab_upload:
    st.header("Upload a Handwritten Word")
    st.write("Upload an image containing a single handwritten word.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        help="Use one handwritten word per image.",
    )

    auto_crop_upload = st.checkbox(
        "Automatically crop handwriting",
        value=True,
        key="auto_crop_upload",
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

            # Small preview
            st.image(
                uploaded_image,
                caption="Uploaded image",
                width=200,
            )

            if st.button(
                "🔎 Recognize Word",
                key="recognize_upload",
                type="primary",
            ):

                with st.spinner("Recognizing..."):

                    # SAME prediction function used by Draw
                    predicted, cropped, processed = predict_word(
                        uploaded_image,
                        auto_crop=auto_crop_upload,
                    )

                # Show exact image sent through prediction pipeline
                st.image(
                    cropped,
                    caption="Image used for recognition",
                    width=200,
                )

                st.subheader("Prediction")

                st.success(
                    predicted
                    if predicted
                    else "[empty prediction]"
                )

        except Exception as e:
            st.error("Could not process this image.")
            st.exception(e)

# ============================================================
# 10. UPLOAD TAB
# ============================================================

with tab_upload:

    st.header(
        "📤 Upload a Handwritten Word"
    )

    st.write(
        "Upload an image containing a single handwritten word."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    auto_crop = st.checkbox(
        "Automatically crop to ink",
        value=False
    )

    if uploaded_file is not None:

        uploaded_image = Image.open(
            uploaded_file
        ).convert("RGB")

        # Keep uploaded preview small
        st.image(
            uploaded_image,
            caption="Uploaded image",
            width=200
        )

        image_tensor = tf.convert_to_tensor(
            np.array(uploaded_image),
            dtype=tf.uint8
        )

        # Optional crop
        if auto_crop:

            image_tensor = crop_to_ink(
                image_tensor
            )

            # Keep cropped preview small
            st.image(
                image_tensor.numpy(),
                caption="Auto-cropped image",
                width=200
            )

        # Prediction
        if st.button(
            "🔍 Recognize Word",
            key="upload_predict"
        ):

            with st.spinner(
                "Recognizing handwriting..."
            ):

                predicted_word = predict_word(
                    image_tensor
                )

            st.subheader(
                "Prediction"
            )

            st.success(
                predicted_word
            )


# ============================================================
# 11. DRAW TAB
# ============================================================

with draw_tab:

    st.header(
        "✍️ Draw a Handwritten Word"
    )

    st.write(
        "Write one word on the canvas using your mouse."
    )

    # --------------------------------------------------------
    # DRAWING CANVAS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RECOGNIZE DRAWING
    # --------------------------------------------------------

    if st.button(
        "🔍 Recognize Drawing",
        key="draw_predict"
    ):

        if canvas_result.image_data is None:

            st.warning(
                "Please write a word on the canvas first."
            )

        else:

            # RGBA canvas image
            canvas_image = (
                canvas_result.image_data
                .astype(np.uint8)
            )

            # Check whether anything was drawn
            has_drawing = np.any(
                canvas_image[:, :, 3] > 0
            )

            if not has_drawing:

                st.warning(
                    "Please write a word on the canvas first."
                )

            else:

                # RGBA -> RGB
                canvas_rgb = (
                    canvas_image[:, :, :3]
                )

                # NumPy -> TensorFlow
                canvas_tensor = tf.convert_to_tensor(
                    canvas_rgb,
                    dtype=tf.uint8
                )

                # Crop only the handwriting
                cropped_canvas = crop_to_ink(
                    canvas_tensor
                )

                # ------------------------------------------------
                # IMPORTANT:
                # Display the exact image used for prediction,
                # but keep it SMALL.
                # ------------------------------------------------

                st.image(
                    cropped_canvas.numpy(),
                    caption="Image used for prediction",
                    width=200
                )

                # Predict
                with st.spinner(
                    "Recognizing handwriting..."
                ):

                    predicted_word = predict_word(
                        cropped_canvas
                    )

                st.subheader(
                    "Prediction"
                )

                st.success(
                    predicted_word
                )
