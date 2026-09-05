# 1. IMPORT REQUIRED LIBRARIES

# Import os for handling file and folder paths
import os

# Import json for reading the character mapping
import json

# Import pickle for loading the saved character mapping
import pickle

# Import NumPy for numerical operations
import numpy as np

# Import TensorFlow for loading and running the CRNN model
import tensorflow as tf

# Import Streamlit for creating the web application
import streamlit as st

# Import the interactive drawing canvas
from streamlit_drawable_canvas import st_canvas


# ============================================================
#2. DEFINE PROJECT PATHS
# What this section does:
# Creates reliable paths to the model and supporting files. Using BASE_DIR means the app can find its files regardless of where Streamlit is launched from.
# ============================================================

# Get the folder containing app.py
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Define the path to the CRNN prediction model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "CRNN_CTC_Basemodel_prediction.keras"
)

# Define the path to the character-ID mapping
NUM_TO_CHAR_PATH = os.path.join(
    BASE_DIR,
    "..",
    "common",
    "num_to_char.pkl"
)

# Define the path to the image configuration file
IMAGE_CONFIG_PATH = os.path.join(
    BASE_DIR,
    "..",
    "common",
    "image_config.json"
)


# ============================================================
# 3. LOAD MODEL AND CHARACTER MAPPING
# 
#
# What this section does:
# Loads the trained CRNN prediction model and the character
# mapping required to convert predicted character IDs back
# into readable text.
#
# The model we are loading is the PREDICTION model, not the
# CTC training model.
# ============================================================


# Load the trained CRNN prediction model
# compile=False is used because we only need the model
# for prediction, not for further training.
prediction_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)
CHAR_MAPPING_JSON = os.path.join(BASE_DIR, "char_mapping.json")
# Load character mapping
if os.path.exists(NUM_TO_CHAR_PATH):
    with open(NUM_TO_CHAR_PATH, "rb") as f:
        num_to_char = pickle.load(f)

elif os.path.exists(CHAR_MAPPING_JSON):
    with open(CHAR_MAPPING_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "num_to_char" in data:
        data = data["num_to_char"]

    num_to_char = {int(k): str(v) for k, v in data.items()}

else:
    raise FileNotFoundError(
        "Character mapping not found."
    )


# Print confirmation so we know the model loaded successfully
print("Prediction model loaded successfully.")


# Print the model's expected input shape
# Expected: (None, 64, 256, 1)
print(
    "Model input shape:",
    prediction_model.input_shape
)


# Print the model's output shape
# Expected: (None, 32, 80)
print(
    "Model output shape:",
    prediction_model.output_shape
)


# Print confirmation that the character mapping loaded
print("Character mapping loaded successfully.")



# ============================================================
# 4. IMAGE PREPROCESSING

# What this section does:
# Converts an uploaded/drawn image into exactly the same format that was used when training the CRNN model.

# The model expects:
# Height = 64
# Width = 256
# Channel = 1 (grayscale)
# We preserve the original aspect ratio and use white padding instead of stretching the handwriting.

# ====================================

# Define the same distortion-free resizing function
# that was used during CRNN training.
def distortion_free_resize(
    image,
    img_size=(256, 64)
):

    # Extract the target width and height
    target_width, target_height = img_size

    # Resize while preserving the original aspect ratio
    image = tf.image.resize(
        image,
        size=(target_height, target_width),
        preserve_aspect_ratio=True
    )

    # Get the current height after resizing
    current_height = tf.shape(image)[0]

    # Get the current width after resizing
    current_width = tf.shape(image)[1]

    # Calculate required height padding
    pad_height = target_height - current_height

    # Calculate required width padding
    pad_width = target_width - current_width

    # Add white padding to the bottom and right
    image = tf.pad(
        image,
        [
            [0, pad_height],
            [0, pad_width],
            [0, 0]
        ],
        constant_values=1.0
    )

    # Return the resized and padded image
    return image


# ============================================================
# PREPROCESS IMAGE FOR CRNN


# This function converts an input image into the exact
# format expected by the CRNN prediction model.
def preprocess_image(image):

    # Convert RGB image to grayscale
    image = tf.image.rgb_to_grayscale(
        image
    )

    # Convert pixel values from [0, 255] to [0, 1]
    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    # Preserve aspect ratio and add white padding
    image = distortion_free_resize(
        image
    )

    # Add batch dimension
    # (64, 256, 1) → (1, 64, 256, 1)
    image = tf.expand_dims(
        image,
        axis=0
    )

    # Return model-ready image
    return image



# ============================================================
# 5. CTC GREEDY DECODING

#
# What this section does:
# Converts the CRNN model's numerical output into readable
# characters using CTC greedy decoding.
#
# The CRNN output contains:
#     32 time steps
#     80 character classes
#
# CTC decoding converts these predictions into character IDs.
# The num_to_char mapping then converts those IDs into actual
# characters and finally into a readable word.
# ============================================================


# Define a function to decode CRNN predictions
def decode_prediction(predictions):

    # Get the number of samples in the prediction batch
    batch_size = predictions.shape[0]

    # Get the number of time steps produced by the CRNN
    time_steps = predictions.shape[1]

    # Create the input length for every sample
    # Every sample has the same number of time steps
    input_length = np.full(
        batch_size,
        time_steps
    )

    # Perform CTC greedy decoding
    decoded_predictions, _ = tf.keras.backend.ctc_decode(
        predictions,
        input_length=input_length,
        greedy=True
    )

    # Take the first decoded result
    # and convert it from TensorFlow Tensor to NumPy array
    decoded_predictions = (
        decoded_predictions[0].numpy()
    )

    # Create a list to store the decoded words
    decoded_words = []

    # Process each sample in the batch
    for sequence in decoded_predictions:

        # Start with an empty predicted word
        predicted_text = ""

        # Process every character ID in the decoded sequence
        for token in sequence:

            # Convert the token to a Python integer
            token = int(token)

            # -1 represents an unused/blank position
            if token == -1:
                continue

            # Convert the character ID into the actual character
            if token in num_to_char:

                # Add the character to the predicted word
                predicted_text += num_to_char[token]

        # Add the completed word to the results list
        decoded_words.append(
            predicted_text
        )

    # Return the decoded words
    return decoded_words

# ============================================================
# 6. PREDICT A SINGLE IMAGE

#
# What this section does:
# Connects the complete prediction pipeline together.
#
# The function receives an image and performs:
#
#     Image
#       ↓
#     Preprocessing
#       ↓
#     CRNN prediction
#       ↓
#     CTC decoding
#       ↓
#     Predicted word
#
# Both the Upload tab and Draw tab will use this same
# function, ensuring that both use exactly the same
# prediction process.
# ============================================================


# Define the function that predicts one handwritten word
def predict_word(image):

    # Preprocess the input image
    # Resulting shape: (1, 64, 256, 1)
    processed_image = preprocess_image(
        image
    )

    # Pass the processed image through the CRNN model
    # Output shape: (1, 32, 80)
    predictions = prediction_model.predict(
        processed_image,
        verbose=0
    )

    # Decode the CRNN output using CTC
    decoded_words = decode_prediction(
        predictions
    )

    # Return the first predicted word
    return decoded_words[0]

# ============================================================
# 7. STREAMLIT PAGE CONFIGURATION

# What this section does:
# Configures the basic appearance and browser settings of
# the Streamlit application.
#
# This must be placed before other Streamlit commands such
# as st.title(), st.tabs(), st.file_uploader(), etc.
# ============================================================


# Configure the Streamlit page
st.set_page_config(
    # Set the title shown in the browser tab
    page_title="CRNN Handwritten Word Recognizer",

    # Use a wide layout for the application
    layout="wide",

    # Use an appropriate icon for the application
    page_icon="✍️"
)


# Display the main application title
st.title(
    "✍️ CRNN Handwritten Word Recognizer"
)


# Display a short description below the title
st.write(
    "Recognize a single handwritten word using a "
    "CRNN + CTC model."
)

# ============================================================
# 8. CREATE THE TWO STREAMLIT TABS

# What this section does:
# Creates the two main sections required for our application:
#
#     1. Upload → user uploads a handwritten word image
#     2. Draw   → user writes a word using the mouse
#
# We will add the actual functionality inside each tab
# in the next sections.
# ============================================================


# Create two tabs in the Streamlit application
upload_tab, draw_tab = st.tabs(
    [
        "📤 Upload",
        "✍️ Draw"
    ]
)


# ============================================================
# 9. UPLOAD TAB

# What this section does:
# Creates the Upload interface.
#
# The user can:
#     1. Upload an image containing one handwritten word
#     2. Choose whether to automatically crop to the ink
#     3. View the image
#     4. Get the model's predicted word
# ============================================================


# Place all Upload-tab components inside the Upload tab
with upload_tab:

    # Display the Upload tab heading
    st.header(
        "📤 Upload a Handwritten Word"
    )

    # Explain what the user should upload
    st.write(
        "Upload an image containing a single handwritten word."
    )

    # Allow the user to upload an image
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    # Allow the user to optionally crop the image to the
    # handwritten ink
    auto_crop = st.checkbox(
        "Automatically crop to ink",
        value=False
    )

    # Continue only if the user has uploaded an image
    if uploaded_file is not None:

        # Import PIL Image for opening the uploaded image
        from PIL import Image

        # Open the uploaded image
        uploaded_image = Image.open(
            uploaded_file
        ).convert("RGB")

        # Display the uploaded image
        st.image(
            uploaded_image,
            caption="Uploaded image",
            width="stretch"
        )

        # Convert the PIL image into a TensorFlow tensor
        image_tensor = tf.convert_to_tensor(
            np.array(uploaded_image),
            dtype=tf.uint8
        )

        # ----------------------------------------------------
        # Optional automatic crop to handwritten ink
        # ----------------------------------------------------

        if auto_crop:

            # Convert the image to grayscale
            grayscale_image = tf.image.rgb_to_grayscale(
                image_tensor
            )

            # Create a mask for pixels that are darker
            # than the white background
            ink_mask = grayscale_image < 250

            # Find the coordinates of the ink pixels
            coordinates = tf.where(
                ink_mask[:, :, 0]
            )

            # Check whether any ink pixels were found
            if tf.shape(coordinates)[0] > 0:

                # Find the top-left corner of the ink
                min_y = tf.reduce_min(
                    coordinates[:, 0]
                )

                min_x = tf.reduce_min(
                    coordinates[:, 1]
                )

                # Find the bottom-right corner of the ink
                max_y = tf.reduce_max(
                    coordinates[:, 0]
                )

                max_x = tf.reduce_max(
                    coordinates[:, 1]
                )

                # Crop the image to the detected ink region
                image_tensor = image_tensor[
                    min_y:max_y + 1,
                    min_x:max_x + 1,
                    :
                ]

                # Display the cropped image
                st.image(
                    image_tensor.numpy(),
                    caption="Auto-cropped image",
                    width="stretch"
                )

            else:

                # Inform the user if no ink was detected
                st.warning(
                    "No handwriting was detected for cropping."
                )

        # ----------------------------------------------------
        # Prediction button
        # ----------------------------------------------------

        # Create a button to start prediction
        if st.button(
            "🔍 Recognize Word",
            key="upload_predict"
        ):

            # Show a progress indicator while prediction runs
            with st.spinner(
                "Recognizing handwriting..."
            ):

                # Run the image through the CRNN prediction pipeline
                predicted_word = predict_word(
                    image_tensor
                )

            # Display the prediction heading
            st.subheader(
                "Prediction"
            )

            # Display the predicted word
            st.success(
                predicted_word
            )


# ============================================================
# 10. CROP HANDWRITING TO INK

# What this section does:
# Finds the handwritten ink inside the canvas and removes
# the unnecessary empty white area around it.
#
# Example:
#
#     Before:
#     ┌───────────────────────────────┐
#     │                               │
#     │       HELLO                   │
#     │                               │
#     └───────────────────────────────┘
#
#     After:
#     ┌───────────────┐
#     │    HELLO      │
#     └───────────────┘
#
# This makes the user's canvas image more similar to the
# tightly framed IAM word images.
#
# IMPORTANT:
# This function does NOT change the visible canvas.
# It only processes the image after the user clicks
# "Recognize Drawing".
# ============================================================


# Define a function to crop an image around the handwriting
def crop_to_ink(image):

    # Convert the RGB image to grayscale
    grayscale_image = tf.image.rgb_to_grayscale(
        image
    )

    # Create a mask for dark pixels
    # White background is approximately 255
    # Handwriting is darker than 250
    ink_mask = (
        grayscale_image[:, :, 0] < 250
    )

    # Find the coordinates of all detected ink pixels
    coordinates = tf.where(
        ink_mask
    )

    # Check whether any handwriting was detected
    if tf.shape(coordinates)[0] == 0:

        # Return the original image if no ink is found
        return image

    # Find the topmost ink pixel
    min_y = tf.reduce_min(
        coordinates[:, 0]
    )

    # Find the leftmost ink pixel
    min_x = tf.reduce_min(
        coordinates[:, 1]
    )

    # Find the bottommost ink pixel
    max_y = tf.reduce_max(
        coordinates[:, 0]
    )

    # Find the rightmost ink pixel
    max_x = tf.reduce_max(
        coordinates[:, 1]
    )

    # Crop the image around the detected handwriting
    cropped_image = image[
        min_y:max_y + 1,
        min_x:max_x + 1,
        :
    ]

    # Return the cropped handwriting image
    return cropped_image



# ============================================================
# 11. DRAW TAB
# ============================================================
#
# What this section does:
# Creates an interactive drawing canvas where the user can
# write a single handwritten word using the mouse.
#
# After the user clicks "Recognize Drawing":
#
#     Canvas image
#          ↓
#     Convert RGBA → RGB
#          ↓
#     Crop to handwritten ink
#          ↓
#     CRNN preprocessing
#          ↓
#     CRNN prediction
#          ↓
#     CTC decoding
#          ↓
#     Predicted word
#
# The visible canvas itself is NOT changed.
# Only the image used for prediction is cropped.
# ============================================================


# Put all Draw-tab components inside the Draw tab
with draw_tab:

    # Display the Draw tab heading
    st.header(
        "✍️ Draw a Handwritten Word"
    )

    # Explain what the user should do
    st.write(
        "Write one word on the canvas using your mouse."
    )


    # ========================================================
    # CREATE DRAWING CANVAS
    # ========================================================

    # Create the interactive drawing canvas
    canvas_result = st_canvas(

        # Set the canvas background to white
        background_color="#FFFFFF",

        # Set the handwriting color to black
        stroke_color="#000000",

        # Set the handwriting thickness
        stroke_width=4,

        # Enable freehand drawing
        drawing_mode="freedraw",

        # Set canvas width
        width=600,

        # Set canvas height
        height=200,

        # Display the canvas toolbar
        display_toolbar=True,

        # Give the canvas a unique Streamlit key
        key="handwriting_canvas"
    )


    # ========================================================
    # RECOGNIZE DRAWING
    # ========================================================

    # Create the prediction button
    if st.button(
        "🔍 Recognize Drawing",
        key="draw_predict"
    ):

        # Check whether the canvas returned image data
        if canvas_result.image_data is None:

            # Tell the user to draw something
            st.warning(
                "Please write a word on the canvas first."
            )

        else:

            # Get the complete canvas image
            # The canvas image is RGBA
            canvas_image = (
                canvas_result.image_data
                .astype(np.uint8)
            )


            # ------------------------------------------------
            # Check whether anything was drawn
            # ------------------------------------------------

            # The fourth channel is the alpha channel.
            # Pixels with alpha > 0 contain drawing data.
            has_drawing = np.any(
                canvas_image[:, :, 3] > 0
            )


            # Check whether the canvas is empty
            if not has_drawing:

                # Ask the user to write a word
                st.warning(
                    "Please write a word on the canvas first."
                )

            else:

                # ------------------------------------------------
                # Convert RGBA → RGB
                # ------------------------------------------------

                # Remove the alpha channel
                # RGBA becomes RGB
                canvas_rgb = (
                    canvas_image[:, :, :3]
                )


                # Convert the NumPy image into a TensorFlow tensor
                canvas_tensor = tf.convert_to_tensor(
                    canvas_rgb,
                    dtype=tf.uint8
                )


                # ------------------------------------------------
                # Crop the handwriting to the ink
                # ------------------------------------------------

                # Remove unnecessary white space around the
                # handwritten word
                cropped_canvas = crop_to_ink(
                    canvas_tensor
                )


                # ------------------------------------------------
                # Display the cropped image
                # ------------------------------------------------

                # Show the exact image that will be sent
                # to the CRNN model
                st.image(
                    cropped_canvas.numpy(),
                    caption="Image used for prediction",
                    width="stretch"
                )


                # ------------------------------------------------
                # Predict the handwritten word
                # ------------------------------------------------

                # Show a progress indicator
                with st.spinner(
                    "Recognizing handwriting..."
                ):

                    # Send the cropped image through the
                    # same preprocessing and prediction
                    # pipeline used by the Upload tab
                    predicted_word = predict_word(
                        cropped_canvas
                    )


                # ------------------------------------------------
                # Display prediction
                # ------------------------------------------------

                # Display the prediction heading
                st.subheader(
                    "Prediction"
                )

                # Display the predicted word
                st.success(
                    predicted_word
                )
