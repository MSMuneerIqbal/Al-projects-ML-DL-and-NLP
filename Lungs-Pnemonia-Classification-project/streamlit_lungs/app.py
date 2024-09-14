import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model from the H5 file
model = tf.keras.models.load_model('./model.h5')

# Define the class labels
class_labels = ['Pneumonia Not Detected', 'Pneumonia Detected']

# Function to preprocess the input image
def preprocess_image(image):
    # Convert the image to RGB if it has only one channel
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize the image to the required input shape of the model
    image = image.resize((150, 150))
    # Convert the image to a NumPy array
    image = np.array(image)
    # Normalize the image
    image = image / 255.0
    # Add an extra dimension to represent the batch size (required by the model)
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image):
    # Preprocess the input image
    processed_image = preprocess_image(image)
    # Make predictions
    predictions = model.predict(processed_image)
    # Get the predicted class label
    predicted_class = class_labels[np.argmax(predictions)]
    # Get the confidence score for the predicted class
    confidence = np.max(predictions)
    return predicted_class, confidence

# Streamlit app
def main():
    st.markdown("""
        <style>
        .title {
            text-align: center;
            font-weight: bold;
            color: #FF0000; /* Red color for the title */
            font-size: 40px;
        }
        .description {
            text-align: center;
            font-size: 22px; /* Font size for description */
            color: #000000; /* Gray color for description */
        }
        .stButton button {
            background-color: #FF0000; /* Red background for the button */
            color: white; /* White text for the button */
            font-weight: bold; /* Bold text */
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 24px;
            cursor: pointer;
        }
        .stButton button:hover {
            color: black;
            font-weight: bold;
            font-size: 20px;
            background-color: #FFC0CB; /* Darker red when hovered */

    
        }
        .output-text {
            font-size: 30px;
            font-weight: bold;
            color: #FF0000; /* Red color for output text */
            text-align: center;
        }
        .confidence-text {
            font-weight: bold;
            font-size: 26px;
            color: #FF0000; /* Red color for confidence text */
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title">AI-Based Pneumonia Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="description">Upload an X-ray image and the model will predict whether pneumonia is detected or not.</p>', unsafe_allow_html=True)

    # Upload image file
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform prediction if the "Predict" button is clicked
        if st.button("Predict"):
            # Predict the class label and confidence score
            predicted_class, confidence = predict(image)
            # Display the predicted class label and confidence score
            st.markdown(f"<p class='output-text'>Result: {predicted_class}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='confidence-text'>Confidence: {confidence:.2f}</p>", unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()
