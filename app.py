# pip install streamlit tensorflow opencv-python-headless streamlit-drawable-canvas

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained model (assuming the model is saved as 'mnist_model.h5')
model = tf.keras.models.load_model('mnist_model.h5')

# Streamlit app title
st.title('Handwritten Digit Recognition')

# Streamlit app description
st.write("Draw a digit (0-9) on the canvas below, and the model will predict it.")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="black",  # Background color of the canvas
    stroke_width=10,  # Width of the drawing stroke
    stroke_color="white",  # Color of the drawing stroke
    background_color="black",  # Background color of the canvas
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict button
if st.button('Predict'):
    if canvas_result.image_data is not None:
        # Preprocess the drawn image
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = img / 255.0  # Normalize pixel values
        img = img.reshape(1, 28, 28, 1)  # Reshape for the model

        # Model prediction
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)

        # Display the predicted digit
        st.write(f'Predicted Digit: {predicted_digit}')
        st.bar_chart(prediction[0])
    else:
        st.write("Please draw a digit on the canvas first.")
