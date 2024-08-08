import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Function to define the model architecture
def build_model(input_shape=(224, 224, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(512, activation='relu'))  # Ensure this matches the original architecture
    model.add(Dense(2, activation='softmax'))  # Ensure this matches the original architecture
    return model

# Build the model
model = build_model()

# Load the weights
model.load_weights("F:/satyam/model.weights.h5")  # Update with the path to your weights file

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Assuming input shape of your model is (224, 224)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array

# Streamlit application
st.title("Image Predictor")

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_array = preprocess_image(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image_array)

    # Display the prediction
    st.write("Prediction:", prediction)
