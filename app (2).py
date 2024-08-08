import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import model_from_json
from keras.preprocessing import image

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Load model architecture
        with open('resnet50.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        # Load weights into the model
        model.load_weights("resnet50.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Make a prediction
def predict(img, model):
    processed_img = preprocess_image(img)
    preds = model.predict(processed_img)
    return preds

# Streamlit app
def main():
    st.title("Skin Cancer Detection")
    st.write("Upload an image to predict whether it's benign or malignant")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image.', use_column_width=True)
            st.write("Classifying...")

            model = load_model()
            if model is not None:
                preds = predict(img, model)
                pred_class = np.argmax(preds, axis=1)[0]
                label = "Malignant" if pred_class == 1 else "Benign"

                st.write(f"Prediction: {label}")
                st.write(f"Confidence: {preds[0][pred_class]*100:.2f}%")
            else:
                st.error("Model could not be loaded. Check the logs for more details.")
        except Exception as e:
            st.error(f"Error processing the image: {e}")

if __name__ == '__main__':
    main()
