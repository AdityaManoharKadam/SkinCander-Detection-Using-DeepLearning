import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

# Load the trained model
def load_model():
    try:
        # Load model architecture
        with open("F:/satyam/resnet50.json", 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        # Load weights into the model
        model.load_weights("resnet50.h5")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
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

# Function to handle image upload and prediction
def upload_image():
    global panelA, panelB
    
    # Open a file dialog to select an image
    path = filedialog.askopenfilename()
    
    if len(path) > 0:
        # Load the image from disk
        img = Image.open(path)
        
        # Convert the image to a format compatible with Tkinter
        img_tk = ImageTk.PhotoImage(img)
        
        # If the panels are not None, we need to update the image
        if panelA is None or panelB is None:
            panelA = Label(image=img_tk)
            panelA.image = img_tk
            panelA.pack(side="left", padx=10, pady=10)
        else:
            panelA.configure(image=img_tk)
            panelA.image = img_tk
        
        # Make prediction
        model = load_model()
        if model is not None:
            preds = predict(img, model)
            pred_class = np.argmax(preds, axis=1)[0]
            label = "Malignant" if pred_class == 1 else "Benign"
            confidence = f"Confidence: {preds[0][pred_class]*100:.2f}%"
            
            result = f"Prediction: {label}\n{confidence}"
            if panelB is None:
                panelB = Label(text=result)
                panelB.pack(side="right", padx=10, pady=10)
            else:
                panelB.config(text=result)

# Initialize the window toolkit along with the two image panels
root = tk.Tk()
panelA = None
panelB = None

# Set up the GUI title and geometry
root.title("Skin Cancer Detection")
root.geometry("800x600")

# Create a button for uploading an image
btn = Button(root, text="Upload an image", command=upload_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# Kick off the GUI
root.mainloop()
