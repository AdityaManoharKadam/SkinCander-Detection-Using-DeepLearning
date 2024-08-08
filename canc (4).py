import numpy as np
from PIL import Image
from tinker.data import ImageDataset
from tinker.model import Model
from tinker.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tinker.optimizers import Adam
from tinker.losses import BinaryCrossEntropy
from tinker.metrics import Accuracy

# Step 2: Load and preprocess the dataset

# Define image folder paths
folder_benign_train = "F:/satyam/train/benign"
folder_malignant_train = "F:/satyam/train/malignant"

# Load images from directories
train_data = ImageDataset.from_directory(folder_benign_train, folder_malignant_train, read_func=Image.open)

# Split the data into features and labels
X_train, y_train = train_data.images, train_data.labels

# Normalize pixel values to range [0, 1]
X_train = np.array(X_train) / 255.0

# Convert labels to one-hot encoding
y_train = np.eye(2)[y_train]

# Step 3: Define and compile the model

# Build the model
model = Model()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss=BinaryCrossEntropy(), metrics=[Accuracy()])

# Step 4: Train the model (optional)

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# Step 5: Use the trained model to predict skin cancer from uploaded images

def predict_skin_cancer(image_path):
    # Load and preprocess the image
    image = np.array(Image.open(image_path).convert('RGB').resize((224, 224))) / 255.0
    
    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    
    # Get the predicted class
    predicted_class = np.argmax(prediction)
    
    # Define classes
    classes = ['benign', 'malignant']
    
    return classes[predicted_class]

# Example usage
image_path = "F:/satyam/train/malignant/5.jpg"
prediction = predict_skin_cancer(image_path)
print("Predicted:", prediction)