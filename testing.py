import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the pre-trained InceptionV3 model (without the top layer)
base_model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the base model (no need for further training)
base_model.trainable = False

# Define the model architecture
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # Assuming 2 classes: Healthy and Diseased
])


# Preprocessing function for new input images
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image
    return img_array

# Prediction function
def predict_image(img_path, model):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])  # Get the predicted class
    class_labels = ['Healthy', 'Diseased']  # Modify according to your class names
    return class_labels[class_idx], predictions[0][class_idx]

# Example of taking an image path from the user
# img_path = input("/content/drive/MyDrive/Plant_diseases/Healthy_Grapes/Grape_Healthy10.JPG ")  # Take image input from the user
# Remove the input function and directly assign the image path
img_path = "/content/drive/MyDrive/testing/Corn_(maize)___Common_rust_5.jpeg"# Directly assign the image path


# Predict the class of the input image
predicted_class, confidence = predict_image(img_path, model)
print(f"Predicted Class: {predicted_class}, Confidence: {confidence * 100:.2f}%")

# Optional: Display the input image
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
plt.title(f"Prediction: {predicted_class}")
plt.axis("off")
plt.show()