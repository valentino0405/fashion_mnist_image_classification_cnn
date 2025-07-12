import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(working_dir), "model_training_notebook", "trained_fashion_mnist_model.h5")

# Create the exact model architecture from the notebook
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    return model

# Load the model using a safer approach
try:
    # First try direct loading with compile=False
    model = tf.keras.models.load_model(model_path, compile=False)
except Exception as e:
    try:
        # Create the model architecture and load weights
        model = create_model()
        model.load_weights(model_path)
    except Exception as e2:
        st.error("❌ Failed to load the model. Please check the model file.")
        model = None

# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

# Streamlit App
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            if model is not None:
                # Preprocess the uploaded image
                img_array = preprocess_image(uploaded_image)

                # Make a prediction using the pre-trained model
                result = model.predict(img_array)
                
                # Apply softmax if the model outputs raw logits
                if result.max() > 1.0 or result.min() < 0.0:
                    result = tf.nn.softmax(result).numpy()
                
                predicted_class = np.argmax(result)
                prediction = class_names[predicted_class]
                confidence = result[0][predicted_class] * 100

                st.success(f'🎯 Prediction: **{prediction}**')
                st.info(f'📊 Confidence: {confidence:.1f}%')
            else:
                st.error("Model failed to load. Cannot make predictions.")
