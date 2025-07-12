import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(working_dir), "model_training_notebook", "trained_fashion_mnist_model.h5")

# Custom function to fix InputLayer compatibility
def custom_input_layer(**kwargs):
    # Convert batch_shape to shape if it exists
    if 'batch_shape' in kwargs:
        batch_shape = kwargs.pop('batch_shape')
        if batch_shape is not None and len(batch_shape) > 1:
            kwargs['shape'] = batch_shape[1:]  # Remove batch dimension
    return tf.keras.layers.InputLayer(**kwargs)

# Custom objects dictionary for compatibility
custom_objects = {
    'InputLayer': custom_input_layer
}

# Load the pre-trained model with error handling
try:
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Trying alternative loading method...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e2:
        st.error(f"Alternative loading also failed: {e2}")
        st.info("Trying to load weights only...")
        try:
            # Create a simple model structure and load weights
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            # Try to load just the weights
            import h5py
            with h5py.File(model_path, 'r') as f:
                if 'model_weights' in f:
                    model.load_weights(model_path)
                    st.success("Model weights loaded successfully!")
                else:
                    model = None
                    st.error("Could not load model weights")
        except Exception as e3:
            st.error(f"All loading methods failed: {e3}")
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
                # st.write(str(result))
                predicted_class = np.argmax(result)
                prediction = class_names[predicted_class]

                st.success(f'Prediction: {prediction}')
            else:
                st.error("Model failed to load. Cannot make predictions.")
