## A webserver with streamlit to display the results of the model with the posibility to upload a new image and get the prediction of the model

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os


# Load the model in the model folder
model = tf.keras.models.load_model(os.path.join('model', 'brain_tumor_classifier.keras'))

# Title of the webserver
st.title('Brain Tumor Classifier')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# If an image is uploaded
if uploaded_file is not None:
    # Load the image and make it compatible with the model input (32, 256, 256, 3)
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image = np.array(image.resize((256, 256)))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Get the prediction of the model
    prediction = model.predict(image)

    # Display the prediction
    st.markdown('### Prediction:')

    # Get the class of the prediction (0: no tumor, 1: tumor)
    # Display the prediction probability
    if prediction[0][0] > 0.5:
        st.markdown('The model predicts that there **is a tumor** in the image.')
        # Display the probability of the prediction
        probability = round((prediction[0][0] - 0.5), 4)

        print((prediction[0][0]- 0.5) * 100)

    else:
        st.markdown('The model predicts that there **is no tumor** in the image.')
        probability = round(((0.5 - prediction[0][0]) / 0.5 ), 4)

        print((0.5 - prediction[0][0])/ 0.5 * 100)

    print(prediction[0][0])

    if probability < 0.5:
        st.markdown('***The model is **not very confident** in its prediction.***')


    # Display the probability of the prediction
    probability_text = f'Probability of the prediction is {round(probability * 100, 2)}%'

    # Display the prediction in a progress bar
    st.progress(probability, probability_text)

# Displays additional information about the model in a collapsible section
with st.expander('Click here to see more information about the model'):
    # Display the model architecture
    st.write('Model Architecture:')
    st.write(model)

    # Display the model layers
    st.write('Model Layers:')
    st.write(model.layers)

