import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load the model in the model folder
model = tf.keras.models.load_model(os.path.join("model", "brain_tumor_classifier.keras"))

# Title of the webserver
st.title('Brain Tumor Classifier')

# Analysis of Brain Tumor Image Dataset
st.markdown('### Analysis of Brain Tumor Image Dataset')

st.markdown("""- **Number of images in the "Tumor" folder:** 1683
- **Number of images in the "Non-Tumor" folder:** 2079

#### Relevant Data Features
For the project, the key features to focus on are the visual characteristics of the MRI images that distinguish tumor from non-tumor cases. These features include:

- **Shape and Size:** Tumors usually have irregular shapes and varying sizes compared to normal brain tissue.
- **Texture:** Tumor regions often have different texture patterns compared to non-tumor regions.
- **Intensity:** The intensity values (brightness and contrast) in tumor regions can differ significantly from those in healthy brain tissue.""")

# Results of the trained model
st.markdown('### Results of the trained model')

# Display the accuracy and loss
st.markdown('### Accuracy and loss')
st.image('dashboard/accuracy_loss.png', caption='Accuracy and loss')

# Display the precsion of the model
st.markdown('### Precision')
st.image('dashboard/precision_valprecision.png', caption='Precision')

# Display confusion matrix
st.markdown('### Confusion Matrix')
st.image('dashboard/confusion_matrix.png', caption='Confusion Matrix')


# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# If an image is uploaded
if uploaded_file is not None:
    # Load the image and make it compatible with the model input (1, 256, 256, 1) (gray scale image)
    image = Image.open(uploaded_file).resize((256, 256)).convert('L')
    image = np.expand_dims(np.array(image), axis=0)

    # Display the image
    st.image(image[0], caption='Uploaded Image.', use_column_width=True)

    # Get the prediction of the model
    prediction = model.predict(image)

    # Display the prediction
    st.markdown('### Prediction:')

    # Get the class of the prediction (0: no tumor, 1: tumor)
    # Display the prediction probability
    if prediction[0][0] > 0.5:
        st.markdown('The model predicts that there **is a tumor** in the image.')
        # Display the probability of the prediction
        probability = round(((prediction[0][0] - 0.5) / 0.5), 4)

        print(f"{round(((prediction[0][0] - 0.5) / 0.5), 4)}")

    else:
        st.markdown('The model predicts that there **is no tumor** in the image.')
        probability = round(((0.5 - prediction[0][0]) / 0.5 ), 4)

        print(f"{round(0.5 - prediction[0][0])/ 0.5 * 100} %")

    print(f"Prediction value: {prediction[0][0]}")

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
