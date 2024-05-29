import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load the model in the model folder
model = tf.keras.models.load_model(os.path.join("model", "brain_tumor_classifier.keras"))

# Create a session file to store settings in the dashboard/data folder
st.set_page_config(page_title='Brain Tumor Classifier', page_icon='🧠', initial_sidebar_state='auto')

if not os.path.exists('dashboard/data'):
    os.makedirs('dashboard/data')

# If the session file not exists, create it
if not os.path.exists('dashboard/data/session.txt'):
    # define presentation mode as false
    data = {'presentation_mode': False}
    # write the data to the session file
    with open('dashboard/data/session.txt', 'w') as f:
        f.write(str(data))

# load the session file
with open('dashboard/data/session.txt', 'r') as f:
    data = f.read()
    data = eval(data)

# Title of the webserver
st.title('Brain Tumor Classifier')

# Button to toggle the presentation mode
if st.button('Toggle Presentation Mode'):
    # If the presentation mode is true, set it to false
    if data['presentation_mode']:
        data['presentation_mode'] = False
    # If the presentation mode is false, set it to true
    else:
        data['presentation_mode'] = True

    # Write the data to the session file
    with open('dashboard/data/session.txt', 'w') as f:
        f.write(str(data))

# If the presentation mode is true, set the presentation mode to true
if data['presentation_mode']:
    

    # Analysis of Brain Tumor Image Dataset
    st.markdown('### Analysis of Brain Tumor Image Dataset')
    st.image('dashboard/data/original_data.png', caption='Original Data')

    st.markdown('### Relevant Data Features')
    st.markdown("""
    - **Shape and Size** 
    - **Texture** 
    - **Intensity**""")

    st.markdown('### Classes')
    st.markdown("""
    - **No Tumor**
    - **Tumor**""")

    # Model Architecture
    st.markdown('### Model Architecture')
    st.markdown("""
    - **Convolutional Neural Network**
    - **Binary Classification**
    - **Conv2D (Convolutional 2D) - 3 layers**
    - **MaxPooling2D - 3 layers**
    - **Flatten - 1 layer**
    - **Dense - 2 layers**
                """)

    # Results of the trained model
    st.markdown('### Results of the trained model')

    # Display the accuracy and loss
    st.markdown('#### Accuracy and loss')
    st.image('dashboard/data/accuracy_loss.png', caption='Accuracy and loss')

    # Display the precsion of the model
    st.markdown('#### Precision and recall')
    st.image('dashboard/data/precision_recall.png', caption='Precision and recall')

    # Display confusion matrix
    st.markdown('#### Confusion Matrix')
    st.image('dashboard/data/confusion_matrix.png', caption='Confusion Matrix')


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

