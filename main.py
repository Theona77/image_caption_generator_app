import streamlit as st 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import DenseNet201 

import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# --- Configuration ---
image_size = 224
max_length = 34

CAPTION_MODEL_NAME = 'model.keras'
FEATURE_EXTRACTOR_NAME = 'feature_extractor.keras' 

# caption generator function
def generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34, image_size=224):
    # Define custom objects as a safeguard, using a dummy class if needed
    # # Since the feature extractor uses DenseNet201, explicitly including its module is safer.
    custom_objects = {
        "Functional": Model,
        "DenseNet201": DenseNet201 
    }

    # load models safely
    try:
        # Load the caption model (Encoder-Decoder)
        caption_model = load_model(model_path, custom_objects=custom_objects, compile=False)
        # Load the feature extractor (DenseNet201)
        feature_extractor = load_model(feature_extractor_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        st.error(f"Error loading models. Check file paths and dependency versions. Error: {e}")
        return

    # load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # preprocess image
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # extract features
    # NOTE: It's often safer to use .predict(..., verbose=0)[0] to get the 1D feature vector
    image_features = feature_extractor.predict(img_array, verbose=0)
    
    # generate caption
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

    # Predict next word
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        
        if word is None or word == 'endseq': 
            break

        in_text += " " + word

    caption = in_text.replace('startseq', "").strip()

    # display image with caption
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=16, color='blue')
    st.pyplot(plt)


def main():
    st.title('Image Caption Generator')
    st.write("Upload an image and generate a caption")

    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
     
    if uploaded_image is not None:
        # save image temporarily
        temp_image_path = "uploaded_image.jpg"
        with open(temp_image_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())

        # Define paths using the NEW model names
        model_path = f'outputs/models/{CAPTION_MODEL_NAME}'
        feature_extractor_path = f'outputs/models/{FEATURE_EXTRACTOR_NAME}'
        # Assuming tokenizer.pkl is saved next to the models
        tokenizer_path = 'outputs/models/tokenizer.pkl' 

        # generate caption and display image
        generate_and_display_caption(
            temp_image_path, 
             model_path, 
            tokenizer_path, 
            feature_extractor_path
        )

# python main
if __name__ =="__main__":
    main()