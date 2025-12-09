import streamlit as st 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model # <-- FIX: Import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

# --- Configuration ---
image_size = 224  
max_length = 34

# Use the names of the *re-saved* models
RESAVED_CAPTION_MODEL_NAME = 'model_resaved.keras'
RESAVED_FEATURE_EXTRACTOR_NAME = 'feature_extractor_resaved.keras'

# Base directory setup
BASE_DIR = os.path.dirname(__file__)
# No need for this global path if you define local paths in main()
# model_path = os.path.join(BASE_DIR, 'outputs', 'models', 'model_resaved.keras')


# caption generator function
def generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34, image_size=224):
    # load models safely
    # This dictionary maps the model's internal 'Functional' name to the correct Keras class (Model)
    custom_objects = {"Functional": Model}
    
    # --- FIX: Pass custom_objects to BOTH model loads ---
    caption_model = load_model(model_path, custom_objects=custom_objects)
    feature_extractor = load_model(feature_extractor_path, custom_objects=custom_objects)
    
    # load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    
    # preprocess image
    img = load_img(image_path, target_size=(image_size, image_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # extract features
    image_features = feature_extractor.predict(img, verbose=0)
    
    # generate caption
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None: 
            break
        in_text += " " + word
        if word == 'endseq': 
            break
            
    caption = in_text.replace('startseq', "").replace('endseq', "").strip()
    
    # display image with caption
    img = load_img(image_path, target_size=(image_size, image_size))
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=16, color='blue')
    st.pyplot(plt)


def main():
    st.title('Image Caption Generator')
    st.write("Upload an image and generate a caption")
    
    # upload image
    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image is not None:
        # save image temporarily
        with open("uploaded_image.jpg", 'wb') as f:
            f.write(uploaded_image.getbuffer())
            
        # Define paths using the RESAVED model names
        # Streamlit execution runs from the root of the app directory
        model_path = f'outputs/models/{RESAVED_CAPTION_MODEL_NAME}'
        feature_extractor_path = f'outputs/models/{RESAVED_FEATURE_EXTRACTOR_NAME}'
        tokenizer_path = 'outputs/models/tokenizer.pkl'

        # generate caption and display image
        generate_and_display_caption(
            'uploaded_image.jpg', 
            model_path, 
            tokenizer_path, 
            feature_extractor_path
        )
    
# python main
if __name__ =="__main__":
    main()