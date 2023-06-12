import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('BC.h5',compile=False)
lab = {0: 'Afghan', 1: 'African Wild Dog', 2: 'Airedale', 3: 'American Hairless', 4: 'American Spaniel', 5: 'Basenji',
       6: 'Basset', 7: 'Beagle', 8: 'Bearded Collie', 9: 'Bermaise', 10: 'Bichon Frise', 11: 'Blenheim',
       12: 'Bloodhound', 13: 'Bluetick', 14: 'Border Collie', 15: 'Borzoi', 16: 'Boston Terrier', 17: 'Boxer',
       18: 'Bull Mastiff', 19: 'Bull Terrier', 20: 'Bulldog', 21: 'Cairn', 22: 'Chihuahua', 23: 'Chinese Crested',
       24: 'Chow', 25: 'Clumber', 26: 'Cockapoo', 27: 'Cocker', 28: 'Collie', 29: 'Corgi', 30: 'Coyote',
       31: 'Dalmation', 32: 'Dhole', 33: 'Dingo', 34: 'Doberman', 35: 'Elk Hound', 36: 'French Bulldog',
       37: 'German Sheperd', 38: 'Golden Retriever', 39: 'Great Dane', 40: 'Great Perenees', 41: 'Greyhound',
       42: 'Groenendael', 43: 'Irish Spaniel', 44: 'Irish Wolfhound', 45: 'Japanese Spaniel', 46: 'Komondor',
       47: 'Labradoodle', 48: 'Labrador', 49: 'Lhasa', 50: 'Malinois', 51: 'Maltese', 52: 'Mex Hairless',
       53: 'Newfoundland', 54: 'Pekinese', 55: 'Pit Bull', 56: 'Pomeranian', 57: 'Poodle', 58: 'Pug',
       59: 'Rhodesian', 60: 'Rottweiler', 61: 'Saint Bernard', 62: 'Schnauzer', 63: 'Scotch Terrier', 64: 'Shar_Pei',
       65: 'Shiba Inu', 66: 'Shih-Tzu', 67: 'Siberian Husky', 68: 'Vizsla', 69: 'Yorkie'}


def processed_img(location):
    img = load_img(location, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)  # Replace model1 with model
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    return res

def run():
    img1 = Image.open('meta/logo.jpeg')
    img1 = img1.resize((900, int(900 * img1.height / img1.width)))  # Adjust the width value as needed
    st.title("Dog Breed Classification")
    st.image(img1, use_column_width=True)
    st.markdown(
        "<h4 style='text-align: center; color: #d73b5c;'>* Data is based on 70 classes of dog breed species</h4>",
        unsafe_allow_html=True
    )

    img_file = st.file_uploader("Choose an Image of a Dog", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            print(result)
            st.success("Predicted Dog Breed is: " + result)
run()