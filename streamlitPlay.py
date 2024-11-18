import streamlit as st
st.set_page_config(
    page_title="Project Leanardo", 
    page_icon="🖼️", 
    initial_sidebar_state="collapsed"
)


import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO


## page set up


# project title
st.markdown("# Project Leonardo")
# description
st.write("Using ML to find similar arts.")



## image loader
 

container_image_loader = st.container(border = True) # make container for the image loader
container_image_loader.markdown("#### Image loader") # title
allowed_types = ["png", "jpg", "jpeg"] # add more image types if necessary
uploaded_file = container_image_loader.file_uploader("Select an image...", type=allowed_types) # upload
text_URL = container_image_loader.text_input("or input image URL", value = "https://static.streamlit.io/examples/owl.jpg") # URL input
button_load_image = container_image_loader.button(label = "Load Image") # button to load

if button_load_image: # if button clicked
    # image loading from file
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # resize to 200 x 200
        resized_image = cv2.resize(opencv_image, (200, 200), interpolation = cv2.INTER_LINEAR)
        # display in the center
        col1, col2, col3 = container_image_loader.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image(resized_image, channels="BGR")
        with col3:
            st.write(' ')
    else:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
            response = requests.get(text_URL, headers=headers)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            resized_image = image.resize((200, 200))
            # Convert image to numpy array
            image_array = np.array(resized_image)
            # display in the center
            col1, col2, col3 = container_image_loader.columns(3)
            with col1:
                st.write(' ')
            with col2:
                st.image(image_array, channels="GRB")
            with col3:
                st.write(' ')
        except Exception as e:
                container_image_loader.write(f"Cannot load image")



## parameter setting and generation

form = st.form(key="form_settings") # make a form for submitting the parameters
form.markdown("#### Parameters") # title

col1, col2, col3 = form.columns([1, 3, 1]) # columns to display the different parameter knobs
# k cluser 
k_cluster = col1.text_input(
    "Number of clusters",
    key="k-clusters",
)
# color / composition ratio
CC_ratio = col2.slider(
    "Color/Composition",
    min_value = 0.,
    max_value = 1.,
    value = 0.5, step=0.01,
    key="CC-ratio",
)
# type of art
art_type = col3.selectbox(
    "Art type",
    options = ["Painting", "asd", "asd"],
    key = "style",
)

# button for generation
form.form_submit_button(label="Submit")
