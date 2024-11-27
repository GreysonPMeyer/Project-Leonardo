import streamlit as st
st.set_page_config(
    page_title="Project Leonardo", 
    page_icon="🖼️", 
    initial_sidebar_state="collapsed"
)


import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import requests
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from k_means_stuff.vectorized_kmeans import display_art, resize_and_convert_image


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
button_load_image = container_image_loader.button(label = "Load Image")

with st.form(key="my_form"):
     # button to load
    # Add form elements here (e.g., text inputs, sliders)
    col1, col2, col3 = st.columns([1, 3, 1]) # columns to display the different parameter knobs
    # k cluser 
    k_cluster = col1.text_input(
        "Number of clusters",
        key="k-clusters",
    )
    # color / composition ratio
    slider = col2.slider(
        "Color/Composition",
        min_value = 0.,
        max_value = 1.,
        value = 0.5, step=0.01,
        key="slider",
    )
    # type of art
    art_type = col3.selectbox(
        "Art type",
        options = ["Painting", "asd", "asd"],
        key = "style",
    )

    # Submit button
    submit_button = st.form_submit_button(label="Submit")

# Code that runs only after the submit button is clicked

if "image_array" not in st.session_state:
    st.session_state.image_array = None

if button_load_image: # if button clicked
    # image loading from file
    if uploaded_file is not None:
        # Convert the file to an opencv image.

        image_stream = BytesIO(uploaded_file.read())

        # Open the image with PIL.Image
        pil_image = Image.open(image_stream)
        resized_image = pil_image.resize((200, 200))
            
        # Convert image to numpy array
        image_array = np.array(resized_image)
        st.session_state.image_array = resize_and_convert_image(image_array, (200, 200))

        # file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # opencv_image = cv2.imdecode(file_bytes, 1)
        # final_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        # resize to 200 x 200
        # resized_image = cv2.resize(opencv_image, (200, 200), interpolation = cv2.INTER_LINEAR)
        # resized_image = resize_and_convert_image(final_image, (200, 200))
        # st.session_state.image_array = np.array(resized_image)
        # print('this got reached!')
        # display in the center

        
        col1, col2, col3 = container_image_loader.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image(st.session_state.image_array)
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
            st.session_state.image_array=resize_and_convert_image(image_array, (200, 200))

            # response = requests.get(text_URL, headers=headers)
            # response.raise_for_status()

            # image_color_array = np.array(bytearray(response.content), dtype=np.uint8)
            # img_color_BGR = cv2.imdecode(image_color_array, cv2.IMREAD_COLOR)
            # img_color = cv2.cvtColor(img_color_BGR, cv2.COLOR_BGR2RGB)
            # resized_image = resize_and_convert_image(img_color, (200, 200))
            # # Convert image to numpy array
            # st.session_state.image_array = np.array(resized_image)
            col1, col2, col3 = container_image_loader.columns(3)
            with col1:
                st.write(' ')
            with col2:
                st.image(st.session_state.image_array)
            with col3:
                st.write(' ')
        except Exception as e:
                container_image_loader.write(f"Cannot load image")

if "slider" not in st.session_state:
    st.session_state.slider = slider

if submit_button:
    # url = 'https://raw.githubusercontent.com/BotanCevik2/Project-Leonardo/main/resized_images_cluster_fix.parquet'
    # print(url)
    # df = pd.read_parquet(url, engine="pyarrow")
    # dataset_list = ["/Users/greysonmeyer/Downloads/resized_images_chunk_modfied_105.h5"]
    img_color, color_title, img_comp, comp_title, img_overall, overall_title = display_art(st.session_state.image_array, st.session_state.slider)
    images = [img_color, img_comp, img_overall]
    # color_image = Image.open(img_color)
    # comp_image = Image.open(img_comp)
    # overall_image = Image.open(img_overall)

    col1, col2, col3 = st.columns([1, 3, 1])
    st.image(img_color, caption=f"Color Recommendation: {color_title}", use_container_width=True)
    st.image(img_comp, caption=f"Composition Recommendation: {comp_title}", use_container_width=True)
    st.image(img_overall, caption=f"Overall Weighted Recommendation: {overall_title}", use_container_width=True)
