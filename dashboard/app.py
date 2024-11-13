import streamlit as st
import helper_tools as ht
import pandas as pd

## One thing I think would be cool is if we recommend images, allow user feedback on the image
## and possibly use recommender system algorithms to recommend new images.
## Or allow user to upload multiple photos and reccommend a photo based on multiple images.

## To run locally, in your terminal type 'streamlit run dashboard/app.py'
# This assumes you are in the Project-Leonardo directory and have already pip installed streamlit

st.title('Image Similarity TEST')

# Allow and read in user input
uploaded_files = st.file_uploader(
        "Choose an image file:", accept_multiple_files=True, type=['png', 'jpg']
    )

if isinstance(uploaded_files, list):
    input = []
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        input.append(bytes_data)
        st.image(bytes_data)
else:
    st.image(uploaded_files)

test_path = r'scrap/gridsearch/resized_images_chunk_modfied_68.h5'
test = ht.h5_to_pandas_metadata(test_path)
st.dataframe(test, hide_index = True)

