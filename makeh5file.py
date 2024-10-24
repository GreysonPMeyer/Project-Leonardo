import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import h5py
import numpy as np
import os
#this is a file combined all the data from omni_clean_split_1,2,3csv files
csv_file = "omni_clean.csv"
chunk_size = 1000  # Adjust chunk size based on memory capacity
#this is the directory where the resized images will be saved
output_dir = "/scratch/skl5876/images/resized_images"
os.makedirs(output_dir, exist_ok=True)

def process_chunk(chunk, output_file):
    with h5py.File(output_file, 'w') as hdf5_file:
        dataset_index = 0
        for idx, row in chunk.iterrows():
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
                image_url = row['image_url']
                artist_full_name = row['artist_full_name']
                artwork_name = row['artwork_name']
                artwork_type = row['artwork_type']
                
                response = requests.get(image_url, headers=headers)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                resized_image = image.resize((200, 200))
                
                # Convert image to numpy array
                image_array = np.array(resized_image)
                
                # Save image and metadata to HDF5 file
                hdf5_file.create_dataset(f"images/{dataset_index}", data=image_array, compression="gzip", compression_opts=1)
                hdf5_file.create_dataset(f"metadata/{dataset_index}/artist_full_name", data=artist_full_name)
                hdf5_file.create_dataset(f"metadata/{dataset_index}/artwork_name", data=artwork_name)
                hdf5_file.create_dataset(f"metadata/{dataset_index}/artwork_type", data=artwork_type)
                
                dataset_index += 1
            except Exception as e:
                print(f"Failed to process image at index {idx}: {e}")

chunk_index = 0
for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
    output_file = os.path.join(output_dir, f"resized_images_chunk_{chunk_index}.h5")
    process_chunk(chunk, output_file)
    chunk_index += 1