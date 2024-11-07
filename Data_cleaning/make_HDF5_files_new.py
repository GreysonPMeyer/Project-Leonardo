import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import h5py
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

csv_file = "omni_clean.csv"
chunk_size = 1000  # Adjust chunk size based on memory capacity
# Direction where you want to save
output_dir = "/scratch/skl5876/images/resized_images2"
os.makedirs(output_dir, exist_ok=True)

def download_and_process_image(row):
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
        # If we have an RGBA image, we need to convert it to RGB
        def resize_and_convert_image(image_array, target_size=(200, 200)):
            image = Image.fromarray(image_array)
            image = image.resize(target_size, Image.LANCZOS)
            
            if image.mode == 'RGBA':
                # Create a white background image
                background = Image.new('RGB', target_size, (255, 255, 255))
                # Paste the RGBA image onto the background
                background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        image_array=resize_and_convert_image(image_array, (200, 200))

        return (image_array, artist_full_name, artwork_name, artwork_type, image_url)
    except Exception as e:
        print(f"Failed to process image: {e}")
        return None

def process_chunk(chunk, output_file):
    with h5py.File(output_file, 'w') as hdf5_file:
        images = []
        metadata = []
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(download_and_process_image, [row for _, row in chunk.iterrows()])
        
        for result in results:
            if result is not None:
                image_array, artist_full_name, artwork_name, artwork_type, image_url = result
                images.append(image_array)
                metadata.append((str(artist_full_name), str(artwork_name), str(artwork_type), str(image_url)))
                
        # Save images and metadata to HDF5 file
        hdf5_file.create_dataset("images", data=np.array(images), compression="gzip", compression_opts=1)
        hdf5_file.create_dataset("metadata", data=np.array(metadata, dtype=h5py.string_dtype()), compression="gzip", compression_opts=1)

chunk_index = 0
for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
    output_file = os.path.join(output_dir, f"resized_images_chunk_modfied_{chunk_index}.h5")
    process_chunk(chunk, output_file)
    chunk_index += 1