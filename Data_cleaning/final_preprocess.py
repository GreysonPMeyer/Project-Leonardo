import pandas as pd
import requests
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import pyarrow as pa
#import pyarrow.parquet as pq
import h5py
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
import time

# Define the output file path
maxnum=4

# Define the output file path
output_file = "/scratch/skl5876/images/resize_image_cluster_4/resized_images_cluster_fix_fin_200.parquet"
def resize_and_convert_image(image_array, target_size=(200, 200)):
    # Copy of Sun's code for consistency in resizing
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

def color_columns(img):
    cv2.setRNGSeed(42)
    # Calculates the color clusters of the image

    img_data = img.reshape(-1, 3)

    # Specify the algorithm's termination criteria
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

    # Run the k-means clustering algorithm on the pixel values
    # labels is a column where each entry contains the center associated to that row's pixel
    # centers is the list of the 5 center colors
    # compactness is just a number
    compactness, labels, centers = cv2.kmeans(data=img_data.astype(np.float32), K=4, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    norms = np.linalg.norm(centers, axis=1)

    # Sort centers by norms and get sorted indices
    sorted_indices = np.argsort(norms)
    centers_sorted = centers[sorted_indices]

    # If there are less than k clusters, add extra white clusters
    if len(centers_sorted) < 4:
        first_center = centers_sorted[0]
        for l in range(4 - len(centers_sorted)):
            centers_sorted = np.concatenate(centers_sorted, np.array(first_center))

    return np.float64(centers_sorted)

def composition_columns(image):
    cv2.setRNGSeed(42)
    # Calculates the composition clusters and adds them to the metadata
    # The cluster order is determined by the cluster position with the largest y-value, and then we work top
    # to bottom 

    # Load image and convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (Canny) to find contours
    edges = cv2.Canny(gray, 30, 300)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare data for clustering - use contour centroids or bounding box centers
    contour_centers = []
    for contour in contours:
        # Compute the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            contour_centers.append([cX, cY])

    # Convert centers to float32 for k-means
    contour_centers = np.float32(contour_centers)
    # print(contour_centers)

    # Define criteria and number of clusters (K)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    if len(contour_centers) == 0:
        # This case exists and is annoying, so I made all of the clusters at the origin
        sorted_centers = np.array([[0,0], [0,0], [0,0], [0,0]])
    elif len(contour_centers) == 1:
        # If you only have one contour center, then kmeans no longer returns tuples
        sorted_centers = np.concatenate((contour_centers,np.array([contour_centers[0], contour_centers[0], contour_centers[0]])))
    elif 1 < len(contour_centers) < 4:
        cv2.setRNGSeed(42)
        # hdf5files struggle to contain informatio that is not of a uniform size, so we add copies of the origin
        K = len(contour_centers)
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
        first_s_center = sorted_centers[0]
        extras = []
        for _ in range(4 - len(sorted_centers)):
            extras.append(first_s_center)
        extras_array = np.array(extras)
        sorted_centers = np.concatenate((sorted_centers,extras_array))
    else:
        cv2.setRNGSeed(42)
        K = 4
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = np.array(sorted(centers, key=lambda c: (c[1], c[0]), reverse=True))

    return np.float64(sorted_centers)

def add_cluster_info(input_file):
    print(f"Processing file: {input_file} \n")
    start_time = time.time()
    
    
    with h5py.File(input_file, 'r') as df:
        images = np.array(df['images'])
        metadata = np.array(df['metadata'])
        color_matrices = []
        comp_matrices = []
        valid_metadata = []

        for i in range(len(images[:])):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
            try:
                response = requests.get(df['metadata'][i][3], headers=headers)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                image = image.resize((200, 200))
                image_array = resize_and_convert_image(np.array(image), (200, 200))
                
                color_cluster_matrix = color_columns(image_array)
                comp_cluster_matrix = composition_columns(image_array)
                color_matrices.append(color_cluster_matrix)
                comp_matrices.append(comp_cluster_matrix)
                valid_metadata.append(metadata[i])
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error occurred: {e}. Status code: {response.status_code}")
                if e.response.status_code == 404:
                    print(f"Image not found for URL: {df['metadata'][i][3]}. Skipping...")
                elif e.response.status_code == 502:
                    print(f"Bad Gateway for URL: {df['metadata'][i][3]}. Skipping...")
                else:
                    print(f"HTTP error occurred: {e}. Skipping...")
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Skipping...")


    # Convert lists to DataFrame
    df_color = pd.DataFrame({'color_clusters': [str(np.array(matrix).tolist()) for matrix in color_matrices]})
    df_comp = pd.DataFrame({'composition_clusters': [str(np.array(matrix).tolist()) for matrix in comp_matrices]})
    df_metadata = pd.DataFrame({'metadata': [str(np.array(meta).tolist()) for meta in valid_metadata]})

    # Combine all DataFrames into one
    df_combined = pd.concat([df_metadata, df_color, df_comp], axis=1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished processing file: {input_file} in {elapsed_time:.2f} seconds \n")
    
    return df_combined


def process_chunk(index):
    df_path = f"/scratch/skl5876/images/resized_images2/resized_images_chunk_modfied_{index}.h5"
    new_data = add_cluster_info(df_path)
    return new_data

def process_batch(batch_indices):
    results = []
    for index in batch_indices:
        result = process_chunk(index)
        results.append(result)
    return results

if __name__ == '__main__':
    batch_size = 40  # Define the batch size
    indices = list(range(361))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    with Pool() as pool:
        for batch in batches:
            results = pool.map(process_chunk, batch)

            # Combine all results into a single DataFrame
            combined_data = pd.concat(results, ignore_index=True)

            # Write the combined data to the Parquet file incrementally
            if os.path.exists(output_file):
                # Append to existing Parquet file
                existing_data = pd.read_parquet(output_file)
                combined_data = pd.concat([existing_data, combined_data], ignore_index=True)
                combined_data.to_parquet(output_file, compression='gzip')
            else:
                # Write new Parquet file
                combined_data.to_parquet(output_file, compression='gzip')

    # Double checks that it works
    df = pd.read_parquet(output_file)
    print(df['color_clusters'].shape)
    print(df['composition_clusters'].shape)