import pandas as pd
import requests
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import pyarrow as pa
#import pyarrow.parquet as pq
import h5py
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool

# Define the output file path
maxnum=4

# Define the output file path
output_file = "/scratch/skl5876/images/resize_image_cluster_4/resized_images_cluster_fix.parquet"

def color_columns(img, maxnum):
    # Find the centers of color clusters for an image
    img_data = img.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(data=img_data.astype(np.float32), K=maxnum, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    norms = np.linalg.norm(centers, axis=1)
    sorted_indices = np.argsort(norms)
    centers_sorted = centers[sorted_indices]
    first_center = centers_sorted[0]
    if len(centers_sorted) < maxnum:
        for l in range(maxnum - len(centers_sorted)):
            centers_sorted = np.append(centers_sorted, [first_center], axis=0)
    return centers_sorted

def composition_columns(img, maxnum):
    # Find the composition cluster centers for an image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 300)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_centers = []
    for contour in contours:
        # Compute the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            contour_centers.append([cX, cY])

    contour_centers = np.array(contour_centers, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.02)
    if len(contour_centers) == 0:
        # This case exists and is annoying, so I made all of the clusters at the origin
        sorted_centers = np.array([[0, 0]] * maxnum)
    elif len(contour_centers) == 1:
        # If you only have one contour center, then kmeans no longer returns tuples
        sorted_centers = np.concatenate([contour_centers, np.array([contour_centers[0]] * (maxnum - 1))])
    elif 1 < len(contour_centers) < maxnum:
        # hdf5files struggle to contain information that is not of a uniform size, so we add copies of the origin
        K = len(contour_centers)
        cv2.setRNGSeed(42)
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
        first_s_center = sorted_centers[0]
        for _ in range(maxnum - len(sorted_centers)):
            sorted_centers = np.append(sorted_centers, [first_s_center], axis=0)
    else:
        K = maxnum
        cv2.setRNGSeed(42)
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
    return sorted_centers

def add_cluster_info(input_file, maxnum):
    with h5py.File(input_file, 'r') as df:
        images = np.array(df['images'])
        metadata = np.array(df['metadata'])
        color_matrices = []
        comp_matrices = []

        for i in range(len(images[:])):
            color_cluster_matrix = color_columns(images[i], maxnum)
            comp_cluster_matrix = composition_columns(images[i], maxnum)
            color_matrices.append(color_cluster_matrix)
            comp_matrices.append(comp_cluster_matrix)

    # Convert lists to DataFrame
    df_color = pd.DataFrame({'color_clusters': [str(np.array(matrix).tolist()) for matrix in color_matrices]})
    df_comp = pd.DataFrame({'composition_clusters': [str(np.array(matrix).tolist()) for matrix in comp_matrices]})
    df_metadata = pd.DataFrame({'metadata': [str(np.array(meta).tolist()) for meta in metadata]})

    # Combine all DataFrames into one
    df_combined = pd.concat([df_metadata, df_color, df_comp], axis=1)

    return df_combined


def process_chunk(index):
    df_path = "/scratch/skl5876/images/resized_images2/resized_images_chunk_modfied_{}.h5".format(index)
    new_data = add_cluster_info(df_path, maxnum)
    return new_data

def process_batch(batch_indices):
    results = []
    for index in batch_indices:
        results.append(process_chunk(index))
    return results

if __name__ == '__main__':
    batch_size = 10  # Define the batch size
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
