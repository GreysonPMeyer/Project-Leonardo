import pandas as pd
import requests
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from PIL import Image
import helper_tools as ht
import logging
import ast

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
    cv2.setRNGSeed(42)
    if len(contour_centers) == 0:
        # This case exists and is annoying, so I made all of the clusters at the origin
        sorted_centers = np.array([[0,0], [0,0], [0,0], [0,0]])
    elif len(contour_centers) == 1:
        # If you only have one contour center, then kmeans no longer returns tuples
        sorted_centers = np.concatenate((contour_centers,np.array([contour_centers[0], contour_centers[0], contour_centers[0]])))
    elif 1 < len(contour_centers) < 4:
        # hdf5files struggle to contain informatio that is not of a uniform size, so we add copies of the origin
        K = len(contour_centers)
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
        first_s_center = sorted_centers[0]
        for _ in range(4 - len(sorted_centers)):
            sorted_centers = np.concatenate((sorted_centers,first_s_center))
    else:
        K = 4
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = np.array(sorted(centers, key=lambda c: (c[1], c[0]), reverse=True))

    return np.float64(sorted_centers)

def color_similarity_df(input_image, df:pd.DataFrame):
    # Calculates how similar the images from the dataset are to the input image based on the values of the 
    # color clusters

    color_image_clusters = color_columns(input_image)
    distances = df['color_clusters'].apply(lambda row: np.linalg.norm(row - color_image_clusters,2))
    # print('new_norm', np.min(distances))
    # distances = df['color_clusters'].apply(lambda row: cdist(row, color_image_clusters, metric = 'euclidean').min(axis =1).mean())
    print('new_dist', np.min(distances))
    color_winner_index = distances.argmin()
    distance_vector = np.array(distances)

    return color_winner_index, distance_vector

def composition_similarity_df(input_image, df:pd.DataFrame):
    # Calculates how similar the images from the dataset are to the input image based on the values of the 
    # composition clusters
    composition_image_clusters = composition_columns(input_image)
    distances = df['composition_clusters'].apply(lambda row: cdist(row, composition_image_clusters, metric = 'euclidean').min(axis = 1).sum()/row.shape[0])
    # distances = df['composition_clusters'].apply(lambda row: np.linalg.norm(row - composition_image_clusters,2))
    composition_winner_index = distances.argmin()
    distance_vector = np.array(distances)

    return composition_winner_index, distance_vector

def similar_art(image, weight, data:pd.DataFrame):
    # Calculates the overall match and consolidates all of the information
    color_match_index, color_averages = color_similarity_df(image, data)
    color_winning_avg = color_averages[color_match_index]
    comp_match_index, comp_averages = composition_similarity_df(image, data)
    comp_winning_avg = comp_averages[comp_match_index]
    overall_avgs = weight * color_averages + (1 - weight) * comp_averages
    overall_match_index = np.argmin(overall_avgs)
    overall_winning_avg = overall_avgs[overall_match_index]

    return color_match_index, color_winning_avg, comp_match_index, comp_winning_avg, overall_match_index, overall_winning_avg

if __name__ == "__main__":
    # This is the chunk of data being used
    # df_path = "./scrap/gridsearch/resized_images_chunk_modfied_1.h5"
    # # dir = fr'./scrap/gridsearch/'
    # test_path = create_test_set(df_path)
    test_image_path = './scrap/validation/test_images/test2.jpg'
    test_image = cv2.imread(test_image_path)
    test_image_conv = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    input_img = resize_and_convert_image(test_image_conv, (200, 200))
    # display_art(input_img, 0.5, [test_path])


    # write_final_parquet('./test3.parquet.gzip')
    from time import time
    ts = time()


    df = pd.read_parquet('./resized_images_cluster_fix.parquet')
    # df = pd.read_parquet('./test3.parquet.gzip')
    # print('DF shape:', df.shape)
    df['color_clusters'] = df['color_clusters'].apply(lambda x: np.array(ast.literal_eval(x)))
    df['composition_clusters'] = df['composition_clusters'].apply(lambda x: np.array(ast.literal_eval(x)))
    # df['metadata'] = df['metadata'].apply(lambda x: np.array(ast.literal_eval(x)))
    # color_centers = df['color_centers'].str.replace('[', '').str.replace(']', '').str.replace('\n', '').apply(lambda x: np.fromstring(x, sep = ' ').reshape(4,3))
    # comp_centers = df['comp_centers'].str.replace('[', '').str.replace(']', '').str.replace('\n', '').apply(lambda x: np.fromstring(x, sep = ' ').reshape(4,2))
    # test_img = resize_and_convert_image(cv2.cvtColor(cv2.imread('./scrap/validation/test_images/test2.jpg'), cv2.COLOR_BGR2RGB))
    # test_img = resize_and_convert_image(cv2.cvtColor(cv2.imread('/Users/dawsonkinsman/Downloads/um-dearborn.png'), cv2.COLOR_BGR2RGB))


    color_idx, color_dist = color_similarity_df(input_img, df)
    comp_idx, comp_dist = composition_similarity_df(input_img, df)
    print(color_idx, comp_idx)
    df.loc[[color_idx, comp_idx], :].to_csv('./test_sim.csv', index=True)
    color_match, _, comp_math, _, overall_match = similar_art(input_img, 0.5, df)
    print('Color:', color_match)
    print('Composition:', comp_math)
    print('Overall:', overall_match)
    df.loc[[color_match, comp_math, overall_match], :].to_csv('./test_sim2.csv', index=True)

    te = time()
    print(f'{round(te-ts, 2)} sec')