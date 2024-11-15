import pandas as pd
import requests
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# This is the chunk of data being used
df_path = "/Users/greysonmeyer/Downloads/resized_images_chunk_modfied_0.h5"

def color_columns(img):
    img_data = img.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(data=img_data.astype(np.float32), K=5, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    norms = np.linalg.norm(centers, axis=1)
    sorted_indices = np.argsort(norms)
    centers_sorted = centers[sorted_indices]
    if len(centers_sorted) < 5:
        for l in range(5 - len(centers_sorted)):
            np.append(centers_sorted, np.array([255, 255, 255]))

    return centers_sorted

def composition_columns(img):
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

    contour_centers = np.float32(contour_centers)
    print(contour_centers)

    # Define criteria and number of clusters (K)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    if len(contour_centers) < 5:
        K = len(contour_centers)
    else:
        K = 5  # Choose number of clusters for forms
    compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
        
    return sorted_centers
  
def add_cluster_info(data):
    with h5py.File(data, 'a') as df:
        metadata = np.array(df['metadata'])
        images = np.array(df['images'])
        color_matrices = []
        comp_matrices = []

        for i in range(len(images[:])):
            color_cluster_matrix = color_columns(images[i])
            comp_cluster_matrix = composition_columns(images[i])
            color_matrices.append(color_cluster_matrix)
            comp_matrices.append(comp_cluster_matrix)

        metadata2 = np.column_stack((metadata, color_matrices))
        metadata3 = np.column_stack((metadata2, comp_matrices))
        print(metadata3.shape)

    return data

new_data = add_cluster_info(df_path)
with h5py.File(new_data, 'r') as df:
    print(df['metadata'].shape)
    print(df['metadata'][0][3].shape)
    print(df['metadata'][0][4].shape)