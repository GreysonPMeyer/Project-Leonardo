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
df_path = "/Users/greysonmeyer/Downloads/resized_images_chunk_modfied_105.h5"

def color_columns(img):
    # Find the centers of color clusters for an image
    img_data = img.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(data=img_data.astype(np.float32), K=4, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    norms = np.linalg.norm(centers, axis=1)
    sorted_indices = np.argsort(norms)
    centers_sorted = centers[sorted_indices]
    first_center = centers_sorted[0]
    if len(centers_sorted) < 4:
        for l in range(4 - len(centers_sorted)):
            np.append(centers_sorted, first_center, axis=0)

    # Re-map labels according to the sorted order of centers for display purposes
    # The labels are nice to have if you want to display the image using only the cluster colors
    # sorted_labels = np.zeros_like(labels)
    # for new_label, old_label in enumerate(sorted_indices):
    #     sorted_labels[labels == old_label] = new_label

    return centers_sorted

def composition_columns(img):
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

    # Define criteria and number of clusters (K)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    if len(contour_centers) == 0:
        sorted_centers = np.array([[0,0], [0,0], [0,0], [0,0]])
    elif len(contour_centers) == 1:
        sorted_centers = np.concatenate((contour_centers, (np.array([contour_centers[0], contour_centers[0], contour_centers[0]]))))
    elif 1 < len(contour_centers) < 4:
        K = len(contour_centers)
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        s_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
        first_s_center = s_centers[0]
        for _ in range(4 - len(s_centers)):
            s_centers.append(first_s_center)
        sorted_centers = np.array(s_centers)
    else:
        K = 4  # Choose number of clusters for forms
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
        
    return sorted_centers
  
def add_cluster_info(data):
    # This was included for when I was debugging. Shouldn't be necessary now.
    with h5py.File(data, 'r+') as df:
        del df['color_clusters']
        del df['composition_clusters']

    with h5py.File(data, 'a') as df:
        images = np.array(df['images'])
        color_matrices = []
        comp_matrices = []

        for i in range(len(images[:])):
            print('image number: ', i)
            color_cluster_matrix = color_columns(images[i])
            comp_cluster_matrix = composition_columns(images[i])
            color_matrices.append(color_cluster_matrix)
            comp_matrices.append(comp_cluster_matrix)

        df.create_dataset('color_clusters', data=color_matrices)
        df.create_dataset('composition_clusters', data=comp_matrices)

    return data

new_data = add_cluster_info(df_path)

# Double checks that it works
with h5py.File(new_data, 'r') as df:
    print(df['color_clusters'].shape)
    print(df['composition_clusters'].shape)