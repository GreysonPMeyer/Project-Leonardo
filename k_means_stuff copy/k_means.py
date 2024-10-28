import pandas as pd
import requests
import os
import cv2
import numpy as np
from matplotlib.pyplot import show, imshow
import h5py

df_path = "/Users/greysonmeyer/Downloads/resized_images_chunk_0.h5".format(2)

# with h5py.File(df_path, 'r') as df:
#     img = df[f'images/{2}'][:]
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

#     # Apply edge detection (Canny) to find contours
#     edges = cv2.Canny(gray, 30, 300)

#     # Find contours in the edge-detected image
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#     # Prepare data for clustering - use contour centroids or bounding box centers
#     contour_centers = []
#     for contour in contours:
#         # Compute the center of the contour
#         M = cv2.moments(contour)
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             contour_centers.append([cX, cY])

#     # Convert centers to float32 for k-means
#     contour_centers = np.float32(contour_centers)
#     print('contour centers are ', contour_centers)

#     # Define criteria and number of clusters (K)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     if len(contour_centers) < 5:
#         K = len(contour_centers)
#     else:
#         K = 5  # Choose number of clusters for forms
#     _, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     print('labels are ', labels)
#     print('centers are', centers)

#     # Optional: Visualize clusters
#     output_image = img.copy()
#     for idx, center in enumerate(centers):
#         cv2.circle(output_image, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
#     cv2.imshow("Cluster Centers", output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# Calculates the color clusters for a dataset and saves this in k+1 news columns in the dataset.
# The cluster order is determined by the magnitude of the RGB vector in R^3
def color_columns(path):
    color_dict = dict()
    with h5py.File(path, 'r') as df:
        # what is the actual range for the first h5? How to generalize to the other h5 files?
        for i in range(3):
            img = df[f'images/{i}'][:]
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Reshape image to an Mx3 array
            img_data = img_RGB.reshape(-1, 3)
            # Specify the algorithm's termination criteria
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
            # Run the k-means clustering algorithm on the pixel values
            # labels is a column where each entry contains the center associated to that row's pixel
            # centers is the list of the 5 center colors
            # compactness is just a number. Kind of annoying that it't not a number for each cluster tho
            compactness, labels, centers = cv2.kmeans(data=img_data.astype(np.float32), K=5, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
            centers_sorted = sorted(centers, key=lambda x: np.linalg.norm(x))
            color_dict[i] = [compactness, centers_sorted]
    with h5py.File(path, 'a') as df:
        for j in color_dict.keys():
            metadata = df['metadata/{j}']
            metadata.attrs['color_compactness'] = color_dict[j][0]
            colors = color_dict[j][1]
            for k in range(len(colors)):
                metadata.attrs['color_cluster_{k}'] = colors[k]
    return path

# Calculates the composition clusters and saves this in k+1 new columns in the dataset.
# The cluster order is determined by the cluster position with the largest y-value, and then we work top
# to bottom
def composition_columns(path):
    comp_dict = dict()
    # Load image and convert to grayscale
    with h5py.File(path, 'r') as df:
        # what is the actual range for the first h5? How to generalize to the other h5 files?
        for i in range(3):
            img = df[f'images/{i}'][:]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

            # Define criteria and number of clusters (K)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            if len(contour_centers) < 5:
                K = len(contour_centers)
            else:
                K = 5  # Choose number of clusters for forms
            compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
            comp_dict[i] = [compactness, sorted_centers]
    with h5py.File(path, 'a') as df:
        for j in comp_dict.keys():
            metadata = df['metadata/{j}']
            metadata.attrs['composition_compactness'] = comp_dict[j][0]
            composition_coordinates = comp_dict[j][1]
            for k in range(len(composition_coordinates)):
                metadata.attrs['color_cluster_{k}'] = composition_coordinates[k]
    return path

# def color_similarity(image_info, weight, data):
    # Start by resizing the image using Sun's code
    # Then apply color_columns to the resized image
    # Now I would like to calculate the differences between the color_composition of the resized image
    # and the color_composition of each image in the dataset.
    # Then we calculate the Euclidean distance between the first color_column of the resized image and 
    # the first color column of every image in the dataset. Average the K distances to get an average cluster
    # position difference
    # Perform the weighted average of the average cluster position difference and the difference in color
    # composition
    # Store these values in new columns and return the path to the altered dataset, along with the image & 
    # metadata of the image that minimizes the weighted average.


# def composition_similarity(image_info, data):
    # Calculates average distance between the cluster-positions for associated clusters. It also calculates the
    # differences between the overall cluster densities. It then takes the average of these values and puts this
    # weighted average into a new column of the dataset. It returns the minimizing element of the dataset and 
    # the dataset itself


# def similar_art(image_info, color_weight, overall_weight, data):
    # Uses the above functions to return the color-similar artwork and the composition-similar artwork. It then
    # takes the datatset with the two new color_similarity & composition_similarity columns and computes the
    # overall_weighted average of these columns. It then returns the image with the smallest overall_weighted 
    # average. So it returns three images and all of the metadata associated to these images
