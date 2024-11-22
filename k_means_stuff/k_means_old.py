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
df_path = "/Users/greysonmeyer/Downloads/resized_images_chunk_0.h5".format(2)

def create_test_set(input_file_path):
    # Creates an h5file from a chunk of the dataset, for the sake of time for testing
    output_file_path = "/Users/greysonmeyer/Desktop/Erdos_Work/k_means_stuff/tester.h5"  

    with h5py.File(input_file_path, 'r') as original_file:
        with h5py.File(output_file_path, 'w') as new_file:
            for i in range(45):
                new_file.create_dataset(f"images/{i}", data=original_file[f'images/{i}'], compression="gzip", compression_opts=1)
                new_file.create_dataset(f"metadata/{i}/artist_full_name", data=original_file[f"metadata/{i}/artist_full_name"])
                new_file.create_dataset(f"metadata/{i}/artwork_name", data=original_file[f"metadata/{i}/artwork_name"])
                new_file.create_dataset(f"metadata/{i}/artwork_type", data=original_file[f"metadata/{i}/artwork_type"])

    print(f"Successfully created a copy of the HDF5 file with limited data at '{output_file_path}'.")
    return output_file_path

def color_columns(path):
    # Calculates the color clusters, sorts them, then includes them in the metadata

    color_dict = dict()
    with h5py.File(path, 'a') as df:
        for i in df['images'].keys():
            img_RGB = np.array(df[f'images/{i}'][:])
            
            # Reshape image to an Mx3 array
            img_data = img_RGB.reshape(-1, 3)

            # Specify the algorithm's termination criteria
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
            # Run the k-means clustering algorithm on the pixel values
            # labels is a column where each entry contains the center associated to that row's pixel
            # centers is the list of the 5 center colors
            # compactness is just a number
            compactness, labels, centers = cv2.kmeans(data=img_data.astype(np.float32), K=5, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
            norms = np.linalg.norm(centers, axis=1)

            # Sort centers by norms and get sorted indices
            sorted_indices = np.argsort(norms)
            centers_sorted = centers[sorted_indices]

            # Re-map labels according to the sorted order of centers
            # The labels are nice to have if you want to display the image using only the cluster colors
            sorted_labels = np.zeros_like(labels)
            for new_label, old_label in enumerate(sorted_indices):
                sorted_labels[labels == old_label] = new_label

            # If there are less than k clusters, add extra white clusters
            if len(centers_sorted) < 5:
                for l in range(5 - len(centers_sorted)):
                    np.append(centers_sorted, np.array([255, 255, 255]))

            color_dict[i] = [compactness, centers_sorted, sorted_labels]

            if i == '0':
                colours = centers_sorted[sorted_labels].reshape(-1, 3)
                img_colours = colours.reshape(img_RGB.shape)
                plt.imshow(img_colours.astype(np.uint8))
                plt.show()

            if i == '44':
                colours = centers_sorted[sorted_labels].reshape(-1, 3)
                img_colours = colours.reshape(img_RGB.shape)
                plt.imshow(img_colours.astype(np.uint8))
                plt.show()

        for j in color_dict.keys():
            df.create_dataset(f"metadata/{j}/color_clusters", data=color_dict[j][1])
            df.create_dataset(f"metadata/{j}/labels", data = color_dict[j][2])

    return path

def composition_columns(path):
    # Calculates the composition clusters and adds them to the metadata
    # The cluster order is determined by the cluster position with the largest y-value, and then we work top
    # to bottom 

    comp_dict = dict()
    # Load image and convert to grayscale
    with h5py.File(path, 'a') as df:
        for i in df['images'].keys():
            img = np.array(df[f'images/{i}'])
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
            compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
            sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
            comp_dict[i] = [compactness, sorted_centers]

        # I haven't been using comp_compactness for anything, but we could maybe use it if we wanted to
        for j in comp_dict.keys():
            df.create_dataset(f"metadata/{j}/comp_compactness", data=comp_dict[j][0])
            df.create_dataset(f"metadata/{j}/comp_clusters", data=comp_dict[j][1])
        
    return path

def color_similarity(image, data):
    # Calculates how similar the images from the dataset are to the input image based on the values of the 
    # color clusters

    color_image_path = color_columns(image)
    color_data_path = color_columns(data)
    
    with h5py.File(color_data_path, 'r') as df, h5py.File(color_image_path, 'r') as color:
        # This is a list of 5 x 3 arrays
        color_data = [
            df[f'metadata/{group}/color_clusters'][:]
            for group in df['metadata'].keys()
        ]

        # This is a 5 x 3 array
        image_clusters = color['metadata/0/color_clusters'][:]

        # Each row is a cluster and each column is an image
        distances = []
        for datum in color_data:
            # Compare the Euclidean distance between each array in color_data and the image_clusters
            cluster_distance = cdist(datum, image_clusters, metric='euclidean')
            # Calculate the minimum distance for each centroid in datum to any centroid in image_clusters
            min_distances = cluster_distance.min(axis=1)
            color_similarity_score = np.mean(min_distances)
            distances.append(color_similarity_score)         
        
        # The index of the image with the smallest mean centroid distance
        color_winner_index = list(df['metadata'].keys())[np.argmin(distances)]
        # We'll use distances again for the overall comparison. We need it to be an array for the calculations
        distance_vector = np.array(distances)

    return color_winner_index, distance_vector

def composition_similarity(image, data):
    # Calculates how similar the images from the dataset are to the input image based on the positions of the 
    # color clusters
    comp_image_path = composition_columns(image)
    comp_data_path = composition_columns(data)
    with h5py.File(comp_data_path, 'r') as df, h5py.File(comp_image_path, 'r') as img:

        comp_clusters = [
            df[f'metadata/{group}/comp_clusters'][:]
            for group in df['metadata']
        ]

        image_clusters = img[f'metadata/0/comp_clusters'][:]
        # plt.imshow(img['images/0'][:])
        # x_coords, y_coords = image_clusters[:, 0], image_clusters[:, 1]
        # plt.scatter(x_coords, y_coords, c='red', s=100, edgecolor='black', label='Centroids')
        # plt.show()

        # Same process as in color similarity
        distances = []
        for datum in comp_clusters:
            cluster_distance = cdist(image_clusters, datum, metric='euclidean')
            row_indices, col_indices = linear_sum_assignment(cluster_distance)
            total_distance = cluster_distance[row_indices, col_indices].sum()
            average_distance = total_distance / len(row_indices)
            distances.append(average_distance)

        winner_image_index = list(df['metadata'].keys())[np.argmin(distances)]
        comp_dist_vector = np.array(distances)

        # winner_clusters = df[f'metadata/{winner_image_index}/comp_clusters'][:]
        # plt.imshow(df[f'images/{winner_image_index}'][:])
        # x_coords, y_coords = winner_clusters[:, 0], winner_clusters[:, 1]
        # plt.scatter(x_coords, y_coords, c='red', s=100, edgecolor='black', label='Centroids')
        # plt.show()

        # plt.imshow(df[f'images/{winner_image_index}'][:])
        # plt.show()

    return winner_image_index, comp_dist_vector

def similar_art(image, weight, data):
    # Start by resizing the image
    img = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)

    # We create an h5file to store the image in, along with some empty metadata groups to be filled in by 
    # color_columns and composition_columns
    img_file = "/Users/greysonmeyer/Desktop/Erdos_Work/k_means_stuff/input_image.h5"
    with h5py.File(img_file, 'w') as file:
        file.create_dataset("images/0", data=img, compression="gzip", compression_opts=1)
        file.create_group("metadata/0") 

    color_match_index, color_averages = color_similarity(img_file, data)
    comp_match_index, comp_averages = composition_similarity(img_file, data)
    overall_avgs = weight * color_averages + (1 - weight) * comp_averages

    with h5py.File(data, 'r') as df:
        # Finds the index of the overall match
        overall_match_index = list(df['metadata'].keys())[np.argmin(overall_avgs)]
        # Finds the winning images
        img_color = df[f'images/{color_match_index}'][:]    
        img_comp = df[f'images/{comp_match_index}'][:]
        img_overall = df[f'images/{overall_match_index}'][:]

        # Titles for display purposes
        color_title = str(df[f'metadata/{color_match_index}/artwork_name'][()].decode('utf-8')) + ' by ' + str(df[f'metadata/{color_match_index}/artist_full_name'][()].decode('utf-8'))
        comp_title = df[f'metadata/{comp_match_index}/artwork_name'][()].decode('utf-8') + ' by ' + df[f'metadata/{comp_match_index}/artist_full_name'][()].decode('utf-8')
        overall_title = df[f'metadata/{overall_match_index}/artwork_name'][()].decode('utf-8') + ' by ' + df[f'metadata/{overall_match_index}/artist_full_name'][()].decode('utf-8')

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_color)
        axes[0].set_title(color_title)
        axes[0].axis('off')

        axes[1].imshow(img_comp)
        axes[1].set_title(comp_title)
        axes[1].axis('off')

        axes[2].imshow(img_overall)
        axes[2].set_title(overall_title)
        axes[2].axis('off')

        plt.subplots_adjust(wspace=0.4)
        plt.tight_layout()
        plt.show()

    return 'color match: ', color_title, '; composition match: ', comp_title, '; overall match: ', overall_title

test_path = create_test_set(df_path)
test_image_path = '/Users/greysonmeyer/Downloads/canal310.jpg'
test_image = cv2.imread(test_image_path)
test_image_conv = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
similar_art(test_image_conv, 0.5, test_path)