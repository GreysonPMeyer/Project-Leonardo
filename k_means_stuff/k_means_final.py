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
from PIL import Image

def create_test_set(input_file_path):
    # Creates an h5file from a chunk of the dataset, for the sake of time for testing
    output_file_path = "/Users/greysonmeyer/Desktop/Erdos_Work/k_means_stuff/tester.h5"  

    with h5py.File(input_file_path, 'r') as original_file:
        with h5py.File(output_file_path, 'w') as new_file:
            new_file.create_dataset("images", data=original_file['images'][:45], compression="gzip", compression_opts=1)
            new_file.create_dataset("metadata", data=np.array(original_file['metadata'][:45]), compression="gzip", compression_opts=1)
            new_file.create_dataset('color_clusters', data=original_file['color_clusters'][:45])
            new_file.create_dataset('composition_clusters', data=original_file['composition_clusters'][:45])

    print(f"Successfully created a copy of the HDF5 file with limited data at '{output_file_path}'.")
    return output_file_path

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
            np.append(centers_sorted, np.array(first_center), axis=0)

    return centers_sorted

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
    print(contour_centers)

    # Define criteria and number of clusters (K)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    if len(contour_centers) == 0:
        # This case exists and is annoying, so I made all of the clusters at the origin
        sorted_centers = np.array([[0,0], [0,0], [0,0], [0,0]])
    elif len(contour_centers) == 1:
        # If you only have one contour center, then kmeans no longer returns tuples
        sorted_centers = contour_centers.concatenate(np.array([contour_centers[0], contour_centers[0], contour_centers[0]]))
    elif 1 < len(contour_centers) < 4:
        # hdf5files struggle to contain informatio that is not of a uniform size, so we add copies of the origin
        K = len(contour_centers)
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
        first_s_center = sorted_centers[0]
        for _ in range(5 - len(sorted_centers)):
            sorted_centers.concatenate(first_s_center)
    else:
        K = 4
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = np.array(sorted(centers, key=lambda c: (c[1], c[0]), reverse=True))
        
    return sorted_centers

def color_similarity(image, data):
    # Calculates how similar the images from the dataset are to the input image based on the values of the 
    # color clusters

    color_image_clusters = color_columns(image)
    
    with h5py.File(data, 'r') as df:
        # This is a list of 5 x 3 arrays
        color_data = df['color_clusters'][:]

        # Each row is a cluster and each column is an image
        distances = []
        for datum in color_data:
            # Compare the Euclidean distance between each array in color_data and the image_clusters
            cluster_distance = cdist(datum, color_image_clusters, metric='euclidean')
            # Calculate the minimum distance for each centroid in datum to any centroid in image_clusters
            min_distances = cluster_distance.min(axis=1)
            color_similarity_score = np.mean(min_distances)
            distances.append(color_similarity_score)           
        
        # The index of the image with the smallest mean centroid distance
        color_winner_index = np.argmin(distances)
        # We'll use distances again for the overall comparison. We need it to be an array for the calculations
        distance_vector = np.array(distances)

    return color_winner_index, distance_vector

def composition_similarity(image, data):
    # Calculates how similar the images from the dataset are to the input image based on the positions of the 
    # color clusters
    comp_image_clusters = composition_columns(image)
    with h5py.File(data, 'r') as df:
        comp_clusters = df['composition_clusters'][:]
        # Same process as in color similarity
        distances = []
        for datum in comp_clusters:
            cluster_distance = cdist(comp_image_clusters, datum, metric='euclidean')
            row_indices, col_indices = linear_sum_assignment(cluster_distance)
            total_distance = cluster_distance[row_indices, col_indices].sum()
            average_distance = total_distance / len(row_indices)
            distances.append(average_distance)

        winner_image_index = np.argmin(distances)
        comp_dist_vector = np.array(distances)

        # If you would like to see the clusters for the input image, use this code
        # plt.imshow(image)
        # x_coords, y_coords = comp_image_clusters[:, 0], comp_image_clusters[:, 1]
        # plt.scatter(x_coords, y_coords, c='red', s=100, edgecolor='black', label='Centroids')
        # plt.show()

    return winner_image_index, comp_dist_vector

def similar_art(image, weight, data):
    # Calculates the overall match and consolidates all of the information
    color_match_index, color_averages = color_similarity(image, data)
    color_winning_avg = color_averages[color_match_index]
    comp_match_index, comp_averages = composition_similarity(image, data)
    comp_winning_avg = comp_averages[comp_match_index]
    overall_avgs = weight * color_averages + (1 - weight) * comp_averages
    overall_match_index = np.argmin(overall_avgs)
    overall_winning_avg = overall_avgs[overall_match_index]

    return [color_match_index, color_winning_avg, comp_match_index, comp_winning_avg, overall_match_index, overall_winning_avg]
    
def display_art(image, weight, data_list):
    # This code will identify the color match, composition match and overall match across a collection of 
    # data chunks. I have so far only tested it with one chunk

    color_indices = []
    color_avgs = []
    comp_indices = []
    comp_avgs = []
    overall_indices = []
    overall_avgs = []

    for dataset in data_list:
        winners = similar_art(image, weight, dataset)
        color_indices.append(winners[0])
        color_avgs.append(winners[1])
        comp_indices.append(winners[2])
        comp_avgs.append(winners[3])
        overall_indices.append(winners[4])
        overall_avgs.append(winners[5])

    # The index corresponds to the data chunk in data_list
    color_ds = np.argmin(color_avgs)
    comp_ds = np.argmin(comp_avgs)
    overall_ds = np.argmin(overall_avgs)

    with h5py.File(data_list[color_ds], 'r') as color:
        # Find the color winner in color_ds
        color_match_index = color_indices[color_ds]

        # Downloads the image from the url and makes it presentable
        img_color_url = color['metadata'][color_match_index][3].decode('utf-8')
        print(img_color_url)
        response = requests.get(img_color_url)
        image_color_array = np.array(bytearray(response.content), dtype=np.uint8)
        img_color_BGR = cv2.imdecode(image_color_array, cv2.IMREAD_COLOR)
        img_color = cv2.cvtColor(img_color_BGR, cv2.COLOR_BGR2RGB)

        # Title for the color winner
        color_title = str(color['metadata'][color_match_index][1].decode('utf-8')) + ' by ' + str(color['metadata'][color_match_index][0].decode('utf-8'))

    with h5py.File(data_list[comp_ds], 'r') as comp:
        # Find the composition winner in comp_ds
        comp_match_index = comp_indices[comp_ds]

        # Downloads the image from the url and makes it presentable
        img_comp_url = comp['metadata'][comp_match_index][3].decode('utf-8')
        response = requests.get(img_comp_url)
        image_comp_array = np.array(bytearray(response.content), dtype=np.uint8)
        img_comp_BGR = cv2.imdecode(image_comp_array, cv2.IMREAD_COLOR)
        img_comp = cv2.cvtColor(img_comp_BGR, cv2.COLOR_BGR2RGB)

        # Title for the composition winner
        comp_title = str(comp['metadata'][comp_match_index][1].decode('utf-8')) + ' by ' + str(comp['metadata'][comp_match_index][0].decode('utf-8'))

    with h5py.File(data_list[overall_ds], 'r') as ov:
        # Find the overall winner in overall_ds
        overall_match_index = overall_indices[overall_ds]

        # Downloads the image from the url and makes it presentable
        img_overall_url = ov['metadata'][overall_match_index][3].decode('utf-8')
        response = requests.get(img_overall_url)
        image_overall_array = np.array(bytearray(response.content), dtype=np.uint8)
        img_overall_BGR = cv2.imdecode(image_overall_array, cv2.IMREAD_COLOR)
        img_overall = cv2.cvtColor(img_overall_BGR, cv2.COLOR_BGR2RGB)

        # Title for the composition winner
        overall_title = str(ov['metadata'][overall_match_index][1].decode('utf-8')) + ' by ' + str(ov['metadata'][overall_match_index][0].decode('utf-8'))

    # Plot the results
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

    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return img_color, color_title, img_comp, comp_title, img_overall, overall_title

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

# This is the chunk of data being used
df_path_1 = "/Users/greysonmeyer/Downloads/resized_images_chunk_modfied_105.h5"
df_path_2 = "/Users/greysonmeyer/Downloads/resized_images_chunk_modfied_0.h5"
test_image_path = '/Users/greysonmeyer/Downloads/coronati.jpg'
# test_image_path = '/Users/greysonmeyer/Desktop/canal310_color_clustered.png'
# test_image_path = '/Users/greysonmeyer/Downloads/dbcwxcx-05d95715-be49-4177-8579-9bc846ed2ab8.jpg'
test_image = cv2.imread(test_image_path)
test_image_conv = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
# test_image_conv_2 = cv2.cvtColor(test_image_conv, cv2.COLOR_RGB2GRAY)
input_img = resize_and_convert_image(test_image_conv, (200, 200))
display_art(input_img, 0.5, [df_path_1, df_path_2])

# diff = cv2.absdiff(input_img, imag)
# print("Max pixel difference:", diff.max())
# cv2.imshow("Difference", diff)
# cv2.waitKey(0)
# cv2.destroyAllWindows()