import pandas as pd
import requests
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py

df_path = "/Users/greysonmeyer/Downloads/resized_images_chunk_0.h5".format(2)


def print_structure(name, obj):
    """Recursive function to print the structure of the HDF5 file."""
    print(f"{name}: {type(obj)}")
    if isinstance(obj, h5py.Group):
        for sub_name, sub_obj in obj.items():
            print_structure(f"{name}/{sub_name}", sub_obj)

def create_test_set(input_file_path):
    output_file_path = "/Users/greysonmeyer/Desktop/Erdos_Work/k_means_stuff/tester.h5"  

    with h5py.File(input_file_path, 'r') as original_file:
        with h5py.File(output_file_path, 'w') as new_file:
            for i in range(1, 20):
                new_file.create_dataset(f"images/{i}", data=original_file[f'images/{i}'], compression="gzip", compression_opts=1)
                new_file.create_dataset(f"metadata/{i}/artist_full_name", data=original_file[f"metadata/{i}/artist_full_name"])
                new_file.create_dataset(f"metadata/{i}/artwork_name", data=original_file[f"metadata/{i}/artwork_name"])
                new_file.create_dataset(f"metadata/{i}/artwork_type", data=original_file[f"metadata/{i}/artwork_type"])

    print(f"Successfully created a copy of the HDF5 file with limited data at '{output_file_path}'.")
    return output_file_path

def color_columns(path):
    # Calculates the color clusters for a dataset and saves this in k+1 news columns in the dataset.
    # The cluster order is determined by the magnitude of the RGB vector in R^3 
    color_dict = dict()
    # display_set = []
    with h5py.File(path, 'a') as df:
        for i in df['images'].keys():
            img = np.array(df[f'images/{i}'])
            img_RGB = img # cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # This has a 200 x 200 x 3 shape
            
            # Reshape image to an Mx3 array
            img_data = img_RGB.reshape(-1, 3)
            # Specify the algorithm's termination criteria
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
            # Run the k-means clustering algorithm on the pixel values
            # labels is a column where each entry contains the center associated to that row's pixel
            # centers is the list of the 5 center colors
            # compactness is just a number. Kind of annoying that it's not a number for each cluster tho
            compactness, labels, centers = cv2.kmeans(data=img_data.astype(np.float32), K=5, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
            centers_sorted = sorted(centers, key=lambda x: np.linalg.norm(x))

            # If there are less than k clusters, add extra white clusters
            if len(centers_sorted) < 5:
                for l in range(5 - len(centers_sorted)):
                    np.append(centers_sorted, np.array([255, 255, 255]))

            color_dict[i] = [compactness, centers_sorted]

        for j in color_dict.keys():
            # df.create_dataset(f"metadata/{j}/color_compactness", data=color_dict[j][0])
            df.create_dataset(f"metadata/{j}/color_clusters", data=color_dict[j][1])

    return path

def composition_columns(path):
    # Calculates the composition clusters and saves this in k+1 new columns in the dataset.
    # The cluster order is determined by the cluster position with the largest y-value, and then we work top
    # to bottom 
    comp_dict = dict()
    # Load image and convert to grayscale
    with h5py.File(path, 'a') as df:
        # what is the actual range for the first h5? How to generalize to the other h5 files?
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

            if i == '2':
                output_image = df['images/2'][:].copy()
                for idx, center in enumerate(centers):
                    cv2.circle(output_image, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
                cv2.imshow("Cluster Centers", output_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


        for j in comp_dict.keys():
            df.create_dataset(f"metadata/{j}/comp_compactness", data=comp_dict[j][0])
            df.create_dataset(f"metadata/{j}/comp_clusters", data=comp_dict[j][1])

    return path

def color_similarity(image, data):
    # Then we calculate the Euclidean distance between the first color_column of the resized image and 
    # the first color column of every image in the dataset. Average the K distances to get an average cluster
    # position difference
    # Perform the weighted average of the average cluster position difference and the difference in color
    # composition
    # Store these values in new columns and return the path to the altered dataset, along with the image & 
    # metadata of the image that minimizes the weighted average.
    color_image_path = color_columns(image)
    color_data_path = color_columns(data) # Comment this out once the data has been pretrained!
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(color_image_path[1][0].astype(np.uint8))
    # axes[1].imshow(color_data_path[1][0].astype(np.uint8))
    # plt.show()
    
    with h5py.File(color_data_path, 'r') as df, h5py.File(color_image_path, 'r') as color:
        # this should be a list of 5 x 3 arrays
        color_data = [
            df[f'metadata/{group}/color_clusters'][:]
            for group in df['metadata']
        ]

        # this should be a 5 x 3 array
        image_clusters = color['metadata/0/color_clusters'][:]

        # each row is a cluster and each column is an image
        distances = np.empty((5,0))
        for datum in color_data:
            cluster_distance = np.linalg.norm(datum - image_clusters, axis=1) # axis = 1 means it's using rows
            distances = np.hstack((distances, cluster_distance[:, np.newaxis]))           

        row_averages = np.mean(distances, axis=0)
        
        winner_image_index = list(df['metadata'].keys())[np.argmin(row_averages)]

    return winner_image_index, row_averages

def composition_similarity(image, data):
    # Calculates average distance between the cluster-positions for associated clusters. It also calculates the
    # differences between the overall cluster densities. It then takes the average of these values and puts this
    # weighted average into a new column of the dataset. It returns the minimizing element of the dataset and 
    # the dataset itself
    comp_image_path = composition_columns(image)
    comp_data_path = composition_columns(data)
    with h5py.File(comp_data_path, 'r') as df, h5py.File(comp_image_path, 'r') as img:

        comp_clusters = [
            df[f'metadata/{group}/comp_clusters'][:]
            for group in df['metadata']
        ]

        comp_compactness = [
            df[f'metadata/{group}/comp_compactness'][()]
            for group in df['metadata']
        ]

        image_clusters = img[f'metadata/0/comp_clusters'][:]
        image_compactness = img[f'metadata/0/comp_compactness'][()]

        output_image = img['images/0'][:].copy()
        for idx, center in enumerate(image_clusters):
            cv2.circle(output_image, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
        cv2.imshow("Cluster Centers", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        comps = np.hstack(comp_compactness) # array with one row of compactness scores 
        comp_distance = np.abs(comps - image_compactness)

        distances = np.empty((5,0))
        for datum in comp_clusters:
            cluster_distance = np.linalg.norm(datum - image_clusters, axis=1)
            distances = np.hstack((distances, cluster_distance[:, np.newaxis])) 

        row_averages = np.mean(distances, axis=0)

        # overall_comp_similarities = 0.5 * np.abs(comp_distance - row_averages)
        winner_image_index = list(df['metadata'].keys())[np.argmin(row_averages)]
        # winner_image_index = list(df['metadata'].keys())[np.argmin(overall_comp_similarities)]

    return winner_image_index, row_averages # , overall_comp_similarities

def similar_art(image, weight, data):
    # Start by resizing the image using Sun's code and put it in an h5file with one image group  
    # Uses the above functions to return the color-similar artwork and the composition-similar artwork. It then
    # takes the datatset with the two new color_similarity & composition_similarity columns and computes the
    # overall_weighted average of these columns. It then returns the image with the smallest overall_weighted 
    # average. So it returns three images and all of the metadata associated to these images
    img = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
    img_file = "/Users/greysonmeyer/Desktop/Erdos_Work/k_means_stuff/input_image.h5"
    with h5py.File(img_file, 'w') as file:
        file.create_dataset("images/0", data=img, compression="gzip", compression_opts=1)
        file.create_group("metadata/0") 

    color_match_index, color_averages = color_similarity(img_file, data)
    comp_match_index, comp_averages = composition_similarity(img_file, data)
    overall_avgs = weight * color_averages + (1 - weight) * comp_averages

    with h5py.File(data, 'r') as df:
        overall_match_index = list(df['metadata'].keys())[np.argmin(overall_avgs)]
        # overall_match_index = starting_group_num + np.argmin(overall_avgs)
        img_color = df[f'images/{color_match_index}'][:]
        img_comp = df[f'images/{comp_match_index}'][:]
        img_overall = df[f'images/{overall_match_index}'][:]

        color_title = str(df[f'metadata/{color_match_index}/artwork_name'][()].decode('utf-8')) + ' by ' + str(df[f'metadata/{color_match_index}/artist_full_name'][()].decode('utf-8'))

        comp_title = df[f'metadata/{comp_match_index}/artwork_name'][()].decode('utf-8') + ' by ' + df[f'metadata/{comp_match_index}/artist_full_name'][()].decode('utf-8')

        overall_title = df[f'metadata/{overall_match_index}/artwork_name'][()].decode('utf-8') + ' by ' + df[f'metadata/{overall_match_index}/artist_full_name'][()].decode('utf-8')

        # img_color = mpimg.imread('path/to/image1.jpg')  # Replace with your image paths
        # img2 = mpimg.imread('path/to/image2.jpg')  # Replace with your image paths
        # img3 = mpimg.imread('path/to/image3.jpg')  # Replace with your image paths

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

    return 'color match: ', color_title, '; composition match: ', comp_title, '; overall match: ', overall_title
    
test_path = create_test_set(df_path)

test_image_path = '/Users/greysonmeyer/downloads/saved.png'
test_image = cv2.imread(test_image_path)
# test_image = cv2.cvtColor(test_image_wrong, cv2.COLOR_BGR2RGB)
similar_art(test_image, 0.5, test_path)