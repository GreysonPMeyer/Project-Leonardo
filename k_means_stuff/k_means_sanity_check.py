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
# import helper_tools as ht
import logging
from skimage.metrics import structural_similarity as compare_ssim

def create_test_set(input_file_path):
    # Creates an h5file from a chunk of the dataset, for the sake of time for testing
    output_file_path = "./scrap/gridsearch/resized_images_chunk_modfied_53.h5"  

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

    sorted_labels = np.zeros_like(labels)
    for new_label, old_label in enumerate(sorted_indices):
        sorted_labels[labels == old_label] = new_label

    # If there are less than k clusters, add extra white clusters
    if len(centers_sorted) < 4:
        first_center = centers_sorted[0]
        for l in range(4 - len(centers_sorted)):
            centers_sorted = np.concatenate(centers_sorted, np.array(first_center))

    return [np.float64(centers_sorted), centers_sorted, sorted_labels]

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
    if len(contour_centers) == 0:
        # This case exists and is annoying, so I made all of the clusters at the origin
        sorted_centers = np.array([[0,0], [0,0], [0,0], [0,0]])
    elif len(contour_centers) == 1:
        # If you only have one contour center, then kmeans no longer returns tuples
        sorted_centers = np.concatenate((contour_centers,np.array([contour_centers[0], contour_centers[0], contour_centers[0]])))
    elif 1 < len(contour_centers) < 4:
        # hdf5files struggle to contain informatio that is not of a uniform size, so we add copies of the origin
        K = len(contour_centers)
        cv2.setRNGSeed(42)
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
        first_s_center = sorted_centers[0]
        for _ in range(4 - len(sorted_centers)):
            sorted_centers = np.concatenate((sorted_centers,first_s_center))
    else:
        K = 4
        cv2.setRNGSeed(42)
        compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        sorted_centers = np.array(sorted(centers, key=lambda c: (c[1], c[0]), reverse=True))
    
    if sorted_centers.shape != (4,2):
        print(sorted_centers.shape)

    return np.float64(sorted_centers)

# @ht.timing
# def write_final_parquet(chunk_dir_path:str, output_path:str)-> None:
#     """Walks through chunk directory, reads in the chunks and processes each image, and saves the
#     color and composition centroids with metadata in parquet format.

#     Args:
#         chunk_dir_path (str): Directory containing the image chunks.
#         output_path (str): Path to save final dataset (must end in .parquet.gzip).
#     """
#     logging.basicConfig(filename='final_processing.log', encoding='utf-8', 
#                         format="%(asctime)s:%(levelname)s:%(message)s",level=logging.DEBUG)
#     results_dict = {}
#     idx = 0
#     for test, file in enumerate(os.listdir(chunk_dir_path)):
#         if test>0:continue
#         chunk_dict = ht.h5_to_dict(os.path.join(chunk_dir_path, file))
#         for meta, img in chunk_dict.items():
#             try:
#                 color_centers = color_columns(img) 
#                 comp_centers = composition_columns(img[0])
#                 artist, img_name, img_type, img_url = meta

#                 results_dict[idx] = [artist, img_name, img_type, img_url, color_centers, comp_centers]
#                 idx +=1
#             except Exception as e:
#                 logging.debug(f'{meta}: {e}')
#                 # print(meta)
        
#     results_df = pd.DataFrame.from_dict(results_dict,orient='index', 
#                                         columns=['artist_name', 'img_name','img_type', 'img_url',
#                                                  'color_centers', 'comp_centers'])
#     # ['Artist', 'Image Name', 'Image Type', 'Image URL']
#     # results_df['artist'] = results_df.iloc[:,0]
#     results_df.astype(str).to_parquet(output_path, compression='gzip')
#     # results_df.to_hdf('./test4.h5', 'results_df', format='table', mode='w')
#     # results_df.to_hdf('./test2.h5', key = 'df', mode = 'w')
#     # results_df.to_csv('./test2.csv')
#     return

def color_similarity_df(input_image, color_centers_df:pd.DataFrame):
    # Calculates how similar the images from the dataset are to the input image based on the values of the 
    # color clusters

    color_image_clusters = color_columns(input_image)[0]
    
    # with h5py.File(data, 'r') as df:
        # This is a list of 5 x 3 arrays
        # color_data = df['color_clusters'][:]

        # Each row is a cluster and each column is an image
    # distances = []
        # for datum in color_data:
            # Compare the Euclidean distance between each array in color_data and the image_clusters
    # for col in color_centers_df.columns:
    #     col_series = color_centers_df[col].str.split().apply(lambda x: np.array(x))
    #     col_array = np.array.from_list(col_series, dtype=np.float64)
    #     cluster_distance = cdist(color_centers_df[col], color_image_clusters, metric='euclidean')
        # Calculate the minimum distance for each centroid in datum to any centroid in image_clusters
    distances = color_centers_df.apply(lambda row: np.linalg.norm(row - color_image_clusters,2))
    min_distances = cluster_distance.min(axis=1)
    color_similarity_score = np.mean(min_distances)
        # distances.append(color_similarity_score)           
            
            # The index of the image with the smallest mean centroid distance
    color_winner_index = np.argmin(distances)
            # We'll use distances again for the overall comparison. We need it to be an array for the calculations
    distance_vector = np.array(distances)

    return color_winner_index, distance_vector


def color_similarity(image, data):
    # Calculates how similar the images from the dataset are to the input image based on the values of the 
    # color clusters

    col = color_columns(image)
    color_image_clusters = col[0]
    col_clust = col[1]
    col_lab = col[2]
    
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

        print('color index: ', color_winner_index)

    return color_winner_index, distance_vector, col_clust, col_lab

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

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(df['images'][325])
        axes[0].set_title('Image 325 in dataset')
        clust = df['composition_clusters'][325]
        x_coords, y_coords = clust[:, 0], clust[:, 1]
        axes[0].scatter(x_coords, y_coords, c='red', s=100, edgecolor='black', label='Centroids')

        axes[1].imshow(image)
        axes[1].set_title('Input image')
        x_coords, y_coords = comp_image_clusters[:, 0], comp_image_clusters[:, 1]
        axes[1].scatter(x_coords, y_coords, c='red', s=100, edgecolor='black', label='Centroids')

        axes[2].imshow(df['images'][winner_image_index])
        axes[2].set_title('Winning composition image')
        clust = df['composition_clusters'][winner_image_index]
        x_coords, y_coords = clust[:, 0], clust[:, 1]
        axes[2].scatter(x_coords, y_coords, c='red', s=100, edgecolor='black', label='Centroids')

        plt.show()

        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(df['images'][325], cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        score, diff = compare_ssim(gray1, gray2, full=True)
        print("SSIM:", score)

        # Normalize the difference for visualization
        diff = (diff * 255).astype("uint8")
        cv2.imshow("SSIM Difference", diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if np.array_equal(image, df['images'][325]):
            print('yupyup')

        # If you would like to see the clusters for the input image, use this code
        # plt.imshow(image)
        # x_coords, y_coords = comp_image_clusters[:, 0], comp_image_clusters[:, 1]
        # plt.scatter(x_coords, y_coords, c='red', s=100, edgecolor='black', label='Centroids')
        # plt.show()
        print('comp index: ', winner_image_index)

    return winner_image_index, comp_dist_vector

def similar_art(image, weight, data):
    # Calculates the overall match and consolidates all of the information
    color_match_index, color_averages, col_clust, col_lab = color_similarity(image, data)

    # c = col_clust[col_lab].reshape(-1,3)
    # recolored_img = c.reshape(image)

    color_winning_avg = color_averages[color_match_index]
    comp_match_index, comp_averages = composition_similarity(image, data)
    comp_winning_avg = comp_averages[comp_match_index]
    overall_avgs = weight * color_averages + (1 - weight) * comp_averages
    overall_match_index = np.argmin(overall_avgs)
    overall_winning_avg = overall_avgs[overall_match_index]

    return [color_match_index, color_winning_avg, comp_match_index, comp_winning_avg, overall_match_index, overall_winning_avg]#, recolored_img]
    
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
        ccenters = color_columns(color['images'][color_match_index])[1]
        clabels = color_columns(color['images'][color_match_index])[2]
        colours = ccenters[clabels].reshape(-1, 3)
        recolored_winner = colours.reshape(color['images'][color_match_index].shape)

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

        print('url: ', img_comp_url)

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
    
    return img_color, color_title, img_comp, comp_title, img_overall, overall_title, recolored_winner #, similar_art[-1]

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

df_path_1 = "/Users/greysonmeyer/Downloads/resized_images_chunk_modfied_105.h5"
df_path_2 = "/Users/greysonmeyer/Downloads/resized_images_chunk_modfied_0.h5"
test_image_path = '/Users/greysonmeyer/Downloads/strangled-woman-1872.jpg'
# test_image_path = '/Users/greysonmeyer/Desktop/canal310_color_clustered.png'
# test_image_path = '/Users/greysonmeyer/Downloads/dbcwxcx-05d95715-be49-4177-8579-9bc846ed2ab8.jpg'
test_image = cv2.imread(test_image_path)
test_image_conv = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
# test_image_conv_2 = cv2.cvtColor(test_image_conv, cv2.COLOR_RGB2GRAY)
input_img = resize_and_convert_image(test_image_conv, (200, 200))
display_art(input_img, 0.5, [df_path_1, df_path_2])

# This is the chunk of data being used
# df_path = "./scrap/gridsearch/resized_images_chunk_modfied_53.h5"
# dir = fr'./scrap/gridsearch/'
# test_path = create_test_set(df_path)
# test_image_path = '/Users/greysonmeyer/Downloads/canal310.jpg'
# test_image = cv2.imread(test_image_path)
# test_image_conv = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
# input_img = resize_and_convert_image(test_image_conv, (200, 200))
# display_art(input_img, 0.5, [test_path])

# write_final_parquet(dir, './test4.parquet.gzip')
# # from time import time
# # ts = time()

# df = pd.read_parquet('./test4.parquet.gzip')
# color_centers = df['color_centers'].str.replace('[', '').str.replace(']', '').str.replace('\n', '').apply(lambda x: np.fromstring(x, sep = ' ').reshape(4,3))
# comp_centers = df['comp_centers'].str.replace('[', '').str.replace(']', '').str.replace('\n', '').apply(lambda x: np.fromstring(x, sep = ' ').reshape(4,3))
# test_img = cv2.cvtColor(cv2.imread('./scrap/validation/test_images/test1.jpg'), cv2.COLOR_BGR2RGB)
# # x,y =  color_similarity_df(test_img, color_centers)
# pass
# te = time()

# print(te-ts)
