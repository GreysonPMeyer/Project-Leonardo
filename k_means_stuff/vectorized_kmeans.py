import pandas as pd
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from PIL import Image
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

def color_similarity_df(input_image, df:pd.DataFrame):
    # Calculates how similar the images from the dataset are to the input image based on the values of the 
    # color clusters

    color_image_clusters = color_columns(input_image)
    distances = df['color_clusters'].apply(lambda row: np.linalg.norm(row - color_image_clusters,2))
    # print('new_norm', np.min(distances))
    # distances = df['color_clusters'].apply(lambda row: cdist(row, color_image_clusters, metric = 'euclidean').min(axis =1).mean())
    # print('new_dist', np.min(distances))
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

def display_art(image:np.array, weight:float, 
                data_file:str = "https://github.com/BotanCevik2/Project-Leonardo/raw/refs/heads/main/resized_images_cluster_fix_2.parquet"):
    """ Find similar art in processed parquet file and display the art.

    Args:
        image (np.array): Input image as a np.array. The input image should be preprocessed 
                          (i.e., color channels switched and resized) prior to being passed to this function.
        weight (float): The weight to be passed to `similar_art` function.
        data_file (str): Path to the processed data stored in a parquet file.

    Returns:
        np.array | str: Returns all of the image matches as image objects and their titles.
    """
    # This code will identify the color match, composition match and overall match across a collection of 
    # data chunks. I have so far only tested it with one chunk

    df = pd.read_parquet(data_file, engine="auto")
    df['color_clusters'] = df['color_clusters'].apply(lambda x: np.array(ast.literal_eval(x)))
    df['composition_clusters'] = df['composition_clusters'].apply(lambda x: np.array(ast.literal_eval(x)))
    df['metadata'] = df['metadata'].apply(lambda x: np.array(ast.literal_eval(x)))
    
    color_winner_idx,_, comp_winner_idx, _, overall_winner_idx, _ = similar_art(image, weight, df)
                    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
                    
    img_color_url = df['metadata'][color_winner_idx][3].decode('utf-8')
    print('color url! ', img_color_url)
    response = requests.get(img_color_url, headers=headers)
    image_color_array = np.array(bytearray(response.content), dtype=np.uint8)
    img_color_BGR = cv2.imdecode(image_color_array, cv2.IMREAD_COLOR)
    img_color = cv2.cvtColor(img_color_BGR, cv2.COLOR_BGR2RGB)

        # Title for the color winner
    color_title = str(df['metadata'][color_winner_idx][1].decode('utf-8')) + ' by ' + str(df['metadata'][color_winner_idx][0].decode('utf-8'))

        # Downloads the image from the url and makes it presentable

    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
                 
# Send a GET request to the URL
# response = requests.get(url, headers=headers)
# image_comp_array = np.array(bytearray(response.content), dtype=np.uint8)
# img_comp_BGR = cv2.imdecode(image_comp_array, cv2.IMREAD_COLOR)
# img_comp = cv2.cvtColor(img_comp_BGR, cv2.COLOR_BGR2RGB)
# plt.imshow(img_comp)
    img_comp_url = df['metadata'][comp_winner_idx][3].decode('utf-8')
    response = requests.get(img_comp_url, headers=headers)
    print('urlurlurl111 ', img_comp_url)
    image_comp_array = np.array(bytearray(response.content), dtype=np.uint8)
    # print('ARRAY ', image_comp_array)
    img_comp_BGR = cv2.imdecode(image_comp_array, cv2.IMREAD_COLOR)
    # print('img_comp_BGR ', img_comp_BGR)
    img_comp = cv2.cvtColor(img_comp_BGR, cv2.COLOR_BGR2RGB)

        # Title for the composition winner
    comp_title = str(df['metadata'][comp_winner_idx][1].decode('utf-8')) + ' by ' + str(df['metadata'][comp_winner_idx][0].decode('utf-8'))

    # Downloads the image from the url and makes it presentable
    img_overall_url = df['metadata'][overall_winner_idx][3].decode('utf-8')
    response = requests.get(img_overall_url, headers=headers)
    image_overall_array = np.array(bytearray(response.content), dtype=np.uint8)
    img_overall_BGR = cv2.imdecode(image_overall_array, cv2.IMREAD_COLOR)
    img_overall = cv2.cvtColor(img_overall_BGR, cv2.COLOR_BGR2RGB)

    # Title for the composition winner
    overall_title = str(df['metadata'][overall_winner_idx][1].decode('utf-8')) + ' by ' + str(df['metadata'][overall_winner_idx][0].decode('utf-8'))

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

if __name__ == "__main__":
    # This is the chunk of data being used
    test_image_path = '/Users/dawsonkinsman/Downloads/output_14_1.png'
    test_image = cv2.imread(test_image_path)
    test_image_conv = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    input_img = resize_and_convert_image(test_image_conv, (200, 200))

    from time import time
    ts = time()

    display_art(input_img, 0.5)

    te = time()
    print(f'{round(te-ts, 2)} sec')
