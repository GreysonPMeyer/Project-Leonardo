import pandas as pd
import cv2
import numpy as np
import random
import helper_tools as ht
from sklearn.metrics import silhouette_score

@ht.timing
def color_columns(h5_dict:dict, k:int, num_sample_img:int = 10):
    # Calculates the color clusters for a dataset and saves this in k+1 news columns in the dataset.
    # The cluster order is determined by the magnitude of the RGB vector in R^3 
    # color_dict = dict()
    results_dict = {'compactness':[], 
                    'metadata':[],
                    'clusters':[k]*num_sample_img, 
                    'sil_score':[]}

    random.seed(369)
    sample_keys = random.sample(sorted(h5_dict.keys()), num_sample_img)
    for meta in sample_keys:
        # print(meta)
        img = h5_dict.get(meta).reshape((200,200,3))
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Reshape image to an Mx3 array
        img_data = img_RGB.reshape(-1, 3)
        # Specify the algorithm's termination criteria
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
        # Run the k-means clustering algorithm on the pixel values
        # labels is a column where each entry contains the center associated to that row's pixel
        # centers is the list of the 5 center colors
        # compactness is just a number. Kind of annoying that it't not a number for each cluster tho
        compactness, labels, centers = cv2.kmeans(data=img_data.astype(np.float32), K=k, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
        # centers_sorted = sorted(centers, key=lambda x: np.linalg.norm(x))
        sscore = silhouette_score(img_data, labels.ravel())
        results_dict['compactness'].append(compactness)
        results_dict['metadata'].append(meta)
        results_dict['sil_score'].append(sscore)
        # If there are less than k clusters, add extra white clusters
        # if len(centers_sorted) < 5:
        #     for l in range(5 - len(centers_sorted)):
        #         centers_sorted.append(np.array([255, 255, 255]))

        # color_dict[i] = [compactness, centers_sorted]

    # for j in color_dict.keys():
    #     # df.create_dataset(f"metadata/{j}/color_compactness", data=color_dict[j][0])
    #     df.create_dataset(f"metadata/{j}/color_clusters", data=color_dict[j][1])

        # print("Should see color clusters!")
    return results_dict

@ht.timing
def composition_columns(h5_dict:dict, k:int, num_sample_img:int = 10):
    # Calculates the composition clusters and saves this in k+1 new columns in the dataset.
    # The cluster order is determined by the cluster position with the largest y-value, and then we work top
    # to bottom 

    # Load image and convert to grayscale
    results_dict = {'compactness':[], 
                    'metadata':[],
                    'clusters':[k]*num_sample_img, 
                    'sil_score':[]}

    random.seed(369)
    sample_keys = random.sample(sorted(h5_dict.keys()), num_sample_img)
        # what is the actual range for the first h5? How to generalize to the other h5 files?
    for meta in sample_keys:
        # print(meta)
        img = h5_dict.get(meta).reshape((200,200,3))
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
        # if len(contour_centers) < 5:
        #     K = len(contour_centers)
        # else:
        #     K = 5  # Choose number of clusters for forms
        if contour_centers.shape[0] < k:
            print(f'{meta} has less contours than number of clusters ({k}).')
            compactness = np.NaN
            sscore = np.NaN
        else:
            compactness, labels, centers = cv2.kmeans(contour_centers, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
            try:
                sscore = silhouette_score(contour_centers, labels.ravel())
            except:
                print(f'{meta} composition k-means results in clusters containing one point.')
                sscore = np.NaN
        results_dict['compactness'].append(compactness)
        results_dict['metadata'].append(meta)
        results_dict['sil_score'].append(sscore)

    # print("Should see color_clusters, comp_compactness & comp_clusters!")
    return results_dict

@ht.timing
def generate_validation_dfs(sample_file_list:list, k_range:list, num_sample_img:int = 10, save_files:bool = True) -> pd.DataFrame:
    color_results_list, comp_results_list = [], []

    for file in sample_file_list:
        path = fr'./scrap/gridsearch/resized_images_chunk_modfied_{file}.h5'
        file_dict = ht.h5_to_dict(path)
        for k in k_range:
            color_results = color_columns(file_dict, k = k, num_sample_img=num_sample_img)  
            color_results_list.append(pd.DataFrame(color_results))

            comp_results = composition_columns(file_dict, k = k, num_sample_img=num_sample_img)
            comp_results_list.append(pd.DataFrame(comp_results))

    color_results_df = pd.concat(color_results_list, ignore_index=True)
    comp_results_df = pd.concat(comp_results_list, ignore_index=True)

    if save_files:
        color_results_df.to_csv('./color_validation.csv')
        comp_results_df.to_csv('./comp_validation.csv')

    return color_results_df, comp_results_df


if __name__ == "__main__":

    num_files = 72
    random.seed(12345)
    sample_files = random.sample(range(72), 10)
    # Download by hand these h5 files from OneDrive
    print(sample_files)

    color_df, comp_df = generate_validation_dfs(sample_files, range(2,6))
    max_idx = color_df.groupby('metadata')['sil_score'].abs().idxmax()
    color_max = color_df.loc[max_idx, ['metadata', 'clusters', 'sil_score']]
    color_max.to_csv('./color_max.csv')

    max_idx = comp_df.groupby('metadata')['sil_score'].abs().idxmax()
    comp_max = comp_df.loc[max_idx, ['metadata', 'clusters', 'sil_score']]
    comp_max.to_csv('./comp_max.csv')
    pass