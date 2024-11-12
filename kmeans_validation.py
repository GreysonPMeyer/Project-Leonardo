import pandas as pd
import cv2
import numpy as np
import random
import helper_tools as ht
from sklearn.metrics import silhouette_score
import time
import os
import pickle

SKIP_IMAGE_LIST = []

logging = open('./kmeans_validation.log', 'a')

@ht.timing
def color_columns(h5_dict:dict, k:int, num_sample_img:int = 10):
    # Calculates the color clusters for a dataset and saves this in k+1 news columns in the dataset.
    # The cluster order is determined by the magnitude of the RGB vector in R^3 
    # color_dict = dict()
    results_dict = {'compactness':[], 
                    'metadata':[],
                    'clusters':[], 
                    'sil_score':[]}

    random.seed(369)
    sample_keys = random.sample(sorted(h5_dict.keys()), num_sample_img)
    for meta in sample_keys:
        if meta in SKIP_IMAGE_LIST: continue
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
        results_dict['clusters'].append(k)

    return results_dict

@ht.timing
def composition_columns(h5_dict:dict, k:int, num_sample_img:int = 10):
    # Calculates the composition clusters and saves this in k+1 new columns in the dataset.
    # The cluster order is determined by the cluster position with the largest y-value, and then we work top
    # to bottom 

    # Load image and convert to grayscale
    results_dict = {'compactness':[], 
                    'metadata':[],
                    'clusters':[], 
                    'sil_score':[]}

    random.seed(369)
    sample_keys = random.sample(sorted(h5_dict.keys()), num_sample_img)
        # what is the actual range for the first h5? How to generalize to the other h5 files?
    for meta in sample_keys:
        if meta in SKIP_IMAGE_LIST: continue
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
        if len(contour_centers) < k:
            K = len(contour_centers) - 1
            if K <= 0:
                SKIP_IMAGE_LIST.append(meta)
            else:
                logging.write(f'{meta} has {len(contour_centers)} contours so k = {K}.\n')
                compactness, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                SKIP_IMAGE_LIST.append(meta)
        else:
            compactness, labels, centers = cv2.kmeans(contour_centers, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # sorted_centers = sorted(centers, key=lambda c: (c[1], c[0]), reverse=True)
        try:
            sscore = silhouette_score(contour_centers, labels.ravel())
        except:
            logging.write(f'{meta} composition k-means results in clusters containing one point.\n')
            sscore = np.NaN
        results_dict['compactness'].append(compactness)
        results_dict['metadata'].append(meta)
        results_dict['sil_score'].append(sscore)
        results_dict['clusters'].append(k)

    # print("Should see color_clusters, comp_compactness & comp_clusters!")
    return results_dict

@ht.timing
def generate_validation_dfs(sample_file_list:list, k_range:list, num_sample_img:int = 10, save_files:bool = True, cache:bool = True) -> pd.DataFrame:
    color_results_list, comp_results_list = [], []

    for file in sample_file_list:
        path = fr'./scrap/gridsearch/resized_images_chunk_modfied_{file}.h5'
        file_dict = ht.h5_to_dict(path)
        for k in k_range:
            if (os.path.isfile(fr'./scrap/cache/color_results_{file}_{k}.pickle')) and (os.path.isfile(fr'./scrap/cache/comp_results_{file}_{k}.pickle')):
                with open(fr'./scrap/cache/color_results_{file}_{k}.pickle', 'wb') as handle:
                    comp_results = pickle.load(handle)
                with open(fr'./scrap/cache/comp_results_{file}_{k}.pickle', 'wb') as handle:
                    comp_results = pickle.load(handle)
            else:    
                color_results = color_columns(file_dict, k = k, num_sample_img=num_sample_img)  
                comp_results = composition_columns(file_dict, k = k, num_sample_img=num_sample_img)
            
            logging.write(f'File: {file} K = {k} has processed/been read in.')
            color_results_list.append(pd.DataFrame(color_results))
            comp_results_list.append(pd.DataFrame(comp_results))
            
            if cache:
                with open(fr'./scrap/cache/color_results_{file}_{k}.pickle', 'wb') as handle:
                    pickle.dump(color_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(fr'./scrap/cache/comp_results_{file}_{k}.pickle', 'wb') as handle:
                    pickle.dump(comp_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logging.write(f'File: {file} K = {k} has processed and written out.')

    color_results_df = pd.concat(color_results_list, ignore_index=True)
    comp_results_df = pd.concat(comp_results_list, ignore_index=True)

    if save_files:
        color_results_df.to_csv('./color_validation.csv', index=False)
        comp_results_df.to_csv('./comp_validation.csv', index=False)

    return color_results_df, comp_results_df


if __name__ == "__main__":

    k_range = range(2,10)

    if not (os.path.isfile('./color_validation.csv')) and not (os.path.isfile('./comp_validation.csv')):
        num_files = 72
        random.seed(12345)
        sample_files = random.sample(range(72), 10)
        # Download by hand these h5 files from OneDrive
        print(sample_files)

        ts = time.time()
        color_df, comp_df = generate_validation_dfs(sample_files, k_range, num_sample_img=10)
        te = time.time()
        print(f'Generating validation datasets took {(ts-te)/60}min.')

    else:
        comp_df = pd.read_csv('./comp_validation.csv')
        color_df = pd.read_csv('./color_validation.csv')
        drop_col = 'Unnamed: 0'
        if drop_col in color_df.columns:
            color_df = color_df.drop(drop_col, axis=1)
        if drop_col in comp_df.columns:
            comp_df = comp_df.drop(drop_col, axis =1)

    print(f'# of NaNs in color df \n{color_df.isna().sum()}.')
    print(f'# of NaNs in comp df \n{comp_df.isna().sum()}.')

    print(comp_df[comp_df['sil_score'].isna()])

    for obj in comp_df.loc[comp_df['sil_score'].isna(), 'metadata'].unique():
        if len(comp_df[comp_df['metadata']== obj]) == len(k_range):
            comp_df = comp_df[comp_df['metadata']!= obj]
            comp_df = pd.concat([comp_df, pd.DataFrame({'compactness':np.NaN,
                                                        'metadata':obj,
                                                        'clusters':0,
                                                        'sil_score': np.inf}, index=[0])])

    # Drop NaNs, since these instances can't be compared
    comp_df.dropna(inplace=True, subset='sil_score')
    color_df.dropna(inplace=True, subset='sil_score')

    max_idx = color_df.groupby('metadata')['sil_score'].idxmax()
    color_max = color_df.loc[max_idx, ['metadata', 'clusters', 'sil_score']]
    color_max.to_csv('./color_max.csv')

    max_idx = comp_df.groupby('metadata')['sil_score'].idxmax()
    comp_max = comp_df.loc[max_idx, ['metadata', 'clusters', 'sil_score']]
    comp_max.to_csv('./comp_max.csv', index = True)


    pass