import pandas as pd
import requests
import os
import cv2
import numpy as np
from matplotlib.pyplot import show, imshow

# df = pd.read_csv('/Users/greysonmeyer/Desktop/Erdos_Work/omni_data_cleaned_copy_2.csv')
# url = df['image_url'][4]

def download_image(url, i):
    # Send a GET request to the URL
    response = requests.get(url)

    save_path = os.path.join('/Users/greysonmeyer/Desktop/Erdos_Work/', 'try{i}.jpg')
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the file in binary-write mode and save the image
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image successfully downloaded to {save_path}")
        return save_path
    else:
        print(f"Failed to download image. Status code: {response.status_code}")


# def get_k_means(img):
#     # Convert it from BGR to RGB
#     img_RGB = cvtColor(img, COLOR_BGR2RGB)
    
# COMPOSITION PORTION

# Load image and convert to grayscale
image = cv2.imread('/Users/greysonmeyer/Desktop/Erdos_Work/try2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection (Canny) to find contours
edges = cv2.Canny(gray, 100, 200)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
K = 5  # Choose number of clusters for forms
_, labels, centers = cv2.kmeans(contour_centers, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Output center coordinates of each form cluster
for idx, center in enumerate(centers):
    print(f"Cluster {idx} center: (x={int(center[0])}, y={int(center[1])})")

# Optional: Visualize clusters
output_image = image.copy()
for idx, center in enumerate(centers):
    cv2.circle(output_image, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
cv2.imshow("Cluster Centers", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# press escape to exit the window!

# COLOR PORTION

# img = cv2.imread('/Users/greysonmeyer/Desktop/Erdos_Work/try2.jpg')

# img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# # Reshape image to an Mx3 array
# img_data = img_RGB.reshape(-1, 3)
 
# # Find the number of unique RGB values. axis = 0 means that it's checking unique rows
# print(len(np.unique(img_data, axis=0)), 'unique RGB values out of', img_data.shape[0], 'pixels')
 
# # Specify the algorithm's termination criteria
# criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
 
# # Run the k-means clustering algorithm on the pixel values
# # labels is a column where each entry contains the center associated to that row's pixel
# # centers is the list of the 5 center colors
# # compactness is just a number. Kind of annoying that it't not a number for each cluster tho
# compactness, labels, centers = cv2.kmeans(data=img_data.astype(np.float32), K=5, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

# # Apply the RGB values of the cluster centers to all pixel labels
# colours = centers[labels].reshape(-1, 3)
 
# # Find the number of unique RGB values
# print(len(np.unique(colours, axis=0)), 'unique RGB values out of', img_data.shape[0], 'pixels')
 
# # Reshape array to the original image shape
# img_colours = colours.reshape(img_RGB.shape)
 
# # Display the quantized image
# imshow(img_colours.astype(np.uint8))
# show()

# for j in range(5):
#     url_j = df['image_url'][4 + j]
#     img_j = download_image(url_j, j)