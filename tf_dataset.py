import h5py
import tensorflow as tf
import cv2
import logging
import numpy as np
from PIL import Image

def logging_turnon():
    """Output debugging info to logfile."""
    logging.basicConfig(
        filename="log.log",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.DEBUG,
    )
    with open("log.log", "w") as file:
        file.write("")


def tf_dataset(file_path: str):
    """Generate Tensorflow Dataset from file path.
    Shuffles records prior to dividing into batches.

    Args:
        file_path (str): Path of HDF file to read in.
        batch_size (int): Number of images in a batch

    Returns:
        _type_: Generator containing Tensorflow batches.
    """
    count = sum(1 for _ in create_generator(file_path))
    return tf.data.Dataset.from_generator(
        create_generator,
        args=[file_path],
        output_signature=(
            {
                "artist": tf.TensorSpec(shape=(), dtype=tf.string),
                "img_name": tf.TensorSpec(shape=(), dtype=tf.string),
                "img_type": tf.TensorSpec(shape=(), dtype=tf.string),
            },
            tf.TensorSpec(shape=(200, 200, 3), dtype=tf.float32),
        ),
    ).shuffle(count)


def create_generator(file_path: str):
    """Create generator from h5 file path.

    Args:
        file_path (str): Path to h5 file.

    Yields:
        _type_: Generator containing metadata and image data.
    """
    with h5py.File(file_path, "r") as hdf5_file:
        for idx in hdf5_file["images"]:
            image_data = hdf5_file["images"][idx][:]
            # Extract metadata
            artist = hdf5_file["metadata"][idx]["artist_full_name"][()].decode("utf-8")
            img_name = hdf5_file["metadata"][idx]["artwork_name"][()].decode("utf-8")
            img_type = hdf5_file["metadata"][idx]["artwork_type"][()].decode("utf-8")
            metadata_dict = {
                "artist": artist,
                "img_name": img_name,
                "img_type": img_type,
            }
            
            # There is an error that gets thrown if the image data is not (200, 200, 3)
            # and there appear to be some.
            # e.g. for the first h5 file: 955.
            if image_data.shape == (200, 200, 3):
                yield metadata_dict, image_data
            else:
                with open("log.log", "w") as file:
                    file.write(idx)
                continue

# Currently not working....
def composition_columns(meta_dict, img_data):
    # Calculates the composition clusters and saves this in k+1 new columns in the dataset.
    # The cluster order is determined by the cluster position with the largest y-value, and then we work top
    # to bottom 
    img_data = img_data.numpy()#.decode("utf-8")
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)

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
    meta_dict['composition_compatctness'] = compactness
    meta_dict['sorted_centers'] = sorted_centers

    return meta_dict, img_data

def parse_func(meta_dict, img):
    out = tf.py_function(composition_columns, [list(meta_dict.items()), img], tf.uint8)
    return out

if __name__ == "__main__":

    logging_turnon()
    path = "./data/resized_images_chunk_0.h5"
    gen = create_generator(path)

    test = tf_dataset(path)
    test2 = test.map(parse_func, num_parallel_calls=10)
    for i in test2: 
        print(i)
