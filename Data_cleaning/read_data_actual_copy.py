import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

df0 = pd.read_csv('/Users/greysonmeyer/Desktop/Erdos_Work/omniart_v3_datadump.csv')

# get rid of the ones with missing artist name
df1 = df0.dropna(subset=['artist_full_name', 'artist_first_name', 'artist_last_name'])

# get rid of all arkyve links since they all seemed to be broken when I was doing initial analysis
df2 = df1[~df1['image_url'].str.contains('arkyve', case=False, na=False)]

# Only keep the painting & painting-adjacent entries
df = df2[df2['artwork_type'].isin(['painting', 'graphics', 'drawings', 'landscape', 'genre painting', 
                                  'print', 'abstract', 'history painting', 'cityscape', 'portrait', 
                                  'symbolic painting', 'self-portrait', 'sketch and study', 
                                  'animal painting', 'nude painting (nu)', 'flower painting', 
                                  'mythological painting', 'literary painting', 'calligraphy', 
                                  'illustration', 'religious painting', 'caricature', 
                                  'bird and flower painting', 'tesselation', 'still life', 'graffiti', 
                                  'allegorical painting', 'glass-painted', 'manuscripts and illuminations', 
                                  'wallpaper', 'posters', 'wildlife painting', 'enamels-painted', 'advertisement', 
                                  'tapestry', 'costumes-tapestries', 'ink tablet', 'woodblocks', 'miscellaneous-paper', 
                                  'panorama', 'pastorale', 'painted canvases', 'miscellaneous-stucco', 
                                  'stucco-reliefs-inscribed', 'miscellaneous-mosaic', 'reproductions-mosaics', 
                                  'pastels & oil sketches on paper', 'sketchbooks', 'works on paper-miscellaneous', 
                                  'paper-graphics', 'miscellaneous-papyrus', 'graphic design', 'drawings|miscellaneous', 
                                  ])]

def check_for_image(url):
    if 'moma' in url:
        return True
    try:
        # Make a GET request to the URL
        response = requests.get(url, timeout=10)  # Adding a timeout for the request
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        content_type = response.headers.get('Content-Type')
        if not content_type:
            print('No Content-Type, False', url)
            return False
        else:
        # Check if the response is successful and if the content type is 'image/jpeg'
            print(not 'text/html' in response.headers.get('Content-Type', ''), url)
            return not 'text/html' in response.headers.get('Content-Type', '')
    except requests.RequestException as e:
        print('False ', url)
        return False  # Return False for any request-related error
    except Exception as e:
        print('False ', url)
        return False  # Return False for any other errors

# I got a couple weird errors when I didn't have this wrapper function, but idk if it's even really doing 
# anything and I'm too scared to delete it
def safe_check_for_image(url):
    """ Wrapper function to catch parsing errors """
    try:
        return check_for_image(url)
    except Exception as parse_error:
        print(f"Parsing error for URL {url}: {parse_error}")
        return True  # Return True in case of any parsing error so that we can check them by hand later.
                    # No exception ever occurred!

# Use ThreadPoolExecutor for concurrent requests
with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
    results = list(executor.map(check_for_image, df2['image_url']))

# Assign the results to the 'image_available' column
df['image_available'] = results

# Keep only the rows where the image is available
df_cleaned = df[df['image_available'] == True]

# Drop the temporary 'image_available' column
df_cleaned = df_cleaned.drop(columns='image_available')

# df_cleaned.to_csv('/Users/greysonmeyer/Desktop/Erdos_Work/omni_clean.csv', index=False)