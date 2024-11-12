import h5py
from functools import wraps
from time import time
import numpy as np
import pandas as pd
# import numpy as np


H5_FILE = './data/resized_images_chunk_0.h5'

def h5_to_dict_OLD(filepath:str= H5_FILE)-> dict:
    """Read in H5 files to dictionary (old version)

    Args:
        filepath (str, optional): Local file path to H5 file. Defaults to H5_FILE.

    Returns:
        dict: Dictionary with keys containing metadata and values being images.
    """
    h5_file = h5py.File(filepath)
    img_arrays = [h5_file['images'][doc][:] for doc in h5_file['images']]

    img_artist, img_name, img_type = [],[],[]
    for i in range(len(h5_file['metadata'].keys())):
        img_artist.append(h5_file[f'metadata/{i}/artist_full_name'][()].decode('utf-8'))
        img_name.append(h5_file[f'metadata/{i}/artwork_name'][()].decode('utf-8'))
        img_type.append(h5_file[f'metadata/{i}/artwork_type'][()].decode('utf-8'))
    img_dict = dict(zip(zip(img_type, img_artist, img_name), img_arrays))
    
    # NEED TO HANDLE ARTWORKS WITH DUPLICATE KEYS 
    # print(len(img_arrays), len(img_artist), len(img_name), len(img_type), len(img_dict))

    return img_dict

def timing(f):
    """
    Timing Decorator to time functions. To use simply import helper_tools to .py file and access using @helper_tools.timing
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r  took: %2.4f sec' % \
          (f.__name__, te-ts)) #args:[%r, %r]
        return result
    return wrap

def h5_to_dict(filepath:str) -> dict:
    """Read in H5 files to dictionary (new version)

    Args:
        filepath (str, optional): Local file path to H5 file. Defaults to H5_FILE.

    Returns:
        dict: Dictionary with keys containing metadata and values being images.
    """
    h5_file = h5py.File(filepath)
    lg_array = np.array(h5_file['images'])
    img_list = np.split(lg_array, lg_array.shape[0])

    lg_array = np.array(h5_file['metadata'])
    meta_list = np.split(lg_array, lg_array.shape[0])
    meta_list_keys = [tuple([i[0,0].decode('utf-8'), i[0,1].decode('utf-8'), i[0,2].decode('utf-8'), i[0,3].decode('utf-8')]) 
                      for i in meta_list]
    img_dict = dict(zip(meta_list_keys, img_list))

    return img_dict

def h5_to_pandas_metadata(filepath:str)->pd.DataFrame:
    h5_file = h5py.File(filepath)
    df = pd.DataFrame(h5_file['metadata'], columns = ['Artist', 'Image Name', 'Image Type', 'Image URL'])
    for col in df.select_dtypes('object'):
        df[col] = df[col].str.decode("utf-8")
    return df

if __name__ == "__main__":
    path = r'scrap/gridsearch/resized_images_chunk_modfied_68.h5'
    test = h5_to_pandas_metadata(path)
    pass