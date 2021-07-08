import os
import pathlib
import nibabel as nib
import numpy as np
import pandas as pd
from provided_code.network_architectures import DefineMnet


def load_file(file_name):
    """Load a file in one of the formats provided in the OpenKBP dataset
    :param file_name: the name of the file to be loaded
    :return: the file loaded
    """
    
    loaded_file_data = nib.load(file_name).get_data()
    affine = nib.load(file_name).affine
    return loaded_file_data, affine


def get_paths(directory_path, ext=''):
    """Get the paths of every file with a specified extension in a directory
    :param directory_path: the path of the directory of interest
    :param ext: the extensions of the files of interest
    :return: the path of all files of interest
    """
    # if dir_name doesn't exist return an empty array
    if not os.path.isdir(directory_path):
        return []
    # Otherwise dir_name exists and function returns contents name(s)
    else:
        all_image_paths = []
        # If no extension given, then get all files
        if ext == '':
            dir_list = os.listdir(directory_path)
            for iPath in dir_list:
                if '.' != iPath[0]:  # Ignore hidden files
                    all_image_paths.append('{}/{}'.format(directory_path, str(iPath)))
        else:
            # Get list of paths for files with the extension ext
            data_root = pathlib.Path(directory_path)
            for iPath in data_root.glob('*.{}'.format(ext)):
                all_image_paths.append(str(iPath))

    return all_image_paths


def get_paths_from_sub_directories(main_directory_path, sub_dir_list, ext=''):
    """Compiles a list of all paths within each sub directory listed in sub_dir_list that follows the main_dir_path
    :param main_directory_path: the path for the main directory of interest
    :param sub_dir_list: the name(s) of the directory of interest that are in the main_directory
    :param ext: the extension of the files of interest (in the usb directories)
    :return:
    """
    # Initialize list of paths
    path_list = []
    # Iterate through the sub directory names and build up the path list
    for sub_dir in sub_dir_list:
        paths_to_add = get_paths('{}/{}'.format(main_directory_path, sub_dir), ext=ext)
        path_list.extend(paths_to_add)

    return path_list



def make_directory_and_return_path(dir_path):
    """Makes a directory only if it does not already exist
    :param dir_path: the path of the directory to be made
    :return: returns the directory path
    """
    os.makedirs(dir_path, exist_ok=True)

    return dir_path

def z_score(array):
    array_mean = np.mean(array)
    array_theta = np.std(array)
    array = (array - array_mean) / array_theta
    return array

def scale(array, n):
    up_bound = np.max(array) * n[1]
    low_bound = np.max(array) * n[0]

    print("up, low", up_bound, low_bound)

    array[array > up_bound] = up_bound
    
    array[array < low_bound] = low_bound 
    
    array = (array - low_bound ) / up_bound
    return array 
    



