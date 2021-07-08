import os
import pathlib
import nibabel as nib
import numpy as np
import pandas as pd
from provided_code.network_architectures import DefineUnet, dice_loss, dice_coff


def load_file(file_name):
    
    loaded_file_data = nib.load(file_name).get_data()
    return loaded_file_data


def get_paths(directory_path, ext=''):

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

    # Initialize list of paths
    path_list = []
    # Iterate through the sub directory names and build up the path list
    for sub_dir in sub_dir_list:
        paths_to_add = get_paths('{}/{}'.format(main_directory_path, sub_dir), ext=ext)
        path_list.extend(paths_to_add)

    return path_list



def make_directory_and_return_path(dir_path):

    os.makedirs(dir_path, exist_ok=True)

    return dir_path


