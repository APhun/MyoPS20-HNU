import numpy as np
from skimage import transform, io
import matplotlib.pyplot as plt
from provided_code.general_functions import get_paths, load_file


class DataLoader:
    """Generates data for tensorflow"""

    def __init__(self, file_paths_list, batch_size=20, shuffle=True,
                 mode_name='training_model'):
        """Initialize the DataLoader class, which loads the data for OpenKBP
        :param file_paths_list: list of the directories or single files where data for each patient is stored
        :param batch_size: the number of data points to lead in a single batch
        :param patient_shape: the shape of the patient data
        :param shuffle: whether or not order should be randomized
        """
        # Set file_loader specific attributes
        self.modalities = ['C0', 'T2', 'DE']
        self.batch_size = batch_size  # Number of patients to load in a single batch
        self.indices = np.arange(len(file_paths_list))  # Indices of file paths
        self.file_paths_list = file_paths_list  # List of file paths
        self.shuffle = shuffle  # Indicator as to whether or not data is shuffled
        self.num_modalities = len(self.modalities)
        self.patient_id_list = ['pt_{}'.format(k.split('/pt_')[1].split('/')[0].split('.nii.gz')[0]) for k in
                                self.file_paths_list]  # the list of patient ids with information in this data loader
        print(self.patient_id_list)

        # Set files to be loaded
        self.required_files = None
        self.mode_name = mode_name  # Defines the mode for which data must be loaded for
        self.set_mode(self.mode_name)  # Set load mode to prediction by default

    def get_batch(self, index=None, patient_list=None):
        """Loads one batch of data
        :param index: the index of the batch to be loaded
        :param patient_list: the list of patients for which to load data for
        :return: a dictionary with the loaded data
        """

        if patient_list is None:
            # Generate batch based on the provided index
            indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        else:
            # Generate batch based on the request patients
            indices = self.patient_to_index(patient_list)

        # Make a list of files to be loaded
        file_paths_to_load = [self.file_paths_list[k] for k in indices]

        # Load the requested files as a tensors
        loaded_data = self.load_data(file_paths_to_load)
        return loaded_data

    def patient_to_index(self, patient_list):
        """Converts a list of patient ids to their appropriate indices
        :param patient_list: list of patient ids
        :return: list of indices for the requested patients
        """
        # Get the indices for the list that is not shuffled
        un_shuffled_indices = [self.patient_id_list.index(k) for k in patient_list]

        # Map the indices to the shuffled indices to the shuffled indices
        shuffled_indices = [self.indices[k] for k in un_shuffled_indices]

        return shuffled_indices

    def set_mode(self, mode_name, single_file_name=None):
        """Selects the type of data that is loaded
        :param mode_name: the name of the mode that the data loader is switching to
        :param single_file_name: the name of the file that should be loaded (only used if the mode_name is 'single_file')
        """
        self.mode_name = mode_name

        if mode_name == 'pre_training_model':
            # The mode that should be used when training or validing a model
            self.required_files = ['C0', 'T2', 'DE', 'gd']

        elif mode_name == 'training_model':
            # The mode that should be used when training or validing a model
            self.required_files = ['C0', 'T2', 'DE', 'gd']

        elif mode_name == 'predicted_segment':
            self.required_files = ['C0', 'T2', 'DE', 'gd']
            self.batch_size = 1
            print('Warning: Batch size has been changed to 1 for prediction mode')

        elif mode_name == 'evaluation':
            self.required_files = ['C0', 'T2', 'DE']
            self.batch_size = 1
            print('Warning: Batch size has been changed to 1 for prediction mode')

        else:
            print('Mode does not exist.')

    def number_of_batches(self):
        """Calculates how many full batches can be made in an epoch
        :return: the number of batches that can be loaded
        """
        return int(np.floor(len(self.file_paths_list) / self.batch_size))

    def on_epoch_end(self):
        """Randomizes the indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def load_data(self, file_paths_to_load):
        """Generates data containing batch_size samples X : (n_samples, *dim, n_channels)
        :param file_paths_to_load: the paths of the files to be loaded
        :return: a dictionary of all the loaded files
        """

        # Initialize dictionary for loaded data and lists to track patient path and ids
        tf_data = {}.fromkeys(self.required_files)
        for key in tf_data:
            tf_data[key] = []
        patient_list = []
        patient_path_list = []
        patient_shape_list = []
        patient_affine_list = []

    
        '''
        # Loop through each key in tf data to initialize the tensor with zeros
        for key in tf_data:
            print(*self.required_files[key])
            # Make dictionary with appropriate data sizes for bath learning
            tf_data[key] = np.zeros((self.batch_size, *self.required_files[key]))
        '''
        # Generate data
        for i, pat_path in enumerate(file_paths_to_load):
            # Get patient ID and location of processed data to load
            patient_path_list.append(pat_path)
            pat_id = pat_path.split('/')[-1].split('.')[0]
            patient_list.append(pat_id)
            # Make a dictionary of all the tensors
            loaded_data_dict, affine = self.load_and_shape_data(pat_path)
            patient_shape_list.append(loaded_data_dict['C0'].shape)
            patient_affine_list.append(affine)
            # Iterate through the dictionary add the loaded data to the "batch channel"
            for key in tf_data:
                tf_data[key].append(loaded_data_dict[key])

        # Add two keys to the tf_data dictionary to track patient information
        tf_data['patient_list'] = patient_list
        tf_data['patient_path_list'] = patient_path_list
        tf_data['patient_shape_list'] = patient_shape_list
        tf_data['patient_affine_list'] = patient_affine_list

        return tf_data

    def load_and_shape_data(self, path_to_load):
        """ Reshapes data that is stored as vectors into matrices
        :param path_to_load: the path of the data that needs to be loaded. If the path is a directory, all data in the
         directory will be loaded. If path is a file then only that file will be loaded.
        :return: Loaded data with the appropriate shape
        """

        # Initialize the dictionary for the loaded files
        loaded_file = {}
        if '.csv' in path_to_load:
            loaded_file[self.mode_name] = load_file(path_to_load)
        else:
            files_to_load = get_paths(path_to_load, ext='')

            # Load files and get names without file extension or directory
            for f in files_to_load:
                #'./provided_data/train/pt_121/myops_training_121_T2.nii.gz'
                f_name = f.split('/')[-1].split('_')[3].split('.')[0]
                if f_name in self.required_files:
                    loaded_file[f_name], affine = load_file(f)


        shaped_data = {}.fromkeys(self.required_files)

        

        # Populate matrices that were no initialized as []
        for key in shaped_data:
            shaped_data[key] = loaded_file[key]
        

        return shaped_data, affine
