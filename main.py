import shutil

import numpy as np

from stage_1.data_loader import DataLoader
from stage_1.general_functions import get_paths, make_directory_and_return_path
from stage_1.network_functions import PredictionModel

if __name__ == '__main__':
    primary_directory = './provided_data'
    training_data_dir = '{}/train'.format(primary_directory)
    validation_data_dir = '{}/validation'.format(primary_directory)
    results_dir = '{}/results'.format(primary_directory)
    prediction_name = 'baseline'
    number_of_training_epochs = 2000

    # Prepare the data directory
    plan_paths = get_paths(training_data_dir, ext='')  # gets the path of each plan's directory
    num_train_pats = np.minimum(20, len(plan_paths))  # number of plans that will be used to train model
    training_paths = plan_paths[:num_train_pats]  # list of training plans
    hold_out_paths = plan_paths[num_train_pats:]  # list of paths used for held out testing

    # Train a model
    
    data_loader_train = DataLoader(training_paths)
    data_loader_hold_out = DataLoader(hold_out_paths, mode_name='predicted_segment')

    
    
    prediction_model_train = PredictionModel(data_loader_train, results_dir, model_name=prediction_name, validation=data_loader_hold_out)
    prediction_model_train.train_model(epochs=number_of_training_epochs, save_frequency=1, keep_model_history=1)
    
    
    data_loader_hold_out = DataLoader(hold_out_paths, mode_name='predicted_segment')
    prediction_model_hold_out = PredictionModel(data_loader_hold_out, results_dir,
                                                     model_name=prediction_name, stage='hold-out')
    prediction_model_hold_out.predict(epoch=number_of_training_epochs)
    
 