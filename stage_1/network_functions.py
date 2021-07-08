""" This class inherits a network architecture and performs various functions on a define architecture like training
 and predicting"""

import os
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from skimage import transform, io
import matplotlib.pyplot as plt
from provided_code.general_functions import get_paths, make_directory_and_return_path
from provided_code.network_architectures import DefineUnet, dice_loss, dice_coff


class PredictionModel(DefineUnet):

    def __init__(self, data_loader, results_patent_path, model_name, stage='training'):
        # set attributes for data shape from data loader
        self.data_loader = data_loader
        self.model_name = model_name

        # Define training parameters
        self.epoch_start = 0  # Minimum epoch (overwritten during initialization if a newer model exists)
        self.epoch_last = 2000  # When training will stop


        # Define filter and stride lengths
        self.filter_size = 4
        self.stride_size = 1

        # Define the initial number of filters in the model (first layer)
        self.initial_number_of_filters = 8  


        # Define place holders for model
        self.unet = None

        # Make directories for data and models
        model_results_path = '{}/{}'.format(results_patent_path, model_name)
        self.model_dir = make_directory_and_return_path('{}/models'.format(model_results_path))
        self.prediction_dir = '{}/{}-predictions'.format(model_results_path, stage)

        # Make template for model path
        self.model_path_template = '{}/epoch_'.format(self.model_dir)

    def train_model(self, epochs=2000, save_frequency=5, keep_model_history=2):
        
        # Define new models, or load most recent model if model already exists
        self.epoch_last = epochs
        self.initialize_networks()

        # Check if training has already finished
        if self.epoch_start == epochs:
            return

        else:
            # Start training GAN
            num_batches = self.data_loader.number_of_batches()

            for e in range(self.epoch_start, epochs):
                # Begin a new epoch
                print('epoch number {}'.format(e))
                self.data_loader.on_epoch_end()  # Shuffle the data after each epoch
                for i in tqdm.tqdm(range(num_batches)):
                    # Load a subset of the data and train the network with the data
                    self.train_network_on_batch(i, e)

                # Create epoch label and save models at the specified save frequency
                current_epoch = e + 1
                
                if 0 == np.mod(current_epoch, save_frequency):
                    self.save_model_and_delete_older_models(current_epoch, save_frequency, keep_model_history)
                

    def save_model_and_delete_older_models(self, current_epoch, save_frequency, keep_model_history):

        # Save the model to a temporary path
        temporary_model_path = '{}_temp.h5'.format(self.model_path_template)
        copy_model_path = '{}{}_temp.h5'.format(self.model_path_template, current_epoch)
        self.unet.save(temporary_model_path)
        if current_epoch % 50 == 0:
            self.unet.save(copy_model_path)


        
        # Define the epoch that should be over written
        epoch_to_overwrite = current_epoch - keep_model_history * save_frequency
        # Make appropriate path to save model at
        if epoch_to_overwrite > 0:
            model_to_delete_path = '{}{}.h5'.format(self.model_path_template, epoch_to_overwrite)
        else:
            model_to_delete_path = '{}{}.h5'.format(self.model_path_template, current_epoch)
        # Save model
        os.rename(temporary_model_path, model_to_delete_path)
        # The code below is a hack to ensure the Google Drive trash doesn't fill up

        if epoch_to_overwrite > 0:
            final_save_model_path = '{}{}.h5'.format(self.model_path_template, current_epoch)
            os.rename(model_to_delete_path, final_save_model_path)
        

    def initialize_networks(self):

        # Initialize variables for models
        all_models = get_paths(self.model_dir, ext='h5')

        # Get last epoch of existing models if they exist
        for model_name in all_models:
            model_epoch_number = model_name.split(self.model_path_template)[-1].split('.h5')[0]
            if model_epoch_number.isdigit():
                self.epoch_start = max(self.epoch_start, int(model_epoch_number))

        # Build new models or load most recent old model if one exists
        if self.epoch_start >= self.epoch_last:
            print('Model fully trained, loading model from epoch {}'.format(self.epoch_last))
            return 0, 0, 0, self.epoch_last

        elif self.epoch_start >= 1:
            # If models exist then load them
            self.unet = load_model('{}{}.h5'.format(self.model_path_template, self.epoch_start), custom_objects={'dice_loss': dice_loss,'dice_coff':dice_coff})
        else:
            # If models don't exist then define them
            #self.define_GAN()
            self.define_unet()


    def train_network_on_batch(self, batch_index, epoch_number):

        # Load images
        image_batch = self.data_loader.get_batch(batch_index)

        C0 = image_batch['C0']
        gd = image_batch['gd']
        
        # Train the generator model with the batch

        C0_list = []
        gd_list = []
        for index in range(len(C0)):
            x, y, z = C0[index].shape

            gd[index][gd[index] == 200] = 1000

            C0[index] = C0[index] / np.max(C0[index]) * 255
            gd[index] = gd[index] / np.max(gd[index]) * 255
            '''
            io.imshow(gd[index][:,:,0])
            plt.show()
            
            gd[index][gd[index] == 200] = 1
            gd[index][gd[index] == 2221] = 1 
            gd[index][gd[index] == 1220] = 1
            gd[index][gd[index] != 1] = 0
            '''
            #io.imshow(gd[index][:,:,0])
            #plt.show()

            if np.random.rand() > 0.5:
                C0[index] = C0[index][:,::-1,:]
                gd[index] = gd[index][:,::-1,:]
            if np.random.rand() > 0.5:
                C0[index] = C0[index][::-1,:,:]
                gd[index] = gd[index][::-1,:,:]

            dr = np.random.randint(0, 10)
            if np.random.rand() > 0.5:
                dr *= -1

            C0_tmp = np.zeros(C0[index].shape)
            gd_tmp = np.zeros(gd[index].shape)
            for i in range(z):
                C0_tmp[:,:,i] = transform.rotate(C0[index][:,:,i], dr, resize=False)
                gd_tmp[:,:,i] = transform.rotate(gd[index][:,:,i], dr, resize=False)
                '''
                gd[index][gd[index] == 200] = 1
                gd[index][gd[index] == 2221] = 1 
                gd[index][gd[index] == 1220] = 1
                gd[index][gd[index] != 1] = 0
                
                io.imshow(C0_tmp[:,:,i])
                plt.show()
                '''

            slice = np.random.randint(z)
            C0_list.append(np.expand_dims(C0_tmp[x//2-128:x//2+128, y//2-128:y//2+128, slice], axis=-1))
            gd_list.append(np.expand_dims(gd_tmp[x//2-128:x//2+128, y//2-128:y//2+128, slice], axis=-1))
        
        C0_list = np.array(C0_list)
        gd_list = np.array(gd_list)

        C0_mean = np.mean(C0_list)
        C0_theta = np.std(C0_list)
        C0_list = (C0_list - C0_mean) / C0_theta

        gd_list[gd_list < 100] = 0
        gd_list[gd_list >= 100] = 1
        
        
        #unet_loss = self.unet_dice.train_on_batch([gd_list, C0_list], gd_list)
        unet_loss = self.unet.fit(C0_list, gd_list)

        test = np.expand_dims(C0_list[0], axis=0)
        test_label = np.expand_dims(gd_list[0], axis=0)
        

        pre = self.unet.predict(test)
        pre = pre * 255
        pre = pre.astype('uint8')
        pre = np.squeeze(pre)
        io.imsave('./test.png',pre)
        





