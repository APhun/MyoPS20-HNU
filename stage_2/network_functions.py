""" This class inherits a network architecture and performs various functions on a define architecture like training
 and predicting"""

import os
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from skimage import transform, io, measure, color, exposure
import matplotlib.pyplot as plt
from provided_code.general_functions import get_paths, make_directory_and_return_path, z_score, scale
from provided_code.network_architectures import DefineMnet, dice_loss, dice_coff, dice_coff_edema, dice_coff_scar
from provided_code import data_loader
from tensorflow.nn import softmax_cross_entropy_with_logits
import nibabel as nib


class PredictionModel(DefineMnet):

    def __init__(self, data_loader, results_patent_path, model_name, stage='training', validation=None):
        """
        Initialize the Prediction model class
        :param data_loader: An object that loads batches of image data
        :param results_patent_path: The path at which all results and generated models will be saved
        :param model_name: The name of your model, used when saving and loading data
        """
        # set attributes for data shape from data loader
        self.data_loader = data_loader
        self.model_name = model_name

        # Define training parameters
        self.epoch_start = 0  # Minimum epoch (overwritten during initialization if a newer model exists)
        self.epoch_last = 2000  # When training will stop


        # Define filter and stride lengths
        self.filter_size = 3
        self.stride_size = 1

        # Define the initial number of filters in the model (first layer)
        self.initial_number_of_filters = 64  # 64


        # Define place holders for model
        self.unet = None

        # Make directories for data and models
        model_results_path = '{}/{}'.format(results_patent_path, model_name)
        self.model_dir = make_directory_and_return_path('{}/models'.format(model_results_path))
        self.predict_dir = make_directory_and_return_path('{}/submit'.format(model_results_path))
        self.prediction_dir = '{}/{}-predictions'.format(model_results_path, stage)

        # Make template for model path
        self.model_path_template = '{}/epoch_'.format(self.model_dir)
        self.locate_1_path = '{}/locate_1.h5'.format(self.model_dir)
        self.locate_2_path = '{}/locate_2.h5'.format(self.model_dir)
        self.rough = '{}/rough.h5'.format(self.model_dir)
        self.predict = '{}/epoch_600.h5'.format(self.model_dir)
        self.unet_1 = load_model(self.locate_1_path, custom_objects={'dice_loss': dice_loss,'dice_coff':dice_coff})
        self.unet_2 = load_model(self.locate_2_path, custom_objects={'dice_loss': dice_loss,'dice_coff':dice_coff})
        self.unet_3 = load_model(self.rough, custom_objects={'dice_loss': dice_loss,'dice_coff':dice_coff})

        self.validation = validation

    def train_model(self, epochs=2000, save_frequency=5, keep_model_history=2):
        """
        Train the model over several epochs
        :param epochs: the number of epochs the model will be trained over
        :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
        :param keep_model_history: how many models back are kept (anything older than
        save_frequency*keep_model_history epochs)
        :return: None
        """
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
        """
        Save the current model and delete old models, based on how many models the user has asked to keep. We overwrite
        files (rather than deleting them) to ensure the user's trash doesn't fill up.
        :param current_epoch: the current epoch number that is being saved
        :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
        :param keep_model_history: how many models back are kept (anything older than
        save_frequency*keep_model_history epochs)
        """

        # Save the model to a temporary path
        temporary_model_path = '{}_temp.h5'.format(self.model_path_template)
        copy_model_path = '{}{}_temp.h5'.format(self.model_path_template, current_epoch)
        self.unet.save(temporary_model_path)
        if current_epoch % 100 == 0:
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
        """
        Load the newest model, or if no model exists with the appropriate name a new model will be created.
        :return:
        """
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
            self.unet = load_model('{}{}.h5'.format(self.model_path_template, self.epoch_start), custom_objects={'dice_loss': dice_loss,'dice_coff_edema':dice_coff_edema, 'dice_coff_scar':dice_coff_scar, 'dice_coff':dice_coff, 'softmax_cross_entropy_with_logits_v2':softmax_cross_entropy_with_logits})

            
        else:
            # If models don't exist then define them
            #self.define_GAN()
            self.define_unet()

    def train_network_on_batch(self, batch_index, epoch_number):
        """Loads a sample of data and uses it to train the model
        :param batch_index: The batch index
        :param epoch_number: The epoch
        """
        # Load images

        image_batch = self.data_loader.get_batch(batch_index)
        validation_image_batch = self.validation.get_batch(batch_index)
        

        C0 = image_batch['C0']
        T2 = image_batch['T2']
        DE = image_batch['DE']
        gd = image_batch['gd']

        # Train the generator model with the batch

        C0_list = []
        T2_list = []
        DE_list = []
        gd_list = []
        print(len(C0))

        for index in range(len(C0)):
            x, y, z = C0[index].shape

            T2[index][T2[index] > 0.5*np.max(T2[index])] = 0.5*np.max(T2[index])
            
            DE[index][DE[index] > 1000] = 1000

            C0[index] = C0[index] / np.max(C0[index])
            T2[index] = T2[index] / np.max(T2[index]) 
            DE[index] = DE[index] / np.max(DE[index])

            gd_tmp = np.zeros((x,y,z,4))

            gd[index][gd[index] == 600] = 0
            gd[index][gd[index] == 500] = 0

            gd_tmp[:,:,:,0][gd[index] == 1220] = 1
            gd_tmp[:,:,:,1][gd[index] == 2221] = 1
            gd_tmp[:,:,:,2][gd[index] == 200] = 1
            gd_tmp[:,:,:,3][gd[index] == 0] = 1

            gd[index] = gd_tmp

            if np.random.rand() > 0.5:
                C0[index] = C0[index][:,::-1,:]
                T2[index] = T2[index][:,::-1,:]
                DE[index] = DE[index][:,::-1,:]
                gd[index] = gd[index][:,::-1,:,:]
            if np.random.rand() > 0.5:
                C0[index] = C0[index][::-1,:,:]
                T2[index] = T2[index][::-1,:,:]
                DE[index] = DE[index][::-1,:,:]
                gd[index] = gd[index][::-1,:,:,:]

            dr = np.random.randint(0, 10)
            if np.random.rand() > 0.5:
                dr *= -1

            gd_tmp = np.zeros(gd[index][:,:,0,:].shape)
            i = np.random.randint(z)

            C0_tmp = transform.rotate(C0[index][:,:,i], dr, resize=False)
            T2_tmp = transform.rotate(T2[index][:,:,i], dr, resize=False)
            DE_tmp = transform.rotate(DE[index][:,:,i], dr, resize=False)

            gd_tmp[:,:,0] = transform.rotate(gd[index][:,:,i,0], dr, resize=False)
            gd_tmp[:,:,1] = transform.rotate(gd[index][:,:,i,1], dr, resize=False)
            gd_tmp[:,:,2] = transform.rotate(gd[index][:,:,i,2], dr, resize=False)
            gd_tmp[:,:,3] = transform.rotate(gd[index][:,:,i,3], dr, resize=False)


            C0_locate = np.expand_dims(C0_tmp[x//2-128:x//2+128, y//2-128:y//2+128], axis=[0,-1])
            C0_locate = z_score(C0_locate)
            
            pre_1 = self.unet_1.predict(C0_locate)
            pre_2 = self.unet_2.predict(C0_locate)

            pre = pre_1 + pre_2
            pre[pre < 1.4] = 0
            pre[pre >= 1.4] = 1

            connect = measure.label(np.squeeze(pre), connectivity=2, background=0)
            for n in range(np.max(connect) + 1):
                if len(np.where(connect == n)[0]) < 500:
                    connect[connect == n] = 0
            
            connect[connect != 0] = 1
            
            pre = connect

            pre_coord = np.where(pre == 1)

            if len(pre_coord[0]) == 0:

                x_mid = int(x/2)
                y_mid = int(y/2)

            else:
                x_mid = int (x / 2 + (pre_coord[0][0]+ pre_coord[0][-1]) / 2 - 128)
                y_mid = int (y / 2 + (max(pre_coord[1]) + min(pre_coord[1])) / 2 - 128)

            C0_tmp = C0_tmp[x_mid-64:x_mid+64, y_mid-64:y_mid+64]

            p2, p98 = np.percentile(np.squeeze(C0_tmp), (2, 98))
            C0_tmp = np.expand_dims(exposure.rescale_intensity(C0_tmp, in_range=(p2, p98)), axis=[0,-1])
            C0_tmp = z_score(C0_tmp)
            #print(C0_tmp.shape)
            C0_tmp = self.unet_3.predict(C0_tmp)[0,:,:,:]

            
            T2_tmp = T2_tmp[x_mid-64:x_mid+64, y_mid-64:y_mid+64]

            T2_tmp= exposure.adjust_gamma(T2_tmp, np.random.uniform(low=0.7, high=1.2))

            p2, p98 = np.percentile(np.squeeze(T2_tmp), (2, 98))
            T2_tmp = exposure.rescale_intensity(T2_tmp, in_range=(p2, p98))
            #T2_tmp = z_score(T2_tmp)
            
            T2_tmp = np.expand_dims(T2_tmp, axis=-1)


            #T2_tmp = z_score(T2_tmp)


            DE_tmp = DE_tmp[x_mid-64:x_mid+64, y_mid-64:y_mid+64]
            DE_tmp= exposure.adjust_gamma(DE_tmp, np.random.uniform(low=0.7, high=1.2))
            p2, p98 = np.percentile(np.squeeze(DE_tmp), (2, 98))
            DE_tmp = exposure.rescale_intensity(DE_tmp, in_range=(p2, p98))
            #DE_tmp = z_score(DE_tmp)
            
            DE_tmp = np.expand_dims(DE_tmp, axis=-1)

            #DE_tmp = z_score(DE_tmp)
            
            gd_tmp = gd_tmp[x_mid-64:x_mid+64, y_mid-64:y_mid+64, :]

            C0_list.append(C0_tmp)
            T2_list.append(T2_tmp)
            DE_list.append(DE_tmp)
            gd_list.append(gd_tmp)

        C0_list = np.array(C0_list)
        T2_list = np.array(T2_list)
        DE_list = np.array(DE_list)
        gd_list = np.array(gd_list)

        gd_list[gd_list < 0.5] = 0
        gd_list[gd_list >= 0.5] = 1


        input_list = np.concatenate([C0_list, T2_list, DE_list], axis=-1)
        #print(input_list.shape)


        validation_C0 = validation_image_batch['C0']
        validation_T2 = validation_image_batch['T2']
        validation_DE = validation_image_batch['DE']
        validation_gd = validation_image_batch['gd']

        print(validation_C0[1].shape)

        validation_C0_list = []
        validation_T2_list = []
        validation_DE_list = []
        validation_gd_list = []

        for index in range(len(validation_C0)):
            
            validation_C0_tmp = validation_image_batch['C0'][index]
            validation_T2_tmp = validation_image_batch['T2'][index]
            validation_DE_tmp = validation_image_batch['DE'][index]
            validation_gd_tmp = validation_image_batch['gd'][index]

            x, y, z = validation_C0_tmp.shape

            validation_T2_tmp[validation_T2_tmp > 0.5*np.max(validation_T2_tmp)] = 0.5*np.max(validation_T2_tmp)
            validation_DE_tmp[validation_DE_tmp > 1000] = 1000

            validation_gd_tmp_array = np.zeros((x,y,z,4))

            validation_gd_tmp[validation_gd_tmp == 600] = 0
            validation_gd_tmp[validation_gd_tmp == 500] = 0

            validation_gd_tmp_array[:,:,:,0][validation_gd_tmp == 1220] = 1
            validation_gd_tmp_array[:,:,:,1][validation_gd_tmp == 2221] = 1
            validation_gd_tmp_array[:,:,:,2][validation_gd_tmp == 200] = 1
            validation_gd_tmp_array[:,:,:,3][validation_gd_tmp == 0] = 1

            validation_gd_tmp = validation_gd_tmp_array



            validation_C0_tmp = validation_C0_tmp / np.max(validation_C0_tmp)
            validation_T2_tmp = validation_T2_tmp / np.max(validation_T2_tmp)
            validation_DE_tmp = validation_DE_tmp / np.max(validation_DE_tmp)


            validation_C0_tmp = np.transpose(validation_C0_tmp, (2,0,1))

            C0_locate = np.expand_dims(validation_C0_tmp[1][x//2-128:x//2+128, y//2-128:y//2+128], axis=[0,-1])
            C0_locate = z_score(C0_locate)

            pre_1 = self.unet_1.predict(C0_locate)
            pre_2 = self.unet_2.predict(C0_locate)

            pre = pre_1 + pre_2
            pre[pre < 1.4] = 0
            pre[pre >= 1.4] = 1

            connect = measure.label(np.squeeze(pre), connectivity=2, background=0)
            for n in range(np.max(connect) + 1):
                if len(np.where(connect == n)[0]) < 500:
                    connect[connect == n] = 0
            
            connect[connect != 0] = 1
            
            pre = connect

            pre_coord = np.where(pre == 1)

            x_mid = int (x / 2 + (pre_coord[0][0]+ pre_coord[0][-1]) / 2 - 128)
            y_mid = int (y / 2 + (max(pre_coord[1]) + min(pre_coord[1])) / 2 - 128)

            validation_C0_tmp = np.expand_dims(validation_C0_tmp, axis=-1)
            validation_T2_tmp = np.expand_dims(np.transpose(np.squeeze(validation_T2_tmp), (2,0,1)), axis=-1) 
            validation_DE_tmp = np.expand_dims(np.transpose(np.squeeze(validation_DE_tmp), (2,0,1)), axis=-1) 
            validation_gd_tmp = np.transpose(np.squeeze(validation_gd_tmp), (2,0,1,3))
        #gd = np.expand_dims(np.transpose(np.squeeze(gd), (2,0,1)), axis=-1) 


            validation_C0_tmp = validation_C0_tmp[:, x_mid-64:x_mid+64, y_mid-64:y_mid+64, :]
            validation_T2_tmp = validation_T2_tmp[:, x_mid-64:x_mid+64, y_mid-64:y_mid+64, :]
            validation_DE_tmp = validation_DE_tmp[:, x_mid-64:x_mid+64, y_mid-64:y_mid+64, :]
            validation_gd_tmp = validation_gd_tmp[:, x_mid-64:x_mid+64, y_mid-64:y_mid+64, :]



            for i in range(validation_C0_tmp.shape[0]):


                p2, p98 = np.percentile(np.squeeze(validation_C0_tmp[i,:,:,0]), (2, 98))
                tmp = self.unet_3.predict(z_score(np.expand_dims(exposure.rescale_intensity(np.squeeze(validation_C0_tmp[i,:,:,0]), in_range=(p2, p98)), axis=[0,-1])))
                validation_C0_list.append(tmp[0,:,:,:])

                p2, p98 = np.percentile(np.squeeze(validation_T2_tmp[i,:,:,0]), (2, 98))
                validation_T2_list.append(np.expand_dims(exposure.rescale_intensity(np.squeeze(validation_T2_tmp[i,:,:,0]), in_range=(p2, p98)), axis=-1))

                p2, p98 = np.percentile(np.squeeze(validation_DE_tmp[i,:,:,0]), (2, 98))
                validation_DE_list.append(np.expand_dims(exposure.rescale_intensity(np.squeeze(validation_DE_tmp[i,:,:,0]), in_range=(p2, p98)), axis=-1))
                
                #validation_gd_tmp[validation_gd_tmp>=0.5] = 1
                #validation_gd_tmp[validation_gd_tmp<0.5] = 0

                validation_gd_list.append(validation_gd_tmp[i,:,:,:])



        validation_C0 = np.array(validation_C0_list)
        validation_T2 = np.array(validation_T2_list)
        validation_DE = np.array(validation_DE_list)
        validation_gd = np.array(validation_gd_list)
        #validation_gd = validation_gd_tmp
        validation_list = np.concatenate([validation_C0, validation_T2, validation_DE], axis=-1)




        #print(validation_gd.shape)
        unet_loss = self.unet.fit(input_list, gd_list, validation_data=(validation_list, validation_gd), verbose=2)

        test = np.expand_dims(validation_list[4], axis=0)
        test_label = validation_gd[4]

        pre = self.unet.predict(test)
        pre[pre>=0.5] = 1

        pre = pre * 255
        pre = pre.astype('uint8')
        pre = np.squeeze(pre)
        test_label = test_label * 255
        test_label = test_label.astype('uint8')

        io.imsave('./test1.png',pre[:,:,0])
        io.imsave('./test2.png',pre[:,:,1])
        io.imsave('./test3.png',pre[:,:,2])
        io.imsave('./test4.png',pre[:,:,3])
        io.imsave('./test_label1.png', test_label[:,:,0])
        io.imsave('./test_label2.png', test_label[:,:,1])
        io.imsave('./test_label3.png', test_label[:,:,2])
        io.imsave('./test_label4.png', test_label[:,:,3])



    
    def predict_label(self, epoch=1):

        # Define new models, or load most recent model if model already exists

        os.makedirs(self.prediction_dir, exist_ok=True)
        number_of_batches = self.data_loader.number_of_batches()
        print('Predicting label')
        for idx in tqdm.tqdm(range(number_of_batches)):
            image_batch = self.data_loader.get_batch(idx)
            # Get patient ID and make a prediction
            pat_id = image_batch['patient_list'][0]
            pat_shape = image_batch['patient_shape_list'][0]
            affine = image_batch['patient_affine_list'][0]
            print(pat_id)
            print(pat_shape)
            x, y, z = pat_shape

            result = np.zeros(pat_shape)

            C0 = image_batch['C0'][0]
            T2 = image_batch['T2'][0]
            DE = image_batch['DE'][0]

            T2[T2 > 0.5*np.max(T2)] = 0.5*np.max(T2)
            #DE[DE > 1000] = 1000

            C0 = C0 / np.max(C0)
            T2 = T2 / np.max(T2)
            DE = DE / np.max(DE)

            C0_list = []
            T2_list = []
            T2_list_1 = []
            T2_list_2 = []
            DE_list = []
            DE_list_1 = []
            DE_list_2 = []
            
            C0_tmp = np.transpose(C0, (2,0,1))
            #print( C0_locate.shape)
            if C0_tmp.shape[0] > 3:
                C0_locate = np.expand_dims(C0_tmp[3][x//2-128:x//2+128, y//2-128:y//2+128], axis=[0,-1])
            else:
                C0_locate = np.expand_dims(C0_tmp[1][x//2-128:x//2+128, y//2-128:y//2+128], axis=[0,-1])
            C0_locate = z_score(C0_locate)


            C0_show = np.squeeze(C0_locate)

            io.imshow(C0_show)
            plt.show()
            
            print(C0_locate.shape)
            
            pre_1 = self.unet_1.predict(C0_locate)
            pre_2 = self.unet_2.predict(C0_locate)

            
            io.imshow(np.squeeze(pre_1))
            plt.show()

            io.imshow(np.squeeze(pre_2))
            plt.show()
            

            pre = pre_1 + pre_2
            pre[pre < 1.4] = 0
            pre[pre >= 1.4] = 1

            connect = measure.label(np.squeeze(pre), connectivity=2, background=0)
            for n in range(np.max(connect) + 1):

                if len(np.where(connect == n)[0]) < 1000:
                    connect[connect == n] = 0
            
            connect[connect != 0] = 1
            
            pre = connect

            pre_coord = np.where(pre == 1)

            x_mid = int (x / 2 + (pre_coord[0][0]+ pre_coord[0][-1]) / 2 - 128)
            y_mid = int (y / 2 + (max(pre_coord[1]) + min(pre_coord[1])) / 2 - 128)

            C0 = np.expand_dims(np.transpose(np.squeeze(C0), (2,0,1)), axis=-1) 
            T2 = np.expand_dims(np.transpose(np.squeeze(T2), (2,0,1)), axis=-1) 
            DE = np.expand_dims(np.transpose(np.squeeze(DE), (2,0,1)), axis=-1) 

            print(C0.shape)
            C0 = C0[:, x_mid-64:x_mid+64, y_mid-64:y_mid+64, :]
            T2 = T2[:, x_mid-64:x_mid+64, y_mid-64:y_mid+64, :]
            DE = DE[:, x_mid-64:x_mid+64, y_mid-64:y_mid+64, :]
            

            for i in range(C0.shape[0]):


                p2, p98 = np.percentile(np.squeeze(C0), (2, 98))
                tmp = self.unet_3.predict(z_score(np.expand_dims(exposure.rescale_intensity(np.squeeze(C0[i,:,:,0]), in_range=(p2, p98)), axis=[0,-1])))
                C0_list.append(tmp[0,:,:,:])

                p2, p98 = np.percentile(np.squeeze(T2[i,:,:,0]), (2, 98))
                T2_list.append(np.expand_dims(exposure.rescale_intensity(np.squeeze(T2[i,:,:,0]), in_range=(p2, p98)), axis=-1))
                T2_list_1.append(np.expand_dims(exposure.rescale_intensity(exposure.adjust_gamma(np.squeeze(T2[i,:,:,0]), 1.1), in_range=(p2, p98)), axis=-1))

                T2_list_2.append(np.expand_dims(exposure.rescale_intensity(exposure.adjust_gamma(np.squeeze(T2[i,:,:,0]), 1.2),in_range=(p2, p98)), axis=-1))

                

                p2, p98 = np.percentile(np.squeeze(DE[i,:,:,0]), (2, 98))
                DE_list.append(np.expand_dims(exposure.rescale_intensity(np.squeeze(DE[i,:,:,0]), in_range=(p2, p98)), axis=-1))

                DE_list_1.append(np.expand_dims(exposure.rescale_intensity(exposure.adjust_gamma(np.squeeze(DE[i,:,:,0]), 1.1),  in_range=(p2, p98)), axis=-1))
                DE_list_2.append(np.expand_dims(exposure.rescale_intensity(exposure.adjust_gamma(np.squeeze(DE[i,:,:,0]), 1.2), in_range=(p2, p98)), axis=-1))
                
            C0 = np.array(C0_list)
            T2 = np.array(T2_list)
            DE = np.array(DE_list)

            


            input_list = np.concatenate([C0, T2, DE], axis=-1)
            print(input_list.shape)

            T2 = np.array(T2_list_1)
            DE = np.array(DE_list_1)
            input_list_1 = np.concatenate([C0, T2, DE], axis=-1)

            T2 = np.array(T2_list_2)
            DE = np.array(DE_list_2)
            input_list_2 = np.concatenate([C0, T2, DE], axis=-1)


            #input_list[:,:,:,0] = z_score(input_list[:,:,:,0])
            #input_list[:,:,:,1] = z_score(input_list[:,:,:,1])
            #input_list[:,:,:,2] = z_score(input_list[:,:,:,2])
  

            self.unet = load_model('{}/epoch_600.h5'.format(self.model_dir), custom_objects={'dice_loss': dice_loss,'dice_coff':dice_coff, 'dice_coff_scar':dice_coff, 'dice_coff_edema':dice_coff})
            self.unet.summary()

            result1 = self.unet.predict(input_list, batch_size=input_list.shape[0])
            result2 = self.unet.predict(input_list_1, batch_size=input_list.shape[0])
            result3 = self.unet.predict(input_list_2, batch_size=input_list.shape[0])

            self.unet = load_model('{}/epoch_500.h5'.format(self.model_dir), custom_objects={'dice_loss': dice_loss,'dice_coff':dice_coff, 'dice_coff_scar':dice_coff, 'dice_coff_edema':dice_coff})
            result4 = self.unet.predict(input_list, batch_size=input_list.shape[0])
            result5 = self.unet.predict(input_list_1, batch_size=input_list.shape[0])
            result6 = self.unet.predict(input_list_2, batch_size=input_list.shape[0])



            result = (result1 + result2 + result4 + result5 )/4


            
            result = np.transpose(np.squeeze(result), (1,2,0,3))
            result_tmp = np.zeros((result.shape[0], result.shape[1],result.shape[2],1))
            result_tmp[result[:,:,:,0] >= 0.5] = 1220
            result_tmp[result[:,:,:,2] >= 0.5] = 200
            result_tmp[result[:,:,:,1] >= 0.5] = 2221

            for i in range(input_list.shape[0]):
                print(np.where(result_tmp[:,:,i]==1220)[0].shape[0], np.where(result_tmp[:,:,i]==2221)[0].shape[0])
                if( (np.where(result_tmp[:,:,i]==1220)[0].shape[0] > 500) and (np.where(result_tmp[:,:,i]==1220)[0].shape[0] < 900) and (np.where(result_tmp[:,:,i]==2221)[0].shape[0] < 150)):
                    result_tmp[:,:,i][result_tmp[:,:,i] == 1220] = 2221
                    print("reverse!", i)
            

            result = np.zeros(pat_shape)
            result[x_mid-64:x_mid+64, y_mid-64:y_mid+64, :] = np.squeeze(result_tmp)
            
            result = nib.Nifti1Image(result, affine)
            nib.save(result, os.path.join(self.predict_dir, 'myops_test_{}_seg.nii.gz'.format(pat_id.split('_')[1])))
