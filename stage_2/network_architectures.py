''' Neural net architectures '''

from tensorflow.python.keras.layers import Input, LeakyReLU, BatchNormalization, \
    Conv2D, concatenate, Activation, SpatialDropout2D, AveragePooling2D, Conv2DTranspose, Flatten, Dense, Conv2D, Lambda, Reshape, add, SeparableConv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, multiply
from tensorflow_addons.layers import GroupNormalization
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras.losses import binary_crossentropy, mean_absolute_error, huber_loss
from tensorflow.keras.regularizers import l2, l1  # l2:5e-4
import tensorflow as tf
import numpy as np
from skimage.filters import laplace
from tensorflow.nn import softmax_cross_entropy_with_logits

class DefineMnet:
    """This class defines the architecture for a U-NET and must be inherited by a child class that
    executes various functions like training or predicting"""

    def generator_convolution(self, x, number_of_filters, strides, use_batch_norm=False, use_group_norm=False, skip_x=None, use_seperate=False, use_dropout=True):
        """Convolution block used for generator"""

        if skip_x is not None:
            skip_x.append(x)
            x = concatenate(skip_x)
        if use_seperate == True:
            x = SeparableConv2D(number_of_filters, self.filter_size, padding="same", kernel_initializer='he_normal',  use_bias=False, depthwise_initializer='glorot_normal', pointwise_initializer='glorot_normal', depthwise_regularizer=l1(1e-6), pointwise_regularizer=l1(1e-6))(x)
        else:
            x = Conv2D(number_of_filters, self.filter_size, strides, padding="same", kernel_initializer='he_normal',  use_bias=False, kernel_regularizer=l1(1e-6))(x)
        if use_batch_norm:
            x = GroupNormalization(groups=8, axis=3, trainable=False)(x)
        if use_group_norm:
            x = BatchNormalization()(x)

        #x = self.channel_attention(x, number_of_filters) * x
        #x = self.spatial_attention(x, number_of_filters) * x
        
        x = LeakyReLU(alpha=0.2)(x)
        if use_dropout:
            x = SpatialDropout2D(0.2)(x)
        return x

    def channel_attention(self, x, number_of_filters, ratio=16):
        gap = GlobalAveragePooling2D()(x)
        gmp = GlobalMaxPooling2D()(x)
        gap_fc1 = Dense(number_of_filters//ratio, kernel_initializer='he_normal', activation='relu')(gap)
        gap_fc2 = Dense(number_of_filters, kernel_initializer='he_normal', activation='sigmoid')(gap_fc1)

        gmp_fc1 = Dense(number_of_filters//ratio, kernel_initializer='he_normal', activation='relu')(gmp)
        gmp_fc2 = Dense(number_of_filters, kernel_initializer='he_normal', activation='sigmoid')(gmp_fc1)

        out = gap_fc2 + gmp_fc2
        out = tf.nn.sigmoid(out)
        out = Reshape((1, 1, number_of_filters))(out)
        return out

    def spatial_attention(self, x, number_of_filters):
        gap = tf.reduce_mean(x, axis=3)
        gmp = tf.reduce_max(x, axis=3)
        out = tf.stack([gap, gmp], axis=-1) 
        out = Conv2D(1, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(out)
        return out


    def residual_blocks(self, x, number_of_filters, use_batch_norm=False, use_group_norm=True):
        y = Conv2D(number_of_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(x)
        y = BatchNormalization(momentum=0.99, epsilon=0.001, trainable=False)(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2D(number_of_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(y)

        y = self.channel_attention(y, number_of_filters) * y
        y = self.spatial_attention(y, number_of_filters) * y

        out = add([x,y])        
        out = LeakyReLU(alpha=0.2)(out)
        return out

    def concat_up_blocks(self, x, number_of_filters, strides, skip_x=None, use_dilate=False, use_batch_norm=False, use_group_norm=False):
        if skip_x is not None:
            skip_x.append(x)
            x = concatenate(skip_x)
        y = self.generator_convolution(x, number_of_filters, strides, use_seperate=True, use_batch_norm=use_batch_norm, use_group_norm=use_group_norm)

        z = self.generator_convolution(y, number_of_filters, strides, use_seperate=True, use_batch_norm=use_batch_norm, use_group_norm=use_group_norm)
        z = self.channel_attention(z, number_of_filters) * z
        if use_dilate==False:
            return z
        z = add([y,z])
        out = self.generator_convolution(z, number_of_filters, strides*2, use_batch_norm=use_batch_norm, use_group_norm=use_group_norm)
        return out

    def concat_down_blocks(self, x, number_of_filters, strides, skip_x=None, use_batch_norm=False, use_group_norm=True):
        if skip_x is not None:
            skip_x.append(x)
            x = concatenate(skip_x)
        y = self.generator_convolution(x, number_of_filters, strides, use_seperate=False, use_batch_norm=use_batch_norm, use_group_norm=use_group_norm)
        #y = concatenate([x,y])
        z = self.generator_convolution(y, number_of_filters, strides, use_seperate=False, use_batch_norm=use_batch_norm, use_group_norm=use_group_norm)
        z = self.spatial_attention(z, number_of_filters) * z
        z = add([y,z])
        out = UpSampling2D(size=2)(z)
        
        return out

    def define_unet(self):

        # Define inputs

        input = Input((128,128,3))

        pooling1 = MaxPooling2D((2, 2))(input) 
        pooling2 = MaxPooling2D((2, 2))(pooling1)
        pooling3 = MaxPooling2D((2, 2))(pooling2)

        x1 = self.concat_up_blocks(input, self.initial_number_of_filters, strides=1, use_dilate=True, use_batch_norm=False, use_group_norm=False) 
        x2 = self.concat_up_blocks(x1, 2 * self.initial_number_of_filters, strides=1, use_dilate=True, skip_x=[pooling1], use_batch_norm=False, use_group_norm=False) 
        x3 = self.concat_up_blocks(x2, 4 * self.initial_number_of_filters, strides=1, use_dilate=True, skip_x=[pooling2], use_batch_norm=False, use_group_norm=False) 
        x4 = self.concat_up_blocks(x3, 8 * self.initial_number_of_filters, strides=1, use_dilate=True, skip_x=[pooling3], use_batch_norm=False, use_group_norm=False) 


        res_1 = self.residual_blocks(x4, 8 * self.initial_number_of_filters)
        res_2 = self.residual_blocks(res_1, 8 * self.initial_number_of_filters)
        res_3 = self.residual_blocks(res_2, 8 * self.initial_number_of_filters)

        x4b = self.concat_down_blocks(res_3, 8 * self.initial_number_of_filters, strides=1, skip_x=[x4], use_batch_norm=False, use_group_norm=False) 
        x3b = self.concat_down_blocks(x4b, 4 * self.initial_number_of_filters, strides=1, skip_x=[x3], use_batch_norm=False, use_group_norm=False) 
        x2b = self.concat_down_blocks(x3b, 2 * self.initial_number_of_filters, strides=1, skip_x=[x2], use_batch_norm=False, use_group_norm=False)
        x1b = self.concat_down_blocks(x2b, 1 * self.initial_number_of_filters, strides=1, skip_x=[x1], use_batch_norm=False, use_group_norm=False)

        up4 = UpSampling2D((16,16))(res_3)
        up3 = UpSampling2D((8,8))(x4b)
        up2 = UpSampling2D((4,4))(x3b)
        up1 = UpSampling2D((2,2))(x2b)
        concat = concatenate([up4, up3, up2, up1, x1b])
        concat = self.generator_convolution(concat, self.initial_number_of_filters, strides=1, use_seperate=False, use_batch_norm=False, use_group_norm=False)



        x0b = Conv2D(4, (1,1), strides=self.stride_size, padding="same")(concat)
        output_label = Activation("softmax")(x0b)

        self.unet = Model(inputs=input, outputs=output_label, name="generator")
        self.unet.compile(loss=[dice_loss], optimizer=Adam(3e-4, 0.5), metrics=[dice_coff_edema, dice_coff_scar, dice_coff])
        self.unet.summary()
        


def dice_loss(y_true, y_pred):

    intersection = tf.reduce_sum(y_true[:,:,:,0] * y_pred[:,:,:,0])
    union = tf.reduce_sum(y_pred[:,:,:,0]) +  tf.reduce_sum(y_true[:,:,:,0])
    dice1 = 1 - tf.math.pow(((2. * intersection + 1) / (union + 1)), 1/2)

    intersection = tf.reduce_sum(y_true[:,:,:,1] * y_pred[:,:,:,1])
    union = tf.reduce_sum(y_pred[:,:,:,1]) +  tf.reduce_sum(y_true[:,:,:,1])
    dice2 = 1 - tf.math.pow(((2. * intersection + 1) / (union + 1)), 1/2)

    intersection = tf.reduce_sum(y_true[:,:,:,2] * y_pred[:,:,:,2])
    union = tf.reduce_sum(y_pred[:,:,:,2]) +  tf.reduce_sum(y_true[:,:,:,2])
    dice3 = 1 - tf.math.pow(((2. * intersection + 1) / (union + 1)), 1/2)

    intersection = tf.reduce_sum(y_true[:,:,:,3] * y_pred[:,:,:,3])
    union = tf.reduce_sum(y_pred[:,:,:,3]) +  tf.reduce_sum(y_true[:,:,:,3])
    dice4 = 1 - tf.math.pow(((2. * intersection + 1) / (union + 1)), 1/2)

    dice = 100 * huber_loss(y_pred, y_true) + dice1+dice2+dice3+0.5*dice4

    return dice


def dice_coff_edema(y_true, y_pred):
    y_pred = tf.where(y_pred >= 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    intersection = tf.reduce_sum(y_true[:,:,:,0] * y_pred[:,:,:,0])
    union = tf.reduce_sum(y_true[:,:,:,0]) + tf.reduce_sum(y_pred[:,:,:,0])
    dice =  ((2. * intersection) / (union))
    return dice

def dice_coff_scar(y_true, y_pred):
    y_pred = tf.where(y_pred >= 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    intersection = tf.reduce_sum(y_true[:,:,:,1] * y_pred[:,:,:,1])
    union = tf.reduce_sum(y_true[:,:,:,1]) + tf.reduce_sum(y_pred[:,:,:,1])
    dice =  ((2. * intersection) / (union))
    return dice

def dice_coff(y_true, y_pred):
    y_pred = tf.where(y_pred >= 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice =  ((2. * intersection) / (union))
    return dice


