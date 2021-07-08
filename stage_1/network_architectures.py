''' Neural net architectures '''

from tensorflow.python.keras.layers import Input, LeakyReLU, BatchNormalization, \
    Conv2D, concatenate, Activation, SpatialDropout2D, AveragePooling2D, Conv2DTranspose, Flatten, Dense, Conv2D, Lambda, Reshape, add
from tensorflow_addons.layers import GroupNormalization
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras.losses import binary_crossentropy, huber_loss
import tensorflow as tf
import numpy as np
from skimage.filters import laplace


class DefineUnet:


    def generator_convolution(self, x, number_of_filters, use_group_norm=True):
        """Convolution block used for generator"""
        x = Conv2D(number_of_filters, self.filter_size, strides=self.stride_size, padding="same", kernel_initializer='he_normal',  use_bias=False)(x)
        if use_group_norm:
            x = GroupNormalization(groups=2, axis=3)(x)
        
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def residual_blocks(self, x, number_of_filters):
        y = Conv2D(number_of_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(x)
        y = BatchNormalization(momentum=0.99, epsilon=1e-3)(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2D(number_of_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(y)
        out = add([x,y])
        out = BatchNormalization(momentum=0.99, epsilon=1e-3)(out)
        out = LeakyReLU(alpha=0.2)(out)
        return out


    def generator_convolution_transpose(self, x, nodes, use_dropout=True, skip_x=None):
        """Convolution transpose block used for generator"""

        if skip_x is not None:
            skip_x.append(x)
            x = concatenate(skip_x)
        x = Conv2DTranspose(nodes, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        if use_dropout:
            x = SpatialDropout2D(0.2)(x)
        x = LeakyReLU(alpha=0)(x)  # Use LeakyReLU(alpha = 0) instead of ReLU because ReLU is buggy when saved

        return x

    def define_unet(self):
        C0 = Input((256, 256, 1))

        x1 = self.generator_convolution(C0, self.initial_number_of_filters)
        x2 = self.generator_convolution(x1, 2 * self.initial_number_of_filters)
        x3 = self.generator_convolution(x2, 4 * self.initial_number_of_filters)
        x4 = self.generator_convolution(x3, 8 * self.initial_number_of_filters)
        x5 = self.generator_convolution(x4, 8 * self.initial_number_of_filters)

        res_1 = self.residual_blocks(x5, 8 * self.initial_number_of_filters)
        res_2 = self.residual_blocks(res_1, 8 * self.initial_number_of_filters)
        res_3 = self.residual_blocks(res_2, 8 * self.initial_number_of_filters)

        x5b = self.generator_convolution_transpose(res_3, 8 * self.initial_number_of_filters, skip_x=[x5])
        x4b = self.generator_convolution_transpose(x5b, 8 * self.initial_number_of_filters, skip_x=[x4])
        x3b = self.generator_convolution_transpose(x4b, 4 * self.initial_number_of_filters, skip_x=[x3])
        x2b = self.generator_convolution_transpose(x3b, 2 * self.initial_number_of_filters, skip_x=[x2])
        x1b = self.generator_convolution_transpose(x2b, self.initial_number_of_filters, skip_x=[x1])
        x0b = Conv2DTranspose(1, self.filter_size, strides=self.stride_size, padding="same")(x1b)
        x_final = AveragePooling2D(3, strides=1, padding="same")(x0b)
        output = Activation("sigmoid")(x_final)

        self.unet = Model(inputs=C0, outputs=output)
        self.unet.compile(loss=[dice_loss], optimizer=Adam(0.0001, 0.5), metrics=dice_coff)
        self.unet.summary()


def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
    dice =  1 -  ((2. * intersection + 1) / (union + 1))
    return dice

def BE_loss(y_true, y_pred):
    '''
    tf.compat.v1.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None)
    '''
    minus = tf.abs(y_true - y_pred)
    minus = minus.numpy()
    print(minus)
    #minus_laplace = laplace(minus)
    #minus_sum = tf.reduce_sum(minus_laplace)
    minus_sum = tf.reduce_sum(minus)
    #print('minus_sum=', minus_sum)
    return minus_sum
    



def dice_coff(y_true, y_pred):
    y_pred = tf.where(y_pred >= 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
    dice =  ((2. * intersection) / (union))
    return dice