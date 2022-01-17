import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Layer
import tensorflow_addons as tfa

########################################################################
### https://www.tensorflow.org/tutorials/customization/custom_layers ###
########################################################################

class CustomConv2D(Layer):
    def __init__(self, filters, kernel_size, strides = (1, 1), padding = 'valid', use_bias = True, normalization = "batch", activation = 'leaky_relu', alpha = 0.02, **kwargs):
        super(CustomConv2D, self).__init__(**kwargs)
        self.conv2d = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = 'HeNormal')
        if normalization == "batch":
            self.normalization = layers.BatchNormalization()
        else:
            self.normalization = tfa.layers.InstanceNormalization()
        
        if activation == 'relu':
            self.activation = layers.ReLU()
        elif activation == 'leaky_relu':
            self.activation = layers.LeakyReLU(alpha)
        else:
            self.activation = layers.Activation(activation)
        self.regularization = layers.Dropout(0.2)

    def call(self, inputs, training = True):
        x = self.conv2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.regularization(x)
        return x

class CustomConvTranspose2D(Layer):
    def __init__(self, filters, kernel_size, strides = (1, 1), padding = 'valid', use_bias = True, normalization = "batch", activation = 'leaky_relu', alpha = 0.02, **kwargs):
        super(CustomConvTranspose2D, self).__init__(**kwargs)
        self.convtranspose2d = layers.Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = 'HeNormal')
        
        if normalization == "batch":
            self.normalization = layers.BatchNormalization()
        else:
            self.normalization = tfa.layers.InstanceNormalization()

        if activation == 'relu':
            self.activation = layers.ReLU()
        elif activation == 'leaky_relu':
            self.activation = layers.LeakyReLU(alpha)
        else:
            self.activation = layers.Activation(activation)

    def call(self, inputs, training = True):
        x = self.convtranspose2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)

        return x

######################################################################################################################################
### https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras ###
######################################################################################################################################

class ReflectionPadding2D(Layer):
    def __init__(self, padding = (1, 1), **kwargs):
        self.padding = padding
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask = None, training = True):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], mode = 'REFLECT')

class ConstantPadding2D(Layer):
    def __init__(self, padding = (1, 1), constant = 0, **kwargs):
        self.padding = tuple(padding)
        self.constant = constant
        super(ConstantPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask = None, training = True):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], mode = 'CONSTANT', constant_values = self.constant)
