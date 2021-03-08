'''
Custom layers specificly for TNRD model: https://arxiv.org/pdf/1508.02848.pdf
icg.activation_rbf opreator code compiled from https://github.com/VLOGroup/denoising-variationalnetwork
'''

from keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.initializers import Constant
import numpy as np
import icg


#Some different initalizations for the radial basis function weights. Taken from:
# https://github.com/VLOGroup/denoising-variationalnetwork
def construct_rbf_weights(filters,w_init,scale,vmin,vmax,num_weights):
    x_0 = np.linspace(vmin, vmax, num_weights, dtype=np.float32)
    if w_init == 'linear':
        w_0 = scale * x_0
    elif w_init == 'tv':
        w_0 = scale * np.sign(x_0)
    elif w_init == 'relu':
        w_0 = scale * np.maximum(x_0, 0)
    elif w_init == 'student-t':
        alpha = 100
        w_0 = scale * np.sqrt(alpha) * x_0 / (1 + 0.5 * alpha * x_0 ** 2)

    w_0 = w_0[np.newaxis, :]
    w = np.tile(w_0, (filters, 1))

    return w
'''
TNRD_Inference:
#####################################
Implements inference portion of TNRD model:

Apply filters, computes activation function, then apply transpose of filters
\sum{bar{k}}\phi(k(u))

Inputs:
###################################
filters - Number of filters
kernel_size - Size of filter kernels
activation_weights - Number of Gaussian radial basis functions used to learn activation across (vmin,vmax)
kernel_initializer - Method of kernel initialization
rbf_initializer - Method of radial basis functon weights initialization
rbf_init_scale - Scaling of initialization
vmin - Min domain for activation function. If x<vmin, rbf(x) = 0
vmax - Max domain for activaiton function. If x>vmax, rbf(x) = 0
F - True or False. False in case of regular TNRD. If F=True, layer uses a different set of filters for transpose operation
'''
class TNRD_Inference(Layer):
    def __init__(self, filters, kernel_size, activation_weights = 31, kernel_initializer = 'random_normal',
                 rbf_initializer = 'linear', rbf_init_scale = 0.04, vmin = -0.25, vmax = 0.25,F = False,**kwargs):

        super(TNRD_Inference, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation_weights = activation_weights
        self.vmin = vmin
        self.vmax = vmax
        self.k_init = kernel_initializer
        self.w_init = rbf_initializer
        self.w_init_scale = rbf_init_scale
        self.F = F
        super(TNRD_Inference, self).__init__(**kwargs)

    def build(self, input_shape):
        self.k = self.add_weight(name = 'filter', shape = [self.kernel_size,self.kernel_size,input_shape[3],self.filters],initializer = self.k_init,trainable=True)
        self.w = self.add_weight(name = 'rbf_weights', shape = [self.filters,self.activation_weights],
                                 initializer = Constant(construct_rbf_weights(self.filters,self.w_init,self.w_init_scale,self.vmin,self.vmax,self.activation_weights)),
                                 trainable = True)
        if self.F:
            self.k2 = self.add_weight(name='filter2',
                                     shape=[self.kernel_size, self.kernel_size, input_shape[3], self.filters],
                                     initializer=self.k_init, trainable=True)
        super(TNRD_Inference, self).build(input_shape)

    def call(self, inputs):
        u_t_1 = inputs

        #Apply padding
        #u_p = tf.pad(tensor=u_t_1, paddings=[[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]], mode='REFLECT')
        #Apply convolutions
        u_k = tf.nn.conv2d(input=u_t_1, filters=self.k, strides=[1, 1, 1, 1], padding='SAME')
        #Apply radial basis function
        f1 = icg.activation_rbf(u_k, self.w, v_min=self.vmin, v_max=self.vmax, num_weights=self.w.shape[1],
                                 feature_stride=1)
        # Apply transposed convolutions
        if self.F:
            Ru = tf.nn.conv2d_transpose(f1, self.k2, tf.shape(input=u_t_1), [1, 1, 1, 1], 'SAME') / (self.filters)
        else:
            Ru = tf.nn.conv2d_transpose(f1, self.k, tf.shape(input=u_t_1), [1, 1, 1, 1], 'SAME') / (self.filters)

        return Ru

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation_weights': self.activation_weights,
            'vmin': self.vmin,
            'vmax': self.vmax,
            'kernel_initializer':self.k_init,
            'rbf_initializer':self.w_init,
            'rbf_init_scale':self.w_init_scale,
        }

        base_config = super(TNRD_Inference, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''
##########################################
# RBF:
# Radial basis function activation layer
##########################################
# Inputs:
#   -filters: Number of filters
#   -activation_weights: Number of weights in the radial basis functions (RBFs)
#   -rbf_initializer: Choice of radial basis function weights initalizer
#           options: linear, tv, relu, student-t
#   -rbf_init_scale: Scaling of rbf initializations
#   -vmin: Lower range of RBF
#   -vmax: Upper range of RBF
'''
class RBF(Layer):
    def __init__(self, filters, activation_weights = 31,rbf_initializer = 'linear', rbf_init_scale = 0.04, vmin = -0.25, vmax = 0.25,**kwargs):

        super(RBF, self).__init__()
        self.filters = filters
        self.activation_weights = activation_weights
        self.vmin = vmin
        self.vmax = vmax
        self.w_init = rbf_initializer
        self.w_init_scale = rbf_init_scale
        super(RBF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name = 'rbf_weights', shape = [self.filters,self.activation_weights],
                                 initializer = Constant(construct_rbf_weights(self.filters,self.w_init,self.w_init_scale,self.vmin,self.vmax,self.activation_weights)),
                                 trainable = True)

        super(RBF, self).build(input_shape)

    def call(self, x):
        #Apply radial basis functions
        f1 = icg.activation_rbf(x, self.w, v_min=self.vmin, v_max=self.vmax, num_weights=self.w.shape[1],
                                 feature_stride=1)
        return f1

    def get_config(self):
        config = {
            'filters': self.filters,
            'activation_weights': self.activation_weights,
            'vmin': self.vmin,
            'vmax': self.vmax,
            'rbf_initializer':self.w_init,
            'rbf_init_scale':self.w_init_scale,
        }

        base_config = super(RBF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''
##########################################
# RemovePaddingAndScale2d:
# Simple layer to remove padding and scale by the number of filters
##########################################
# Inputs:
#   -filters: Number of filters from convolution layer
#   -pad: Size of padding
'''
class RemovePaddingAndScale2d(Layer):
    def __init__(self, filters = 8, pad = 9, **kwargs):

        super(RemovePaddingAndScale2d, self).__init__()
        self.filters = filters
        self.pad = pad
        super(RemovePaddingAndScale2d, self).__init__(**kwargs)

    def build(self, input_shape):

        super(RemovePaddingAndScale2d, self).build(input_shape)

    def call(self, x):

        #Pad the image
        x = x[:, self.pad:-self.pad, self.pad:-self.pad, :] / (self.filters)

        return x

    def get_config(self):
        config = {
            'filters':self.filters,
            'pad': self.pad,
        }

        base_config = super(RemovePaddingAndScale2d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''
##########################################
# DataFidelity:
# Retrieves fidelity term
##########################################
# Inputs:
#   -learn_datatermweight: Choice to learn weight on data fidelity term
#   -datatermweight_init: Constant initialization for data term weight
'''
class FidelityDescent(Layer):
    def __init__(self, learn_datatermweight = True, datatermweight_init = 0.1, **kwargs):

        super(FidelityDescent, self).__init__()
        self.learn_datatermweight = learn_datatermweight
        self.lambdaa_init = datatermweight_init
        super(FidelityDescent, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lambdaa = self.add_weight(name = 'lambda', shape = [1], initializer=Constant(self.lambdaa_init), trainable = self.learn_datatermweight)

        super(FidelityDescent, self).build(input_shape)

    def call(self, inputs):
        u_t_1 = inputs[0]
        Ru = inputs[1]
        f = inputs[2]

        # data term
        Du = self.lambdaa * (u_t_1 - f)

        nabla_f_t = Ru + Du
        u_t = u_t_1 - 1. / (1 + self.lambdaa) * nabla_f_t

        return u_t

    def get_config(self):
        config = {
            'learn_datatermweight':self.learn_datatermweight,
            'datatermweight_init':self.lambdaa_init,
        }

        base_config = super(FidelityDescent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

