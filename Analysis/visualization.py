import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from layers.TNRD import RBF
import os
from keras import Input, Model

'''
Functions for plotting kernels and activation functions of TNRD model.
Code modified from TNRD GitHub: https://github.com/VLOGroup/denoising-variationalnetwork
'''



def rescaleKernel(kernel):
    """ rescaleKernel to range [0,1]."""
    vmin = np.min(kernel)
    vmax = np.max(kernel)
    vabs = np.maximum(np.abs(vmin), np.abs(vmax))
    rescaled_kernel = (2.0 * vabs) / (vmax - vmin) * (kernel - vmin) - vabs
    rescaled_kernel = (1.0) / (2*vabs) * (rescaled_kernel + vabs)
    return rescaled_kernel

#Plots and saves convolution kernels
def saveAllKernels(kernels, kernel_names, num_f, filename):
    plt.clf()

    num_rows = len(kernels)
    num_cols = num_f

    plt.subplots(num_rows, num_cols,figsize=(12,12))

    idx = 1
    for i in range(num_rows):
        for j in range(num_cols):
            ax = plt.subplot(num_rows, num_cols, idx)
            ax.set_title(kernel_names[i] + str(j+1),fontweight="bold", size=22)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            kernel = kernels[i][:,:,j]
            kernel = rescaleKernel(kernel)
            ax.imshow(kernel, cmap='gray')
            idx += 1
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.savefig(filename, format='png')

def plot_kernels(out_dir, layer, weight_loc):
    weights = layer.get_weights()[weight_loc]
    filters = weights.shape[3]
    all_kernels = [weights[:, :, 0, :]]
    kernel_names = ['k']
    saveAllKernels(all_kernels, kernel_names, filters, os.path.join(out_dir,str(weight_loc) +  'Kernels.png'))
    print('Kernels have been drawn in {}'.format(out_dir))

def Activation_Model(filters = 8, image_channels = 1):
    layer = 0
    inpt = Input(shape=(None,None,image_channels),name = 'input'+str(layer))
    x  = inpt
    x = RBF(filters = filters)(x)
    model = Model(inputs=inpt, outputs=x)

    return model

def plot_functions(out_dir, layer, weight_loc):
    weights = layer.get_weights()[weight_loc]
    tf.keras.backend.clear_session()
    filters = weights.shape[0]
    # Construct a RBF activation model to plot
    model = Activation_Model(filters=filters, image_channels=1)

    # Insert learned parameters in activation model
    for layer in model.layers:
        if layer.name == 'rbf_1':
            layer.w = weights

    test = np.linspace(2*(-0.5), 2*(0.5), num=1000)
    x = np.linspace(2*(-0.5), 2*(0.5), num=1000)
    test = np.expand_dims(test, 1)
    test = np.tile(test, [1, 2])
    test = np.expand_dims(test, 0)
    test = np.expand_dims(test, 3)
    test_ = np.tile(test, [1, 1, 1, filters])

    # Apply RBF activation functions from values in range (-0.5,0.5)
    act_output = model.predict([test_])
    print(os.path.join(out_dir, 'Phi' + str(1) + '.png'))

    # Plot individual RBF activation functions
    for i in range(filters):
        act = act_output[0, :, :, i]
        plt.plot(x, act[:, 0])
        plt.title('Activation Function')
        plt.grid()
        fig_name = os.path.join(out_dir, 'Phi' + str(i+1) + '.png')
        plt.savefig(fig_name)
        plt.clf()

    # Plot all RBF activation functions
    plt.grid()
    for i in range(filters):
        act = act_output[0, :, :, i]
        plt.plot(x, act[:, 0])
        plt.title('All Activation Functions')
    fig_name = os.path.join(out_dir, 'AllPhis')
    plt.savefig(fig_name)

    plt.clf()
    # Plot individual rho functions
    for i in range(filters):
        act = act_output[0, :, :, i]
        rho = np.cumsum(act[:, 0])
        rho -= np.min(rho)
        rho /= len(x)
        rho *= (x[1] - x[0])
        plt.plot(x, rho, 'r', linewidth=4)
        plt.title('Rho Function (Integration of Phi)')
        plt.grid()
        fig_name = os.path.join(out_dir, 'Rho' + str(i + 1) + '.png')
        plt.savefig(fig_name)
        plt.clf()

    # Plot all rho functions
    plt.grid()
    for i in range(filters):
        act = act_output[0, :, :, i]
        rho = np.cumsum(act[:, 0])
        rho -= np.min(rho)
        rho /= len(x)
        rho *= (x[1] - x[0])
        plt.plot(x, rho)
        plt.plot(x, rho)
        plt.title('All Rho Functions (Integration of Phis)')
    fig_name = os.path.join(out_dir, 'AllRhos')
    plt.savefig(fig_name)

    print('Acivation Functions have been plotted to {}'.format(out_dir))