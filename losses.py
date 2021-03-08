'''
Just a simple file for storing loss functions that I need to use in models.py
'''

from keras.losses import mse
import keras.backend as K

def sum_squared_error_loss(clean,x):
    """
    Simple implementation of sum squared error loss
    Parameters
    ----------
    clean : Clean image tensor
    x : Estimate of clean image tensor

    Returns
    -------
    Sum of squared error loss

    """
    loss = K.sum(mse(clean, x)) / 2
    return loss