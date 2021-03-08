'''
Ryan Cecil
File created 12/20/20
Updated: 12/29/20

Used for research.

Contains numpy methods for computing curvature
Contains Tensorflow code for computing curvature

'''
import numpy as np
import tensorflow as tf

def Curvature(tensor):
    """
    Takes in image tensor (batch_size, rows, cols, features)
    and computes the curvature of the images. This is the first
    method I wrote based on the matlab code. This code was not
    working in the ROF denoising.

    Parameters
    ----------
    tensor : Image tensor

    Returns
    -------
    Curvature of images

    """
    _,m,n,_ = tensor.shape
    eps = np.finfo(np.float32).tiny
    U = np.pad(tensor,((0,0),(1,1),(1,1),(0,0)),mode='symmetric')

    Uxpij = U[:,2:m + 2, 1: n + 1,:]-U[:,1: m + 1, 1: n + 1,:]
    Uypij = U[:,1:m + 1, 2: n + 2,:]-U[:,1: m + 1, 1: n + 1,:]

    Uxmij = U[:,1:m + 1, 1: n + 1,:]-U[:,0: m, 1: n + 1,:]
    Uymij = U[:,1:m + 1, 1: n + 1,:]-U[:,1: m + 1, 0: n,:]

    Uxpijm1 = U[:,2:m + 2, 0: n,:]-U[:,1: m + 1, 0: n,:]
    Uypim1j = U[:,0:m, 2: n + 2,:]-U[:,0: m, 1: n + 1,:]

    normgradij = np.sqrt(np.square(Uxpij) + np.square(Uypij) + eps)
    normgradim1j = np.sqrt(np.square(Uxmij) + np.square(Uypim1j) + eps)
    normgradijm1 = np.sqrt(np.square(Uxpijm1) + np.square(Uymij) + eps)

    vx = Uxpij/normgradij - Uxmij/normgradim1j
    vy = Uypij/normgradij - Uymij/normgradijm1

    return vx + vy

def Curvature2(tensor):
    """
    Takes in image tensor (batch_size, rows, cols, features)
    and computes the curvature of the images. This is the second
    method I wrote based on Katie's ROF code. This code works in
    the ROF denoising.

    Parameters
    ----------
    tensor : Image tensor

    Returns
    -------
    Curvature of images

    """
    b,m,n,f = tensor.shape
    eps = np.finfo(np.float32).tiny

    U = np.pad(tensor,((0,0),(1,1),(1,1),(0,0)),mode='symmetric')

    m = m+2
    n = n+2

    uyp = np.zeros(U.shape)
    uyn = np.zeros(U.shape)
    uxp = np.zeros(U.shape)
    uxn = np.zeros(U.shape)


    uyp[:,:,1:n,:] = U[:,:,0:n-1,:] - U[:,:,1:n,:]
    uyn[:,:,0:n-1,:] = U[:,:,0:n-1,:] - U[:,:,1:n,:]
    uxp[:,1:m,:,:] = U[:,0:m-1,:,:] - U[:,1:m,:,:]
    uxn[:,0:m-1,:,:] = U[:,0:m-1,:,:] - U[:,1:m,:,:]

    a = np.zeros(U.shape)
    b = np.zeros(U.shape)
    c = np.zeros(U.shape)
    d = np.zeros(U.shape)

    def get_value(udp, ud2p, ud2n, s,m,s2,n):
        out = udp[:, s:m, s2:n, :] / np.sqrt(
            eps + np.square(udp[:, s:m, s2:n, :]) + np.square(
                ((np.sign(ud2p[:, s:m, s2:n, :]) + np.sign(ud2n[:, s:m, s2:n, :])) / 2) * np.minimum(
                    np.abs(ud2p[:, s:m,s2:n, :]), np.abs(ud2n[:, s:m, s2:n]))))
        return out

    a[:,1:m-1,1:n-1,:] = get_value(uxp, uyp, uyn, 1,m-1,1,n-1)
    b[:,1:m-1,1:n-1,:] = get_value(uxp, uyp, uyn, 2, m, 1,n - 1)
    c[:,1:m-1,1:n-1,:] = get_value(uyp, uxp, uxn, 1, m - 1, 1,n - 1)
    d[:,1:m-1,1:n-1,:] = get_value(uyp, uxp, uxn, 1, m - 1, 2,n)

    K = a[:,1:m-1,1:n-1,:]-b[:,1:m-1,1:n-1,:]+c[:,1:m-1,1:n-1,:]-d[:,1:m-1,1:n-1,:]

    return K

def get_Curvature_filters():
    """

    Returns
    -------
    Filters for the tensorflow curvature method

    """

    kernels = np.zeros((3,3,1,4))

    kernels[:,:,0,0] = [[0,0,0],[0,-1,0],[0,1,0]]
    kernels[:, :, 0, 1] = [[0, 0, 0], [0, -1, 1], [0, 0, 0]]

    return tf.constant(kernels,dtype=tf.float32)


def tf_Curvature(u):
    """
    Takes in image tensor (batch_size, rows, cols, features)
    and computes the curvature of the images. This is based on
    the first Curvature method above. It is to be used with
    Tensorflow and uses filters instead to compute the curvature.
    This method is correct.

    Parameters
    ----------
    tensor : Image tensor

    Returns
    -------
    Curvature of images

    """
    kernels = get_Curvature_filters()

    eps = tf.constant(float("1.0e-16"))

    x = tf.nn.conv2d(u,kernels, strides=[1, 1, 1, 1], padding='SAME')
    norm = tf.sqrt(tf.reduce_sum(tf.square(x),axis=3, keepdims=True) + eps)
    x = tf.divide(x,norm)
    return x
