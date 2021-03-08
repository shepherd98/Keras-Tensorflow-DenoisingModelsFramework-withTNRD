#Ryan Cecil

##########################################
# To compute curvature with Keras-Tensorflow
##########################################

from keras.layers import Layer
import tensorflow as tf
import numpy as np

def get_Curvature_filters():

    dxforward = np.array([[0,0,0],[0,-1,0],[0,1,0]])
    dxforward = dxforward[:,:,np.newaxis,np.newaxis]

    dxback = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    dxback = dxback[:,:,np.newaxis,np.newaxis]

    dyforward = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    dyforward = dyforward[:,:,np.newaxis,np.newaxis]

    dybackward = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    dybackward = dybackward[:,:,np.newaxis,np.newaxis]

    dystepbackforward = np.array([[0, -1, 1], [0, 0, 0], [0, 0, 0]])
    dystepbackforward = dystepbackforward[:,:,np.newaxis,np.newaxis]

    dxstepleftforward = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
    dxstepleftforward = dxstepleftforward[:,:,np.newaxis,np.newaxis]

    return tf.constant(dxforward,dtype=tf.float32),tf.constant(dxback,dtype=tf.float32), \
            tf.constant(dyforward,dtype=tf.float32),tf.constant(dybackward,dtype=tf.float32), \
            tf.constant(dystepbackforward,dtype=tf.float32),tf.constant(dxstepleftforward,dtype=tf.float32)

class Curvature(Layer):
    def __init__(self, **kwargs):
        super(Curvature, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dxforward, self.dxback, self.dyforward, self.dybackward, self.dystepbackforward, self.dxstepleftforward = get_Curvature_filters()
        super(Curvature, self).build(input_shape)

    def call(self, u):
        eps = tf.constant(float("1.0e-16"))

        u_p = tf.pad(tensor=u, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        dxforwardu = tf.nn.conv2d(input=u_p, filters=self.dxforward, strides=[1, 1, 1, 1], padding='VALID')
        dxbacku = tf.nn.conv2d(input=u_p, filters=self.dxback, strides=[1, 1, 1, 1], padding='VALID')
        dyforwardu = tf.nn.conv2d(input=u_p, filters=self.dyforward, strides=[1, 1, 1, 1], padding='VALID')
        dybacku = tf.nn.conv2d(input=u_p, filters=self.dybackward, strides=[1, 1, 1, 1], padding='VALID')
        dystepbackforwardu = tf.nn.conv2d(input=u_p, filters=self.dystepbackforward, strides=[1, 1, 1, 1], padding='VALID')
        dxstepleftforwardu = tf.nn.conv2d(input=u_p, filters=self.dxstepleftforward, strides=[1, 1, 1, 1], padding='VALID')

        F = tf.sqrt(tf.square(dxforwardu) + tf.square(dyforwardu) + eps)
        G = tf.sqrt(tf.square(dxbacku) + tf.square(dystepbackforwardu) + eps)
        H = tf.sqrt(tf.square(dxstepleftforwardu) + tf.square(dybacku) + eps)

        Curvature = tf.divide(dxforwardu, F) - tf.divide(dxbacku, G) + tf.divide(dyforwardu, F) - tf.divide(dybacku, H)
        return Curvature


    def get_config(self):
        config = {
        }

        base_config = super(Curvature, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))








