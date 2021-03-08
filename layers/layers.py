'''
Ryan Cecil - Duquesne University (2020). Research under Dr. Stacey Levine, Ph.D.

Some miscellaneous custom layers for use in any model

Reference Paper at https://arxiv.org/abs/1508.02848
Reference Code at https://github.com/VLOGroup/denoising-variationalnetwork
'''

from keras.layers import Layer
from tensorflow.keras.initializers import Constant


'''
Scalar_Multiply;
Multiplies the input by a scalar. Can either be learned or not.

inputs:
- learn_scalar: True or False
- scalar_int: Initialization for scalar value
'''
class Scalar_Multiply(Layer):
    def __init__(self, learn_scalar = True, scalar_init = 0.1, **kwargs):
        super(Scalar_Multiply, self).__init__()
        self.learn_scalar = learn_scalar
        self.scalar_init = scalar_init
        super(Scalar_Multiply, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lambdaa = self.add_weight(name='lambda', shape=[1], initializer=Constant(self.scalar_init),
                                       trainable=self.learn_scalar)

        super(Scalar_Multiply, self).build(input_shape)

    def call(self, input):
        return self.lambdaa*input

    def get_config(self):
        config = {
            'learn_scalar' : self.learn_scalar,
            'scalar_init' : self.scalar_init,
        }

        base_config = super(Scalar_Multiply, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))