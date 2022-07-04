"""Python layer for icgvn_ops."""

from tensorflow.python.framework import ops
import tensorflow as tf

#Must be modified to match system
icg_module = tf.load_op_library(
        '/home/zsz/TestGit/Keras-Tensorflow-DenoisingModelsFramework-withTNRD/CondaEnvandICG/icg_bazel_files/icg.so')

fftshift2d = icg_module.fftshift2d
ifftshift2d = icg_module.ifftshift2d

activation_rbf = icg_module.activation_rbf
activation_prime_rbf = icg_module.activation_prime_rbf
activation_int_rbf = icg_module.activation_int_rbf
activation_interpolate_linear = icg_module.activation_interpolate_linear
activation_b_spline = icg_module.activation_b_spline
activation_cubic_b_spline = icg_module.activation_cubic_b_spline
activation_prime_cubic_b_spline = icg_module.activation_prime_cubic_b_spline

@ops.RegisterGradient("Fftshift2d")
def _Fftshift2dGrad(op, grad):
    in_grad = icg_module.ifftshift2d(grad)
    return [in_grad]

@ops.RegisterGradient("Ifftshift2d")
def _Iftshift2dGrad(op, grad):
    in_grad = icg_module.fftshift2d(grad)
    return [in_grad]

@ops.RegisterGradient("ActivationRBF")
def _ActivationRBFGrad(op, grad):
    rbf_prime = activation_prime_rbf(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    grad_w = icg_module.activation_rbf_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

@ops.RegisterGradient("ActivationPrimeRBF")
def _ActivationPrimeRBFGrad(op, grad):
    rbf_double_prime = icg_module.activation_double_prime_rbf(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_double_prime * grad
    grad_w = icg_module.activation_prime_rbf_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

@ops.RegisterGradient("ActivationBSpline")
def _ActivationCubicBSplineGrad(op, grad):
    rbf_prime = icg_module.activation_prime_b_spline(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    grad_w = icg_module.activation_b_spline_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

@ops.RegisterGradient("ActivationCubicBSpline")
def _ActivationCubicBSplineGrad(op, grad):
    rbf_prime = activation_prime_cubic_b_spline(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_prime * grad
    grad_w = icg_module.activation_cubic_b_spline_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

@ops.RegisterGradient("ActivationPrimeCubicBSpline")
def _ActivationPrimeCubicBSplineGrad(op, grad):
    rbf_double_prime = icg_module.activation_double_prime_cubic_b_spline(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = rbf_double_prime * grad
    grad_w = icg_module.activation_prime_cubic_b_spline_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

@ops.RegisterGradient("ActivationInterpolateLinear")
def _ActivationInterpolateLinearGrad(op, grad):
    act_prime = icg_module.activation_prime_interpolate_linear(op.inputs[0], op.inputs[1], op.get_attr("v_min"), op.get_attr("v_max"), op.get_attr("num_weights"), op.get_attr("feature_stride"))
    grad_x = act_prime * grad
    grad_w = icg_module.activation_interpolate_linear_grad_w(op.inputs[0], grad, op.get_attr("v_min"), op.get_attr("v_max"), op.inputs[1].shape[0], op.get_attr("num_weights"), op.get_attr("feature_stride"))
    return [grad_x, grad_w]

def conv2d_complex(u, k, strides=[1,1,1,1], padding='SAME', data_format='NHWC'):
    """ Complex 2d convolution with the same interface as `conv2d`.
    """
    conv_rr = tf.nn.conv2d(input=tf.math.real(u), filters=tf.math.real(k),  strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ii = tf.nn.conv2d(input=tf.math.imag(u), filters=tf.math.imag(k),  strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ri = tf.nn.conv2d(input=tf.math.real(u), filters=tf.math.imag(k), strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ir = tf.nn.conv2d(input=tf.math.imag(u), filters=tf.math.real(k), strides=strides, padding=padding,
                                     data_format=data_format)
    return tf.complex(conv_rr-conv_ii, conv_ri+conv_ir)

def conv2d_transpose_complex(u, k, output_shape, strides=[1,1,1,1], padding='SAME', data_format='NHWC'):
    """ Complex 2d transposed convolution with the same interface as `conv2d_transpose`.
    """
    convT_rr = tf.nn.conv2d_transpose(tf.math.real(u), tf.math.real(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ii = tf.nn.conv2d_transpose(tf.math.imag(u), tf.math.imag(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ri = tf.nn.conv2d_transpose(tf.math.real(u), tf.math.imag(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ir = tf.nn.conv2d_transpose(tf.math.imag(u), tf.math.real(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    return tf.complex(convT_rr+convT_ii, convT_ir-convT_ri)

def ifftc2d(inp):
    """ Centered inverse 2d Fourier transform, performed on axis (-1,-2).
    """
    shape = tf.shape(input=inp)
    numel = shape[-2]*shape[-1]
    scale = tf.sqrt(tf.cast(numel, tf.float32))

    out = fftshift2d(tf.signal.ifft2d(ifftshift2d(inp)))
    out = tf.complex(tf.math.real(out)*scale, tf.math.imag(out)*scale)
    return out

def fftc2d(inp):
    """ Centered 2d Fourier transform, performed on axis (-1,-2).
    """
    shape = tf.shape(input=inp)
    numel = shape[-2]*shape[-1]
    scale = 1.0 / tf.sqrt(tf.cast(numel, tf.float32))

    out = fftshift2d(tf.signal.fft2d(ifftshift2d(inp)))
    out = tf.complex(tf.math.real(out) * scale, tf.math.imag(out) * scale)
    return out
