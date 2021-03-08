// activations.cc
#define EIGEN_USE_THREADS
#include "activations.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/user_ops/common/definitions.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

constexpr auto ActivationCommonIOs = R"doc(
x: Input to the activation function. Must be at least a 2D tensor. The last
    dimmension defines the number of features.
w: Weights of the parametrized activation function. Must be a 2D tensor. The
    first dimmension defines the number of distinct activation functions and the
    last dimmension defines the number of used parameters and must match
    'num_weights'.
output: Output of the activation function.
)doc";


constexpr auto ActivationGradWCommonIOs = R"doc(
x: Input to the activation function.
grad_out: Gradient that should be backpropagted to the activation function
    weights.
output: Backpropagated gradient given the inputs and the output gradient.
)doc";

constexpr auto ActivationCommonAttrs = R"doc(
v_min: Defines the mean of the first basis function.
v_max: Defines the mean of the last basis function.
num_weights: Defines the number of used basis functions. All basis functions are
    equally distributed between 'v_min' and 'v_max'.
feature_stride: Defines the number of successive input features that share the
    same activation function. Thus, 'x.shape[-1] == w.shape[0] * feature_stride'
    must be true.
)doc";

constexpr auto Activation2dCommonIOs = R"doc(
x1: Input to the activation function. Must be at least a 2D tensor. The last
    dimension defines the number of features. Must have same shape as x2.
x2: Input to the activation function. Must be at least a 2D tensor. The last
      dimmension defines the number of features. Must have same shape as x1.
w: Weights of the parametrized activation function. Must be a 2D tensor. The
    first dimension defines the number of distinct activation functions and the
    last dimension defines the number of used parameters and must match
    'num_weights'.
output: Output of the activation function.
)doc";

constexpr auto Activation2dGradWCommonIOs = R"doc(
x1: First Input to the activation function. Must have same shape as x2.
x2: Second Input to the activation function. Must have same shape as x1.
grad_out: Gradient that should be backpropagated to the activation function
    weights.
output: Backpropagated gradient given the inputs and the output gradient.
)doc";

// Operator registration
// radial basis activation
REGISTER_OP("ActivationRBF")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Computes an activation function parameterized by weighted Gaussian radial basis
functions.)doc",
ActivationCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationRBFGradW")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("grad_out: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_channels: int >= 1")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int num_weights, num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      c->set_output(0, c->Matrix(num_channels, num_weights));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the Gaussian radial basis activation function
output to the corresponding weights.
)doc",
ActivationGradWCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationPrimeRBF")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the first derivative of a RBF activation function to the input 'x'.
Note that the derivative is computed by a weighted sum of the first derivative
of the basis functions.)doc",
ActivationCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationPrimeRBFGradW")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("grad_out: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_channels: int >= 1")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int num_weights, num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      c->set_output(0, c->Matrix(num_channels, num_weights));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the Gaussian radial basis activation function
derivative output to the corresponding weights.
)doc",
ActivationGradWCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationDoublePrimeRBF")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the second derivative of a RBF activation function to the input 'x'.
Note that the second derivative is computed by a weighted sum of the second
derivative of the bais functions.)doc",
ActivationCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationIntRBF")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the integral of an activation function to the input 'x'.
Note that the integral is computed by a weighted sum of the integrated basis
functions.)doc",
ActivationCommonIOs,
ActivationCommonAttrs));




// Operator registration
// radial basis activation 2d
REGISTER_OP("Activation2dRBF")
    .Attr("T: realnumbertype")
    .Input("x1: T")
    .Input("x2: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Computes an activation function parameterized by weighted Gaussian radial basis
functions.)doc",
Activation2dCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("Activation2dRBFGradW")
    .Attr("T: realnumbertype")
    .Input("x1: T")
    .Input("x2: T")
    .Input("grad_out: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_channels: int >= 1")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int num_weights, num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      c->set_output(0, c->Matrix(num_channels, num_weights));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the Gaussian radial basis activation function
output to the corresponding weights.
)doc",
Activation2dGradWCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("Activation2dX1PrimeRBF")
    .Attr("T: realnumbertype")
    .Input("x1: T")
    .Input("x2: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the first derivative of a RBF activation function to the input 'x1'.
Note that the derivative is computed by a weighted sum of the first derivative
of the basis functions.)doc",
Activation2dCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("Activation2dX2PrimeRBF")
    .Attr("T: realnumbertype")
    .Input("x1: T")
    .Input("x2: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the first derivative of a RBF activation function to the input 'x2'.
Note that the derivative is computed by a weighted sum of the first derivative
of the basis functions.)doc",
Activation2dCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("Activation2dX1PrimeRBFGradW")
    .Attr("T: realnumbertype")
    .Input("x1: T")
    .Input("x2: T")
    .Input("grad_out: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_channels: int >= 1")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int num_weights, num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      c->set_output(0, c->Matrix(num_channels, num_weights));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the Gaussian radial basis activation function
derivative output to the corresponding weights.
)doc",
Activation2dGradWCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("Activation2dX2PrimeRBFGradW")
    .Attr("T: realnumbertype")
    .Input("x1: T")
    .Input("x2: T")
    .Input("grad_out: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_channels: int >= 1")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int num_weights, num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      c->set_output(0, c->Matrix(num_channels, num_weights));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the Gaussian radial basis activation function
derivative output to the corresponding weights.
)doc",
Activation2dGradWCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("Activation2dX1DoublePrimeRBF")
    .Attr("T: realnumbertype")
    .Input("x1: T")
    .Input("x2: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the second derivative of a RBF activation function to the input 'x1'.
Note that the second derivative is computed by a weighted sum of the second
derivative of the bais functions.)doc",
Activation2dCommonIOs,
ActivationCommonAttrs));


REGISTER_OP("Activation2dX2DoublePrimeRBF")
    .Attr("T: realnumbertype")
    .Input("x1: T")
    .Input("x2: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the second derivative of a RBF activation function to the input 'x2'.
Note that the second derivative is computed by a weighted sum of the second
derivative of the bais functions.)doc",
Activation2dCommonIOs,
ActivationCommonAttrs));


REGISTER_OP("Activation2dIntRBF")
    .Attr("T: realnumbertype")
    .Input("x1: T")
    .Input("x2: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the integral of an activation function to the inputs 'x1,x2'.
Note that the integral is computed by a weighted sum of the integrated basis
functions.)doc",
Activation2dCommonIOs,
ActivationCommonAttrs));


// linear b-spline activation

REGISTER_OP("ActivationBSpline")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Computes an activation function parameterized by weighted linear B-spline basis
functions.)doc",
ActivationCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationBSplineGradW")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("grad_out: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_channels: int >= 1")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int num_weights, num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      c->set_output(0, c->Matrix(num_channels, num_weights));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the linear B-spline basis activation function
output to the corresponding weights.
)doc",
ActivationGradWCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationPrimeBSpline")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the first derivative of a linear B-spline activation function to the
input 'x'. Note that the derivative is computed by a weighted sum of the first
derivative of the basis functions.)doc",
ActivationCommonIOs,
ActivationCommonAttrs));

// cubic b-spline activation
REGISTER_OP("ActivationCubicBSpline")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Computes an activation function parameterized by weighted cubic B-spline basis
functions.)doc",
ActivationCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationCubicBSplineGradW")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("grad_out: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_channels: int >= 1")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int num_weights, num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      c->set_output(0, c->Matrix(num_channels, num_weights));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the cubic B-spline basis activation function
output to the corresponding weights.
)doc",
ActivationGradWCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationPrimeCubicBSpline")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the first derivative of a cubic B-spline activation function to the
input 'x'. Note that the derivative is computed by a weighted sum of the first
derivative of the basis functions.)doc",
ActivationCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationPrimeCubicBSplineGradW")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("grad_out: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_channels: int >= 1")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int num_weights, num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      c->set_output(0, c->Matrix(num_channels, num_weights));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the cubic B-spline radial basis activation
function derivative output to the corresponding weights.
)doc",
ActivationGradWCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationDoublePrimeCubicBSpline")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Applies the second derivative of a cubic B-spline activation function to the
input 'x'. Note that the second derivative is computed by a weighted sum of the
second derivative of the basis functions.)doc",
ActivationCommonIOs,
ActivationCommonAttrs));

// Linear interpolation activation
REGISTER_OP("ActivationInterpolateLinear")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Computes an activation function parameterized by a linear interpolation between
weighted dirac delta basis functions.)doc",
ActivationCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationPrimeInterpolateLinear")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the output to the input of the linear
interpolation activation operator.
)doc",
ActivationCommonIOs,
ActivationCommonAttrs));

REGISTER_OP("ActivationInterpolateLinearGradW")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("grad_out: T")
    .Output("output: T")
    .Attr("v_min: float")
    .Attr("v_max: float")
    .Attr("num_channels: int >= 1")
    .Attr("num_weights: int >= 1")
    .Attr("feature_stride: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int num_weights, num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_weights", &num_weights));
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      c->set_output(0, c->Matrix(num_channels, num_weights));
      return Status::OK();
    })
    .Doc(strings::StrCat(R"doc(
Backpropagates the gradient from the output to the weights of the linear
interpolation activation operator.
)doc",
ActivationGradWCommonIOs,
ActivationCommonAttrs));

const unsigned int max_num_weights = 128;


/**
 * Activation operator interface
 * Defines the IOs, attributes and performs size checks.
 */
template<typename Device, typename T>
class ActivationBaseOp : public OpKernel {
  public:
    explicit ActivationBaseOp(OpKernelConstruction* context) : OpKernel(context)
    {
      // Get attributes
      float v_tmp;
      OP_REQUIRES_OK(context, context->GetAttr("v_min", &v_tmp));
      v_min_ = static_cast<T>(v_tmp);
      OP_REQUIRES_OK(context, context->GetAttr("v_max", &v_tmp));
      v_max_ = static_cast<T>(v_tmp);
      OP_REQUIRES_OK(context, context->GetAttr("num_weights", &num_weights_));
      OP_REQUIRES_OK(context, context->GetAttr("feature_stride", &feature_stride_));
    }

    virtual ~ActivationBaseOp() {};

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& x_tensor = context->input(0);
      const Tensor& weight_tensor = context->input(1);

      // Check v_min and v_max
      OP_REQUIRES(context, v_min_ < v_max_,
                  errors::Unimplemented("v_min is not smaller than v_max! ",
                                        "v_min: ", v_min_, " > v_max: ", v_max_));

      // Check the dimensionality and size of the weights
      OP_REQUIRES(context, weight_tensor.dims() == 2,
                  errors::Unimplemented("Expected a 2d Tensor, got ",
                                        weight_tensor.dims(), "d."));
      auto weight_shape = weight_tensor.shape();
      OP_REQUIRES(context, num_weights_ < max_num_weights,
                  errors::Unimplemented("Number of weights must be samller than ", max_num_weights,
                                        "Got: ", num_weights_, "!"));
      OP_REQUIRES(context, weight_shape.dim_size(1) == num_weights_,
                  errors::Unimplemented("Second dimension of weights must be ",
                                        num_weights_, ", got ", weight_shape.dim_size(1)));
      // Check that the number of channels is identical
      auto x_shape = x_tensor.shape();
      auto dims = x_tensor.dims();
      OP_REQUIRES(context, x_shape.dim_size(dims-1) == weight_shape.dim_size(0) * feature_stride_,
                  errors::Unimplemented("'x.shape[-1] = w.shape[0]*feature_stride' is not fulfilled!",
                  "Got ", x_shape.dim_size(dims-1), " != ", weight_shape.dim_size(0)*feature_stride_));

      // Create an output tensor
      Tensor *output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, x_shape,
                                                       &output_tensor));
      // Do the computation
      auto out_tensor_map = output_tensor->flat_inner_dims<T,2>();
      ApplyActivation(context, x_tensor.flat_inner_dims<T,2>(),
                      weight_tensor.tensor<T,2>(), out_tensor_map);
    }

  private:
    virtual void ApplyActivation(OpKernelContext *context,
                                 const typename Tensor2<T>::ConstTensor &x,
                                 const typename Tensor2<T>::ConstTensor &w,
                                 typename Tensor2<T>::Tensor &out) = 0;

  protected:
    T v_min_, v_max_;
    int num_weights_;
    int feature_stride_;
};

/**
 * Activation gradient w.r.t. weights interface
 * Defines the IOs, attributes and performs size checks.
 */
template<typename Device, typename T>
class ActivationBaseGradWOp : public OpKernel {
  public:
    explicit ActivationBaseGradWOp(OpKernelConstruction* context) : OpKernel(context)
    {
      // Get attributes
      float v_tmp;
      OP_REQUIRES_OK(context, context->GetAttr("v_min", &v_tmp));
      v_min_ = static_cast<T>(v_tmp);
      OP_REQUIRES_OK(context, context->GetAttr("v_max", &v_tmp));
      v_max_ = static_cast<T>(v_tmp);
      OP_REQUIRES_OK(context, context->GetAttr("num_channels", &num_channels_));
      OP_REQUIRES_OK(context, context->GetAttr("num_weights", &num_weights_));
      OP_REQUIRES_OK(context, context->GetAttr("feature_stride", &feature_stride_));
    }

    virtual ~ActivationBaseGradWOp() {};

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& x_tensor = context->input(0);
      const Tensor& grad_out_tensor = context->input(1);

      // Check v_min and v_max
      OP_REQUIRES(context, v_min_ < v_max_,
                  errors::Unimplemented("v_min is not smaller than v_max! ",
                                        "v_min: ", v_min_, " > v_max: ", v_max_));

      // Check the dimensionality and size of the weights
      OP_REQUIRES(context, num_weights_ < max_num_weights,
                  errors::Unimplemented("Number of weights must be samller than ", max_num_weights,
                                        "Got: ", num_weights_, "!"));
      // Check that the number of channels is identical
      auto x_shape = x_tensor.shape();
      auto dims = x_tensor.dims();
      OP_REQUIRES(context, x_shape.dim_size(dims-1) == num_channels_ * feature_stride_,
                  errors::Unimplemented("'x.shape[-1] = num_channels_*feature_stride' is not fulfilled!",
                  "Got ", x_shape.dim_size(dims-1), " != ", num_channels_*feature_stride_));

      // Check that x and gradient have same shape
      OP_REQUIRES(context, x_shape == grad_out_tensor.shape(),
                  errors::Unimplemented("Shape of input and gradient do not match!"));

      // Create the output tensors
      TensorShape grad_w_shape({num_channels_, num_weights_});
      Tensor *grad_w_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, grad_w_shape,
                                                       &grad_w_tensor));

      // Do the computation
      auto grad_w_tensor_map = grad_w_tensor->tensor<T,2>();

      ComputeGradient(context, x_tensor.flat_inner_dims<T,2>(),
                      grad_out_tensor.flat_inner_dims<T,2>(),
                      grad_w_tensor_map);
    }

  private:
    virtual void ComputeGradient(OpKernelContext *context,
                                 const typename Tensor2<T>::ConstTensor &x,
                                 const typename Tensor2<T>::ConstTensor &grad_out,
                                 typename Tensor2<T>::Tensor &grad_w) = 0;

  protected:
    T v_min_, v_max_;
    int num_weights_;
    int num_channels_;
    int feature_stride_;
};

// Radial basis function activation
template <typename T, tficg::DerivativeOrder N>
struct ActivationRBFFunctor<CPUDevice, T, N> {
  void operator()(OpKernelContext *context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &w,
                  typename Tensor2<T>::Tensor &out,
                  T v_min, T v_max, int feature_stride) {

    const unsigned int n_w = w.dimensions()[1];

    const T delta = (v_max - v_min) / (n_w - 1);
    const T stddev_2 = pow((v_max - v_min) / n_w, 2);

    for(int idc = 0; idc < x.dimensions()[1]; idc++)
    {
      for(int idx = 0; idx < x.dimensions()[0]; idx++)
      {
        T output = 0.0;
        for(int idw = 0; idw < n_w; idw++)
        {
          const T dist = x(idx, idc) - (v_min + idw * delta);
          const T val = exp(-(dist*dist) / (2*stddev_2));
          const T inner_der = -dist / stddev_2;

          switch(N)
          {
            case tficg::DO_ZERO:
              output += w(idc, idw) * val;
            break;
            case tficg::DO_FIRST:
              output += w(idc, idw) * inner_der * val;
            break;
            case tficg::DO_SECOND:
              output += w(idc, idw) * ((-1 / stddev_2) + inner_der * inner_der) * val;
            break;
          }
        }
        out(idx, idc) = output;
      }
    }
  }
};

template<typename Device, typename T, tficg::DerivativeOrder N>
class ActivationRBFOp : public ActivationBaseOp<Device, T> {
  public:
    explicit ActivationRBFOp(OpKernelConstruction* context) :
      ActivationBaseOp<Device, T>(context)
    {
    }

  private:
    void ApplyActivation(OpKernelContext* context,
                         const typename Tensor2<T>::ConstTensor &x,
                         const typename Tensor2<T>::ConstTensor &w,
                         typename Tensor2<T>::Tensor &out) override
    {
      ActivationRBFFunctor<Device, T, N>()(context,
        x, w, out, this->v_min_, this->v_max_, this->feature_stride_);
    }
};

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationRBF") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFOp<CPUDevice, T, tficg::DO_ZERO>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationRBF") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFOp<GPUDevice, T, tficg::DO_ZERO>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeRBF") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFOp<CPUDevice, T, tficg::DO_FIRST>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeRBF") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFOp<GPUDevice, T, tficg::DO_FIRST>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationDoublePrimeRBF") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFOp<CPUDevice, T, tficg::DO_SECOND>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationDoublePrimeRBF") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFOp<GPUDevice, T, tficg::DO_SECOND>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationIntRBF") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFOp<GPUDevice, T, tficg::DO_INT>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

// Gradient Implementation
template <typename T, tficg::DerivativeOrder N>
struct ActivationRBFGradWFunctor<CPUDevice, T, N> {
  void operator()(OpKernelContext* context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &grad_out,
                  typename Tensor2<T>::Tensor &grad_w,
                  T v_min, T v_max, int feature_stride) {
    // initialize the gradient w
    grad_w.setZero();

    //number of weights
    const unsigned int n_w = grad_w.dimensions()[1];

    const T delta = (v_max - v_min) / (n_w - 1);
    const T stddev_2 = pow((v_max - v_min) / n_w, 2);

    for(int idc = 0; idc < x.dimensions()[1]; idc++)
    {
      for(int idx = 0; idx < x.dimensions()[0]; idx++)
      {
        for(int idw = 0; idw < n_w; idw++)
        {
          const T dist = x(idx, idc) - (v_min + idw * delta);
          const T val = exp(-(dist * dist) / (2 * stddev_2));
          const T inner_der = -dist / stddev_2;

          switch(N)
          {
            case tficg::DO_ZERO:
              grad_w(idc, idw) += val * grad_out(idx, idc);
            break;
            case tficg::DO_FIRST:
              grad_w(idc, idw) += val * inner_der * grad_out(idx, idc);
            break;
            case tficg::DO_SECOND:
              grad_w(idc, idw) += val * ((-1 / stddev_2) + inner_der * inner_der) * grad_out(idx, idc);
            break;
          }
        }
      }
    }
  }
};

template<typename Device, typename T, tficg::DerivativeOrder N>
class ActivationRBFGradWOp : public ActivationBaseGradWOp<Device, T> {
  public:
    explicit ActivationRBFGradWOp(OpKernelConstruction* context) :
    ActivationBaseGradWOp<Device, T>(context)
    {
    }

  private:
    void ComputeGradient(OpKernelContext *context,
                         const typename Tensor2<T>::ConstTensor &x,
                         const typename Tensor2<T>::ConstTensor &grad_out,
                         typename Tensor2<T>::Tensor &grad_w) override
    {
      ActivationRBFGradWFunctor<Device, T, N>()(context, x, grad_out, grad_w,
                this->v_min_, this->v_max_, this->feature_stride_);
    }
};

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationRBFGradW") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFGradWOp<CPUDevice, T, tficg::DO_ZERO>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationRBFGradW") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFGradWOp<GPUDevice, T, tficg::DO_ZERO>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeRBFGradW") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFGradWOp<CPUDevice, T, tficg::DO_FIRST>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeRBFGradW") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationRBFGradWOp<GPUDevice, T, tficg::DO_FIRST>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif


/**
 * Activation 2d operator interface
 * Defines the IOs, attributes and performs size checks.
 */
template<typename Device, typename T>
class Activation2dBaseOp : public OpKernel {
  public:
    explicit Activation2dBaseOp(OpKernelConstruction* context) : OpKernel(context)
    {
      // Get attributes
      float v_tmp;
      OP_REQUIRES_OK(context, context->GetAttr("v_min", &v_tmp));
      v_min_ = static_cast<T>(v_tmp);
      OP_REQUIRES_OK(context, context->GetAttr("v_max", &v_tmp));
      v_max_ = static_cast<T>(v_tmp);
      OP_REQUIRES_OK(context, context->GetAttr("num_weights", &num_weights_));
      OP_REQUIRES_OK(context, context->GetAttr("feature_stride", &feature_stride_));
    }

    virtual ~Activation2dBaseOp() {};

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& x1_tensor = context->input(0);
      const Tensor& x2_tensor = context->input(1);
      const Tensor& weight_tensor = context->input(2);

      // Check v_min and v_max
      OP_REQUIRES(context, v_min_ < v_max_,
                  errors::Unimplemented("v_min is not smaller than v_max! ",
                                        "v_min: ", v_min_, " > v_max: ", v_max_));

      // Check the dimensionality and size of the weights
      OP_REQUIRES(context, weight_tensor.dims() == 2,
                  errors::Unimplemented("Expected a 2d Tensor, got ",
                                        weight_tensor.dims(), "d."));
      auto weight_shape = weight_tensor.shape();
      OP_REQUIRES(context, num_weights_ < max_num_weights,
                  errors::Unimplemented("Number of weights must be smaller than ", max_num_weights,
                                        "Got: ", num_weights_, "!"));
      //Check that number of weights are a perfect square
      long double sr = sqrt(num_weights_);
      OP_REQUIRES(context, sr == floor(sr),
                   errors::Unimplemented("The number of weights for a 2d activation must be a perfect square. Got: ",
                                          num_weights_));
      OP_REQUIRES(context, weight_shape.dim_size(1) == num_weights_,
                  errors::Unimplemented("Second dimension of weights must be ",
                                        num_weights_, ", got ", weight_shape.dim_size(1)));
      // Check that the number of channels is identical
      auto x1_shape = x1_tensor.shape();
      auto dims = x1_tensor.dims();
      OP_REQUIRES(context, x1_shape.dim_size(dims-1) == weight_shape.dim_size(0) * feature_stride_,
                  errors::Unimplemented("'x.shape[-1] = w.shape[0]*feature_stride' is not fulfilled!",
                  "Got ", x1_shape.dim_size(dims-1), " != ", weight_shape.dim_size(0)*feature_stride_));

      auto x2_shape = x2_tensor.shape();
      OP_REQUIRES(context, x2_shape.dim_size(dims-1) == weight_shape.dim_size(0) * feature_stride_,
                  errors::Unimplemented("'x.shape[-1] = w.shape[0]*feature_stride' is not fulfilled!",
                  "Got ", x2_shape.dim_size(dims-1), " != ", weight_shape.dim_size(0)*feature_stride_));

      // Create an output tensor
      Tensor *output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, x1_shape,
                                                       &output_tensor));
      // Do the computation
      auto out_tensor_map = output_tensor->flat_inner_dims<T,2>();
      ApplyActivation2d(context, x1_tensor.flat_inner_dims<T,2>(), x2_tensor.flat_inner_dims<T,2>(),
                      weight_tensor.tensor<T,2>(), out_tensor_map);
    }

  private:
    virtual void ApplyActivation2d(OpKernelContext *context,
                                 const typename Tensor2<T>::ConstTensor &x1,
                                 const typename Tensor2<T>::ConstTensor &x2,
                                 const typename Tensor2<T>::ConstTensor &w,
                                 typename Tensor2<T>::Tensor &out) = 0;

  protected:
    T v_min_, v_max_;
    int num_weights_;
    int feature_stride_;
};

/**
 * Activation2d gradient w.r.t. weights interface
 * Defines the IOs, attributes and performs size checks.
 */
template<typename Device, typename T>
class Activation2dBaseGradWOp : public OpKernel {
  public:
    explicit Activation2dBaseGradWOp(OpKernelConstruction* context) : OpKernel(context)
    {
      // Get attributes
      float v_tmp;
      OP_REQUIRES_OK(context, context->GetAttr("v_min", &v_tmp));
      v_min_ = static_cast<T>(v_tmp);
      OP_REQUIRES_OK(context, context->GetAttr("v_max", &v_tmp));
      v_max_ = static_cast<T>(v_tmp);
      OP_REQUIRES_OK(context, context->GetAttr("num_channels", &num_channels_));
      OP_REQUIRES_OK(context, context->GetAttr("num_weights", &num_weights_));
      OP_REQUIRES_OK(context, context->GetAttr("feature_stride", &feature_stride_));
    }

    virtual ~Activation2dBaseGradWOp() {};

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& x1_tensor = context->input(0);
      const Tensor& x2_tensor = context->input(1);
      const Tensor& grad_out_tensor = context->input(2);

      // Check v_min and v_max
      OP_REQUIRES(context, v_min_ < v_max_,
                  errors::Unimplemented("v_min is not smaller than v_max! ",
                                        "v_min: ", v_min_, " > v_max: ", v_max_));

      // Check the dimensionality and size of the weights
      OP_REQUIRES(context, num_weights_ < max_num_weights,
                  errors::Unimplemented("Number of weights must be samller than ", max_num_weights,
                                        "Got: ", num_weights_, "!"));
      //Check that number of weights are a perfect square
      long double sr = sqrt(num_weights_);
      OP_REQUIRES(context, sr == floor(sr),
                   errors::Unimplemented("The number of weights for a 2d activation must be a perfect square. Got: ",
                                          num_weights_));
      // Check that the number of channels is identical
      auto x1_shape = x1_tensor.shape();
      auto dims = x1_tensor.dims();
      OP_REQUIRES(context, x1_shape.dim_size(dims-1) == num_channels_ * feature_stride_,
                  errors::Unimplemented("'x.shape[-1] = num_channels_*feature_stride' is not fulfilled!",
                  "Got ", x1_shape.dim_size(dims-1), " != ", num_channels_*feature_stride_));

      auto x2_shape = x2_tensor.shape();
      OP_REQUIRES(context, x2_shape.dim_size(dims-1) == num_channels_ * feature_stride_,
                  errors::Unimplemented("'x.shape[-1] = num_channels_*feature_stride' is not fulfilled!",
                  "Got ", x2_shape.dim_size(dims-1), " != ", num_channels_*feature_stride_));

      // Check that x and gradient have same shape
      OP_REQUIRES(context, x1_shape == grad_out_tensor.shape(),
                  errors::Unimplemented("Shape of input 0 and gradient do not match!"));
      OP_REQUIRES(context, x2_shape == grad_out_tensor.shape(),
                  errors::Unimplemented("Shape of input 1 and gradient do not match!"));

      // Create the output tensors
      TensorShape grad_w_shape({num_channels_, num_weights_});
      Tensor *grad_w_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, grad_w_shape,
                                                       &grad_w_tensor));

      // Do the computation
      auto grad_w_tensor_map = grad_w_tensor->tensor<T,2>();

      ComputeGradient2d(context, x1_tensor.flat_inner_dims<T,2>(),
                      x2_tensor.flat_inner_dims<T,2>(),
                      grad_out_tensor.flat_inner_dims<T,2>(),
                      grad_w_tensor_map);
    }

  private:
    virtual void ComputeGradient2d(OpKernelContext *context,
                                 const typename Tensor2<T>::ConstTensor &x1,
                                 const typename Tensor2<T>::ConstTensor &x2,
                                 const typename Tensor2<T>::ConstTensor &grad_out,
                                 typename Tensor2<T>::Tensor &grad_w) = 0;

  protected:
    T v_min_, v_max_;
    int num_weights_;
    int num_channels_;
    int feature_stride_;
};


// Radial basis function activation 2d
template <typename T, tficg::DerivativeOrder N>
struct Activation2dRBFFunctor<CPUDevice, T, N> {
  void operator()(OpKernelContext *context,
                  const typename Tensor2<T>::ConstTensor &x1,
                  const typename Tensor2<T>::ConstTensor &x2,
                  const typename Tensor2<T>::ConstTensor &w,
                  typename Tensor2<T>::Tensor &out,
                  T v_min, T v_max, int feature_stride) {

    const unsigned int n_w = w.dimensions()[1];
    const unsigned int n_wd = int(sqrt(w.dimensions()[1]));

    const T delta = (v_max - v_min) / (n_wd - 1);
    const T stddev_2 = pow((v_max - v_min) / n_wd, 2);

    for(int idc = 0; idc < x1.dimensions()[1]; idc++)
    {
      for(int idx = 0; idx < x1.dimensions()[0]; idx++)
      {
        T output = 0.0;
        for(int idwx = 0; idwx < n_wd; idwx++)
        {
          for(int idwy = 0; idwy < n_wd; idwy++)
          {
            const T dist1 = x1(idx,idc) - (v_min + idwx * delta);
            const T dist2 = x2(idx, idc) - (v_min + idwy * delta);
            const T val = exp(-(dist1*dist1+dist2*dist2) / (2*stddev_2));
            const T inner_der1 = -dist1 / stddev_2;
            const T inner_der2 = -dist2 / stddev_2;

            switch(N)
            {
              case tficg::DO_ZERO:
                output += w(idc, idwx*n_wd + idwy) * val;
              break;
              case tficg::DO_FIRSTX1:
                output+= w(idc, idwx*n_wd + idwy) * inner_der1 * val;
              break;
              case tficg::DO_SECONDX1:
                output+= w(idc, idwx*n_wd + idwy) * ((-1 / stddev_2) + inner_der1 * inner_der1) * val;
              break;
              case tficg::DO_FIRSTX2:
                output+= w(idc, idwx*n_wd + idwy) * inner_der2 * val;
              break;
              case tficg::DO_SECONDX2:
                output+= w(idc, idwx*n_wd + idwy) * ((-1 / stddev_2) + inner_der2 * inner_der2) * val;
              break;
            }
          }
        }
        out(idx, idc) = output;
      }
    }
  }
};

template<typename Device, typename T, tficg::DerivativeOrder N>
class Activation2dRBFOp : public Activation2dBaseOp<Device, T> {
  public:
    explicit Activation2dRBFOp(OpKernelConstruction* context) :
      Activation2dBaseOp<Device, T>(context)
    {
    }

  private:
    void ApplyActivation2d(OpKernelContext* context,
                         const typename Tensor2<T>::ConstTensor &x1,
                         const typename Tensor2<T>::ConstTensor &x2,
                         const typename Tensor2<T>::ConstTensor &w,
                         typename Tensor2<T>::Tensor &out) override
    {
      Activation2dRBFFunctor<Device, T, N>()(context,
        x1, x2, w, out, this->v_min_, this->v_max_, this->feature_stride_);
    }
};

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dRBF") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<CPUDevice, T, tficg::DO_ZERO>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dRBF") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<GPUDevice, T, tficg::DO_ZERO>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX1PrimeRBF") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<CPUDevice, T, tficg::DO_FIRSTX1>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX2PrimeRBF") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<CPUDevice, T, tficg::DO_FIRSTX2>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX1PrimeRBF") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<GPUDevice, T, tficg::DO_FIRSTX1>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX2PrimeRBF") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<GPUDevice, T, tficg::DO_FIRSTX2>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX1DoublePrimeRBF") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<CPUDevice, T, tficg::DO_SECONDX1>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX2DoublePrimeRBF") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<CPUDevice, T, tficg::DO_SECONDX2>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX1DoublePrimeRBF") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<GPUDevice, T, tficg::DO_SECONDX1>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX2DoublePrimeRBF") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<GPUDevice, T, tficg::DO_SECONDX2>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dIntRBF") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFOp<GPUDevice, T, tficg::DO_INT>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

// Gradient Implementation
template <typename T, tficg::DerivativeOrder N>
struct Activation2dRBFGradWFunctor<CPUDevice, T, N> {
  void operator()(OpKernelContext* context,
                  const typename Tensor2<T>::ConstTensor &x1,
                  const typename Tensor2<T>::ConstTensor &x2,
                  const typename Tensor2<T>::ConstTensor &grad_out,
                  typename Tensor2<T>::Tensor &grad_w,
                  T v_min, T v_max, int feature_stride) {
    // initialize the gradient w
    grad_w.setZero();

    //number of weights
    const unsigned int n_wd = int(sqrt(grad_w.dimensions()[1]));

    const T delta = (v_max - v_min) / (n_wd - 1);
    const T stddev_2 = pow((v_max - v_min) / n_wd, 2);

    for(int idc = 0; idc < x1.dimensions()[1]; idc++)
    {
      for(int idx = 0; idx < x1.dimensions()[0]; idx++)
      {
        for(int idwx = 0; idwx < n_wd; idwx++)
        {
          for(int idwy = 0; idwy < n_wd; idwy++)
          {
            const T dist1 = x1(idx,idc) - (v_min + idwx * delta);
            const T dist2 = x2(idx, idc) - (v_min + idwy * delta);
            const T val = exp(-(dist1*dist1+dist2*dist2) / (2*stddev_2));
            const T inner_der1 = -dist1 / stddev_2;
            const T inner_der2 = -dist2 / stddev_2;

            switch(N)
            {
              case tficg::DO_ZERO:
                grad_w(idc, idwx*n_wd + idwy) += val * grad_out(idx, idc);
              break;
              case tficg::DO_FIRSTX1:
                grad_w(idc, idwx*n_wd + idwy) += val * inner_der1 * grad_out(idx, idc);
              break;
              case tficg::DO_SECONDX1:
                grad_w(idc, idwx*n_wd + idwy) += val * ((-1 / stddev_2) + inner_der1 * inner_der1) * grad_out(idx, idc);
              break;
              case tficg::DO_FIRSTX2:
                grad_w(idc, idwx*n_wd + idwy) += val * inner_der2 * grad_out(idx, idc);
              break;
              case tficg::DO_SECONDX2:
                grad_w(idc, idwx*n_wd + idwy) += val * ((-1 / stddev_2) + inner_der2 * inner_der2) * grad_out(idx, idc);
              break;
            }
          }
        }
      }
    }
  }
};


template<typename Device, typename T, tficg::DerivativeOrder N>
class Activation2dRBFGradWOp : public Activation2dBaseGradWOp<Device, T> {
  public:
    explicit Activation2dRBFGradWOp(OpKernelConstruction* context) :
    Activation2dBaseGradWOp<Device, T>(context)
    {
    }

  private:
    void ComputeGradient2d(OpKernelContext *context,
                         const typename Tensor2<T>::ConstTensor &x1,
                         const typename Tensor2<T>::ConstTensor &x2,
                         const typename Tensor2<T>::ConstTensor &grad_out,
                         typename Tensor2<T>::Tensor &grad_w) override
    {
      Activation2dRBFGradWFunctor<Device, T, N>()(context, x1, x2, grad_out, grad_w,
                this->v_min_, this->v_max_, this->feature_stride_);
    }
};

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dRBFGradW") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFGradWOp<CPUDevice, T, tficg::DO_ZERO>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dnRBFGradW") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFGradWOp<GPUDevice, T, tficg::DO_ZERO>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX1PrimeRBFGradW") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFGradWOp<CPUDevice, T, tficg::DO_FIRSTX1>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX2PrimeRBFGradW") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFGradWOp<CPUDevice, T, tficg::DO_FIRSTX2>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX1PrimeRBFGradW") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFGradWOp<GPUDevice, T, tficg::DO_FIRSTX1>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Activation2dX2PrimeRBFGradW") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    Activation2dRBFGradWOp<GPUDevice, T, tficg::DO_FIRSTX2>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif



// linear b-spline activation

/**
 * definition of B-spline basis function
 */
template<typename T>
inline T b_spline_linear(T x)
{
  x = fabs(x);

  if (x < 1.0f) return 1.f - x;
  else return 0.0f;
}

/**
 * first derivative of a B-spline basis function
 */
template<typename T>
inline T b_spline_linear_prime(T x)
{
  if (-1.0f < x && x < 0.f) return 1.f;
  else if (0.f < x && x < 1.f) return -1.f;
  else return 0.f;
}


// b-spline activation

/**
 * definition of B-spline basis function
 */
template<typename T>
inline T b_spline_cubic(T x)
{
  x = fabs(x);
  const T a = 2.0f - x;

  if (x < 1.0f) return 2.0f/3.0f - 0.5f*x*x*a;
  else if (x < 2.0f) return a*a*a / 6.0f;
  else return 0.0f;
}

/**
 * first derivative of a B-spline basis function
 */
template<typename T>
inline T b_spline_cubic_prime(T x)
{
  if (-2.0f < x && x <= -1.0f) return 0.5f*x*x + 2.0f*x + 2.0f;
  else if (-1.0f < x && x <= 0.0f) return -1.5f*x*x - 2.0f*x;
  else if ( 0.0f < x && x <= 1.0f) return  1.5f*x*x - 2.0f*x;
  else if ( 1.0f < x && x <  2.0f) return -0.5f*x*x + 2.0f*x - 2.0f;
  else return 0.0f;
}

/**
 * second derivative of a B-spline basis function
 */
template<typename T>
inline T b_spline_cubic_double_prime(T x)
{
  x = fabs(x);

  if (x < 1.0f) return 3.0f*x - 2.0f;
  else if (x < 2.0f) return 2.0f - x;
  else return 0.0f;
}

template <typename T, tficg::SplineOrder S, tficg::DerivativeOrder N>
struct ActivationBSplineFunctor<CPUDevice, T, S, N> {
  void operator()(OpKernelContext *context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &w,
                  typename Tensor2<T>::Tensor &out,
                  T v_min, T v_max, int feature_stride) {

    //number of weights
    const unsigned int n_w = w.dimensions()[1];
    //equidistance
    const T d = (v_max - v_min) / (n_w - 1);
    //input_size
    const unsigned int n_x = x.dimensions()[0];
    //number of channels
    const unsigned int n_c = x.dimensions()[1];

    for(int idc = 0; idc < n_c; idc++)
    {
      for(int idx = 0; idx < n_x; idx++)
      {
        const T pos = (x(idx, idc) - v_min) / d;
        const T pos_f = (std::floor(pos));
        const T alpha = pos - pos_f;
        int idw = static_cast<int>(pos_f);

        if(pos < -2.0 || pos > (n_w + 1))
        {
          out(idx, idc) = 0;
        }
        else
        {
          T output = 0;
          for(int dx = -1; dx <= 2; dx++)
          {
            T b_spline = 0;
            switch(N)
            {
              case tficg::DO_ZERO:
                switch(S)
                {
                  case tficg::SO_LINEAR:
                    b_spline = b_spline_linear<T>(dx - alpha);
                  break;
                  case tficg::SO_CUBIC:
                    b_spline = b_spline_cubic<T>(dx - alpha);
                  break;
                }
              break;
              case tficg::DO_FIRST:
                switch(S)
                {
                  case tficg::SO_LINEAR:
                    b_spline = b_spline_linear_prime<T>(dx - alpha) / (-d);
                  break;
                  case tficg::SO_CUBIC:
                    b_spline = b_spline_cubic_prime<T>(dx - alpha) / (-d);
                  break;
                }
              break;
              case tficg::DO_SECOND:
                b_spline = b_spline_cubic_double_prime<T>(dx - alpha) / (d * d);
              break;
            }
            if ((idw + dx) >= 0 && (idw + dx) < n_w)
              output += b_spline * w(idc, idw + dx);
          }
          out(idx, idc) = output;
        }
      }
    }
  }
};

template<typename Device, typename T, tficg::SplineOrder S, tficg::DerivativeOrder N>
class ActivationBSplineOp : public ActivationBaseOp<Device, T> {
  public:
    explicit ActivationBSplineOp(OpKernelConstruction* context) :
      ActivationBaseOp<Device, T>(context)
    {
    }

  private:
    void ApplyActivation(OpKernelContext* context,
                         const typename Tensor2<T>::ConstTensor &x,
                         const typename Tensor2<T>::ConstTensor &w,
                         typename Tensor2<T>::Tensor &out) override
    {
      ActivationBSplineFunctor<Device, T, S, N>()(context,
        x, w, out, this->v_min_, this->v_max_, this->feature_stride_);
    }
};

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationBSpline") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineOp<CPUDevice, T, tficg::SO_LINEAR, tficg::DO_ZERO>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationBSpline") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineOp<GPUDevice, T, tficg::SO_LINEAR, tficg::DO_ZERO>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeBSpline") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineOp<CPUDevice, T, tficg::SO_LINEAR, tficg::DO_FIRST>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeBSpline") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineOp<GPUDevice, T, tficg::SO_LINEAR, tficg::DO_FIRST>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationCubicBSpline") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineOp<CPUDevice, T, tficg::SO_CUBIC, tficg::DO_ZERO>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationCubicBSpline") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineOp<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_ZERO>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeCubicBSpline") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineOp<CPUDevice, T, tficg::SO_CUBIC, tficg::DO_FIRST>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeCubicBSpline") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineOp<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_FIRST>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationDoublePrimeCubicBSpline") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineOp<CPUDevice, T, tficg::SO_CUBIC, tficg::DO_SECOND>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationDoublePrimeCubicBSpline") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineOp<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_SECOND>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

template <typename T, tficg::SplineOrder S, tficg::DerivativeOrder N>
struct ActivationBSplineGradWFunctor<CPUDevice, T, S, N> {
  void operator()(OpKernelContext* context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &grad_out,
                  typename Tensor2<T>::Tensor &grad_w,
                  T v_min, T v_max, int feature_stride) {
    // initialize the gradient w
    grad_w.setZero();

    //number of weights
    const unsigned int n_w = grad_w.dimensions()[1];
    //equidistance
    const T d = (v_max - v_min) / (n_w - 1);
    //input_size
    const unsigned int n_x = x.dimensions()[0];
    //number of channels
    const unsigned int n_c = x.dimensions()[1];

    for(int idc = 0; idc < n_c; idc++)
    {
      for(int idx = 0; idx < n_x; idx++)
      {
        const T pos = (x(idx, idc) - v_min) / d;
        const T pos_f = (std::floor(pos));
        const T alpha = pos - pos_f;
        int idw = static_cast<int>(pos_f);

        for(int dx = -1; dx <= 2; dx++)
        {
          T b_spline = 0;
          switch(N)
          {
            case tficg::DO_ZERO:
              switch(S)
              {
                case tficg::SO_LINEAR:
                b_spline = b_spline_linear<T>(dx - alpha);
                break;
                case tficg::SO_CUBIC:
                b_spline = b_spline_cubic<T>(dx - alpha);
                break;
              }
            break;
            case tficg::DO_FIRST:
              b_spline = b_spline_cubic_prime<T>(dx - alpha)/ (-d);
            break;
            case tficg::DO_SECOND:
              b_spline = b_spline_cubic_double_prime<T>(dx - alpha) / (d * d);
            break;
          }
          if ((idw + dx) >= 0 && (idw + dx) < n_w)
            grad_w(idc, idw + dx) += grad_out(idx, idc) * b_spline;
        }
      }
    }
  }
};

template<typename Device, typename T, tficg::SplineOrder S, tficg::DerivativeOrder N>
class ActivationBSplineGradWOp : public ActivationBaseGradWOp<Device, T> {
  public:
    explicit ActivationBSplineGradWOp(OpKernelConstruction* context) :
    ActivationBaseGradWOp<Device, T>(context)
    {
    }

  private:
    void ComputeGradient(OpKernelContext *context,
                         const typename Tensor2<T>::ConstTensor &x,
                         const typename Tensor2<T>::ConstTensor &grad_out,
                         typename Tensor2<T>::Tensor &grad_w) override
    {
      ActivationBSplineGradWFunctor<Device, T, S, N>()(context, x, grad_out, grad_w,
                this->v_min_, this->v_max_, this->feature_stride_);
    }
};

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationBSplineGradW") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineGradWOp<CPUDevice, T, tficg::SO_LINEAR, tficg::DO_ZERO>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationBSplineGradW") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineGradWOp<GPUDevice, T, tficg::SO_LINEAR, tficg::DO_ZERO>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationCubicBSplineGradW") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineGradWOp<CPUDevice, T, tficg::SO_CUBIC, tficg::DO_ZERO>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationCubicBSplineGradW") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineGradWOp<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_ZERO>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeCubicBSplineGradW") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineGradWOp<CPUDevice, T, tficg::SO_CUBIC, tficg::DO_FIRST>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeCubicBSplineGradW") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationBSplineGradWOp<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_FIRST>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif


// Linear interpolation Activation

template <typename T, tficg::DerivativeOrder N>
struct ActivationInterpolateLinearFunctor<CPUDevice, T, N> {
  void operator()(OpKernelContext *context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &w,
                  typename Tensor2<T>::Tensor &out,
                  T v_min, T v_max, int feature_stride) {
    std::cout << "Using int lin CPU kernel!" << std::endl;

    //number of weights
    const unsigned int n_w = w.dimensions()[1];
    //input_size
    const unsigned int n_x = x.dimensions()[0];
    //number of channels
    const unsigned int n_c = x.dimensions()[1];

    //equidistance
    const T d = (v_max - v_min) / (n_w - 1);

    for(int cnt_n = 0; cnt_n < n_c; cnt_n++)
    {
      for(int cnt_x = 0; cnt_x < n_x; cnt_x++)
      {
        const T pos = (x(cnt_x, cnt_n) - v_min) / d;
        const T pos_f = std::floor(pos);
        const T a = pos - pos_f;
        const int idw = static_cast<int>(pos_f);

        switch(N)
        {
          case tficg::DO_ZERO:
            if(pos >= n_w || pos <= -1)
            {
              out(cnt_x, cnt_n) = 0;
            }
            else if(idw < 0)
            {
              out(cnt_x, cnt_n) = w(cnt_n, 0) * a;
            }
            else if(pos >= (n_w - 1))
            {
              out(cnt_x, cnt_n) = w(cnt_n, n_w - 1) * (1 - a);
            }
            else
            {
              out(cnt_x, cnt_n) = w(cnt_n, idw) * (1 - a) + a * w(cnt_n, idw + 1);
            }
          break;
          case tficg::DO_FIRST:
            if(pos >= n_w || pos <= -1)
            {
              out(cnt_x, cnt_n) = 0;
            }
            else if(idw < 0)
            {
              out(cnt_x, cnt_n) = w(cnt_n, 0) / d;
            }
            else if(pos >= (n_w - 1))
            {
              out(cnt_x, cnt_n) = -w(cnt_n, n_w - 1) / d;
            }
            else
            {
              out(cnt_x, cnt_n) = (w(cnt_n, idw + 1) - w(cnt_n, idw)) / d;
            }
          break;
        }
      }
    }
  }
};

template<typename Device, typename T, tficg::DerivativeOrder N>
class ActivationInterpolateLinearOp : public ActivationBaseOp<Device, T> {
  public:
    explicit ActivationInterpolateLinearOp(OpKernelConstruction* context) :
      ActivationBaseOp<Device, T>(context)
    {
    }

  private:
    void ApplyActivation(OpKernelContext* context,
                         const typename Tensor2<T>::ConstTensor &x,
                         const typename Tensor2<T>::ConstTensor &w,
                         typename Tensor2<T>::Tensor &out) override
    {
      ActivationInterpolateLinearFunctor<Device, T, N>()(context,
        x, w, out, this->v_min_, this->v_max_, this->feature_stride_);
    }
};


#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationInterpolateLinear") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationInterpolateLinearOp<CPUDevice, T, tficg::DO_ZERO>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationInterpolateLinear") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationInterpolateLinearOp<GPUDevice, T, tficg::DO_ZERO>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeInterpolateLinear") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationInterpolateLinearOp<CPUDevice, T, tficg::DO_FIRST>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationPrimeInterpolateLinear") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationInterpolateLinearOp<GPUDevice, T, tficg::DO_FIRST>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

// Gradient Implementation
template <typename T, tficg::DerivativeOrder N>
struct ActivationInterpolateLinearGradWFunctor<CPUDevice, T, N> {
  void operator()(OpKernelContext* context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &grad_out,
                  typename Tensor2<T>::Tensor &grad_w,
                  T v_min, T v_max, int feature_stride) {
    std::cout << "Using CPU int lin gradient kernel!" << std::endl;
    // initialize the gradient w
    grad_w.setZero();

    //number of weights
    const unsigned int n_w = grad_w.dimensions()[1];
    //input_size
    const unsigned int n_x = x.dimensions()[0];
    //number of channels
    const unsigned int n_c = x.dimensions()[1];

    //equidistance
    const T d = (v_max - v_min) / (n_w - 1);

    for(int cnt_n = 0; cnt_n < n_c; cnt_n++)
    {
      for(int cnt_x = 0; cnt_x < n_x; cnt_x++)
      {
        const T pos = (x(cnt_x, cnt_n) - v_min) / d;
        const T pos_f = std::floor(pos);
        const T a = pos - pos_f;
        const int idw = static_cast<int>(pos_f);

        if(idw < 0)
        {
          grad_w(cnt_n, 0) += a * grad_out(cnt_x, cnt_n);
        }
        else if(pos >= (n_w - 1))
        {
          grad_w(cnt_n, idw) += (1 - a) * grad_out(cnt_x, cnt_n);
        }
        else
        {
          grad_w(cnt_n, idw) += grad_out(cnt_x, cnt_n) * (1 - a);
          grad_w(cnt_n, idw + 1) += grad_out(cnt_x, cnt_n) * a;
        }
      }
    }
  }
};

// Linear Interpolation Activation Class
template<typename Device, typename T, tficg::DerivativeOrder N>
class ActivationInterpolateLinearGradWOp : public ActivationBaseGradWOp<Device, T> {
  public:
    explicit ActivationInterpolateLinearGradWOp(OpKernelConstruction* context) :
    ActivationBaseGradWOp<Device, T>(context)
    {
    }

  private:
    void ComputeGradient(OpKernelContext *context,
                         const typename Tensor2<T>::ConstTensor &x,
                         const typename Tensor2<T>::ConstTensor &grad_out,
                         typename Tensor2<T>::Tensor &grad_w) override
    {
      ActivationInterpolateLinearGradWFunctor<Device, T, N>()(context, x, grad_out, grad_w,
                this->v_min_, this->v_max_, this->feature_stride_);
    }
};

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationInterpolateLinearGradW") \
    .Device(DEVICE_CPU) \
    .TypeConstraint<T>("T"), \
    ActivationInterpolateLinearGradWOp<CPUDevice, T, tficg::DO_ZERO>);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("ActivationInterpolateLinearGradW") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    ActivationInterpolateLinearGradWOp<GPUDevice, T, tficg::DO_ZERO>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif
