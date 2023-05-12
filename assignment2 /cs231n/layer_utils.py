from .layers import *
from .fast_layers import *
import numpy as np


def affine_relu_forward(x, w, b, dropout_params):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """

    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    if dropout_params:
      out, drop_cache = dropout_forward(out,dropout_params)
    else:
      drop_cache = None
    cache = (fc_cache, relu_cache, drop_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache, drop_cache = cache
    if drop_cache:
      dout = dropout_backward(dout, drop_cache) 
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def general_forward(x,w,b,gamma, beta, bn_param, normalization, dropout_params):
    """
    affine - batch_norm - relu

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    batch_cache, layer_cache, drop_cache = None, None, None
    a, fc_cache = affine_forward(x, w, b)
    D = a.shape[1]
    if normalization == "batchnorm":
      out_norm , batch_cache = batchnorm_forward(a,gamma, beta, bn_param)
      out, relu_cache = relu_forward(out_norm)
    elif normalization == "layernorm":
      out_norm , layer_cache = layernorm_forward(a,gamma, beta, bn_param)
      out, relu_cache = relu_forward(out_norm)
    else:
      out, relu_cache = relu_forward(a)

    if dropout_params:
      out, drop_cache = dropout_forward(out,dropout_params)
    else:
      drop_cache = None

    cache = (fc_cache,batch_cache, layer_cache, relu_cache, drop_cache)
    return out, cache

def general_backward(dout, cache):
    fc_cache, batch_cache, layer_cache, relu_cache, drop_cache = cache 
    if drop_cache:
      dout = dropout_backward(dout, drop_cache) 
    da = relu_backward(dout, relu_cache)
    dgamma,dbeta = None, None
    if batch_cache:
      d_batch, dgamma, dbeta = batchnorm_backward(da,batch_cache)
      dx, dw, db = affine_backward(d_batch, fc_cache)
    elif layer_cache:
      d_layer, dgamma, dbeta = layernorm_backward(da,layer_cache)
      dx, dw, db = affine_backward(d_layer, fc_cache)
    else:
      dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta



# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for the conv-relu-pool convenience layer.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
