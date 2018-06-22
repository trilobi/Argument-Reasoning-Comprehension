import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.training.moving_averages import assign_moving_average

BN_DECAY = 0.9997
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
RESNET_VARIABLES = 'resnet_variables'

def _get_variable(name, shape, initializer, weight_decay = 0.0, dtype = 'float', trainable = True):
	
	# a little wrapper around tf.get_variable to do weight decay and add to resnet collection
	if weight_decay > 0:
		regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
	else:
		regularizer = None
	collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
	return tf.get_variable(name, shape = shape, initializer = initializer, dtype = dtype, regularizer =regularizer, collections = collections, trainable = trainable)


def bn(x, is_training):
	x_shape = x.get_shape()
	params_shape  = x_shape[-1:]
	axis = list(range(len(x_shape) - 1))
	beta = _get_variable('beta', params_shape, initializer = tf.zeros_initializer())
	gamma = _get_variable('gamma', params_shape, initializer = tf.ones_initializer())
	moving_mean = _get_variable('moving_mean', params_shape, initializer = tf.zeros_initializer(), trainable = False)
	moving_variance = _get_variable('moving_variance', params_shape, initializer = tf.ones_initializer(), trainable = False)

	# These ops will only be preformed when training
	mean, variance = tf.nn.moments(x, axis)
	update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
	update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
	tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
	tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
	mean, variance = control_flow_ops.cond(is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance))
	return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)


def bn2(x, is_training, eps = 1e-05, decay = 0.9, affine = True, name = None):
	with tf.variable_scope(name, default_name = 'BatchNorm'):
		params_shape = x.shape[-1:]
		axis = list(range(len(x.get_shape()) - 1))
		moving_mean = tf.get_variable('moving_mean', params_shape, initializer = tf.zeros_initializer(), trainable = False)
		moving_variance = tf.get_variable('moving_variance', params_shape, initializer = tf.ones_initializer(), trainable = False)
		def mean_var_with_update():
			mean, variance = tf.nn.moments(x, axis, name = 'moments')
			with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay), assign_moving_average(moving_variance, variance, decay)]):
				return tf.identity(mean), tf.identity(variance)
		mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_variance))
		if affine:
			beta = tf.get_variable('beta', params_shape, initializer = tf.zeros_initializer())
			gamma = tf.get_variable('gamma', params_shape, initializer = tf.ones_initializer())
			x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
		else:
			x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
		return x

if __name__  == '__main__':
	pass
