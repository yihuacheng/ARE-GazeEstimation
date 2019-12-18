#encoding=utf8
#VGGmodel
import traceback
import tensorflow as tf
import numpy as np
import math
import pdb
#batch normalization

def Conv2d(inputs, name=None, filters=32, size=3, strides=1,training=False,
			isbatch=True, isrelu=True, bias=0.1, kernel=0.1, padding="VALID"):
	
	with tf.variable_scope(name):
		output = tf.layers.conv2d(inputs=inputs, filters=filters,
						kernel_size=[size, size],
						strides=[strides, strides],
						padding=padding, activation=None,
						bias_initializer=tf.constant_initializer(bias),
						kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
		if isbatch:
			net = tf.layers.batch_normalization(output,training=training)
		else:
			net = output
		
		if isrelu:
			net = tf.nn.relu(net)
		return net


def dense(inputs, units, training, name=None, kernel=0.1, bias=0.1, activation=tf.nn.relu, isbatch=False):
	
	with tf.variable_scope(name):
		if isbatch:
			activation = None

		output = tf.layers.dense(inputs=inputs, units=units, activation=activation,
			kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
			bias_initializer=tf.constant_initializer(bias))
		
		if isbatch:
			net = tf.layers.batch_normalization(output,training=training)
			net = tf.nn.relu(net)
		else:
			net = output
		
		return net

def SingleEyeBlock(inputs, training, units=500, name=None):
	with tf.variable_scope(name):
		net = Conv2d(inputs, name="Conv1", filters=64, strides=1, training=training)
		net = Conv2d(net, name="Conv2", filters=64, strides=2, training=training)
		net = Conv2d(net, name="Conv3", filters=128, strides=1, training=training)

		net = Conv2d(net, name="Conv5", filters=256, strides=1, training=training)
		net = Conv2d(net, name="Conv6", filters=256, strides=2, training=training)
		node = tf.layers.flatten(net)
		net = dense(inputs=node, units=units, training=training, name="fc", isbatch=True)
		net = tf.layers.dropout(net, training=training)
		return net	

def angular(x, y):
	xy = tf.reduce_sum(x*y,1)
	x_len = tf.sqrt(tf.reduce_sum(tf.square(x), 1))
	y_len = tf.sqrt(tf.reduce_sum(tf.square(y), 1))
	degree = tf.acos(tf.minimum( xy/(x_len* y_len ), 0.999999))
	return degree


def ARModel(features, labels, mode, params):
	is_training = (mode == tf.estimator.ModeKeys.TRAIN)
	#input
	x_left_image = tf.feature_column.input_layer(features, params['left'])
	x_left_image = tf.reshape(x_left_image,[-1,36,60,1])

	x_right_image = tf.feature_column.input_layer(features, params['right'])
	x_right_image = tf.reshape(x_right_image,[-1,36,60,1])
	
	headpose = tf.feature_column.input_layer(features, params['head'])
	headpose = tf.reshape(headpose, [-1,6])
	
	label = tf.feature_column.input_layer(features, params['label'])
	label = tf.reshape(label, [-1,6])
	
	name = tf.feature_column.input_layer(features, params['name'])


	with tf.variable_scope("ARNET"):
		with tf.variable_scope("Both"):
			left = SingleEyeBlock(x_left_image, units=500, name="left",training=is_training)
			right = SingleEyeBlock(x_right_image, units=500, name="right",training=is_training)
			fusion = tf.concat([left,right],1)
			fusion = dense(fusion, units = 500, training=is_training, name="fusion", isbatch=True)
		
		with tf.variable_scope("Split"):
			# left feature network
			left = SingleEyeBlock(x_left_image, units=1000, name="left",training=is_training)
			left = dense(left, units=500, training=is_training, name="left", isbatch=True) 

			# right feature network
			right = SingleEyeBlock(x_right_image, units=1000, name="right",training=is_training)
			right = dense(right, units=500, training=is_training, name="right", isbatch=True)
			
		# concat all feature
		finalfeature = tf.concat([fusion, left, right, headpose],1)
 
		leftgaze = dense(finalfeature, units=3, training=is_training, name="gazeleft", activation=None)
		rightgaze = dense(finalfeature, units=3, training=is_training, name="gazeright", activation=None)
		
		leftgaze = tf.cast(leftgaze, tf.float64)
		leftgaze = tf.cast(leftgaze / tf.reshape(
					tf.sqrt(tf.reduce_sum(tf.square(leftgaze), 1)),[-1,1]), tf.float32)
		
		rightgaze = tf.cast(rightgaze, tf.float64)
		rightgaze = tf.cast(rightgaze / tf.reshape(
					tf.sqrt(tf.reduce_sum(tf.square(rightgaze), 1)),[-1,1]), tf.float32)


		
	left_ac = angular(label[: , 0:3], leftgaze)
	right_ac = angular(label[:, 3:6], rightgaze)
		
	gaze_ac = tf.concat([ tf.reshape(left_ac, [-1,1]), tf.reshape(right_ac, [-1,1]) ],1)

	ar_error = 1/(right_ac + 1e-10 )/( 1/(right_ac + 1e-10) + 1/( left_ac + 1e-10 )) * right_ac +\
			1/(left_ac + 1e-10 )/( 1/(right_ac + 1e-10) + 1/( left_ac + 1e-10 )) * left_ac
	
	avg_error =  (left_ac + right_ac)/2
	

	with tf.variable_scope("ENET"):
		#left image network
		net = SingleEyeBlock(inputs=x_left_image, units=1000, name="left",training=is_training)
		left = dense(net, units=500, training=is_training, name="left", isbatch=True)

		net = SingleEyeBlock(inputs=x_right_image, units=1000, name="right",training=is_training)
		right = dense(net, units=500, training=is_training, name="right", isbatch=True)
			
		feature = tf.concat([left, right, headpose], 1)
		choose = dense(feature, units=2, training=is_training, name="choose", activation=tf.nn.softmax)
		
		#1 is right . 0 is left.
		results = tf.cast( tf.argmax(choose, 1), tf.float32)

		trues = tf.concat([tf.reshape(left_ac,[-1,1]),
							tf.reshape(right_ac,[-1,1])],1)
		
		groundtrues = tf.cast( tf.argmax(trues, 1), tf.float32)
	
		correct_prediction = tf.cast(tf.equal(results, groundtrues),tf.float32)
		
		weight = tf.reshape(tf.reduce_max(choose, 1), [-1,1]) - tf.reshape(tf.reduce_min(choose, 1), [-1,1])
		weight = (weight * (correct_prediction * 2 - 1) + 1) * 0.5
	
	choose_ac = gaze_ac * tf.one_hot(tf.cast(results, tf.uint8), 2) * 2

	allvar = tf.trainable_variables()
	r_vars = [var for var in allvar if 'ARNET' in var.name]
	c_vars = [var for var in allvar if 'ENET' in var.name]

	cross_entropy =  -tf.reduce_mean( tf.one_hot(tf.cast(groundtrues, tf.uint8), 2) * tf.log(choose) )
		
	are_error = weight * ar_error + (1-weight) * 0.1 * avg_error


	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		estimation_step =  tf.train.AdamOptimizer(0.05).minimize(tf.reduce_mean(are_error),
								var_list = r_vars, global_step = tf.train.get_global_step())

		choose_step     =  tf.train.AdamOptimizer(0.001).minimize(cross_entropy,  
								var_list = c_vars, global_step = tf.train.get_global_step())
	
	predict={
		"name": name,
		"left": leftgaze,
		"right": rightgaze,
		"left_ac": left_ac,
		"right_ac": right_ac,
		"choose": results
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=predict)

	if mode == tf.estimator.ModeKeys.TRAIN:	
		train_op = tf.group(estimation_step, choose_step)
		return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_mean(avg_error)*180/math.pi, train_op=train_op)
	
	metric = {
		"avg": tf.metrics.mean(avg_error*180/math.pi),
		"left": tf.metrics.mean(left_ac*180/math.pi),
		"right": tf.metrics.mean(right_ac*180/math.pi),
		"choose": tf.metrics.mean(choose_ac*180/math.pi),
		"accuracy": tf.metrics.mean(correct_prediction)
	}

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode, loss=tf.reduce_mean(avg_error)*180/math.pi, eval_metric_ops = metric)
	
	

