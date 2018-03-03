# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 07:37:24 2018

@author: Oliver
"""

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper

from cleverhans.utils_tf import model_eval


import tensorflow as tf

#from tensorflow.python.platform import flags
#FLAGS = flags.FLAGS

def attacks(model, sess):
	# Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
	report = AccuracyReport()
	
	# Define input TF placeholder
	x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
	y = tf.placeholder(tf.float32, shape=(None, 10))


	wrap = KerasModelWrapper(model)
	fgsm = FastGradientMethod(wrap, sess=sess)
	fgsm_params = {'eps': 0.3,
	                   'clip_min': 0.,
	                   'clip_max': 1.}
	adv_x = fgsm.generate(x, **fgsm_params)
	# Consider the attack to be constant
	adv_x = tf.stop_gradient(adv_x)
	preds_adv = model(adv_x)
	
	# Evaluate the accuracy of the MNIST model on adversarial examples
	eval_par = {'batch_size': batch_size}
	acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
	print('Test accuracy on adversarial examples: %0.4f\n' % acc)
	report.clean_train_adv_eval = acc