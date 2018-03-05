# -*- coding: utf-8 -*-
"""
Adapted from:
https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.py
"""


from tensorflow.contrib.slim import fully_connected as fc
import tensorflow as tf
import sys

#from sklearn.base import BaseEstimator, ClusterMixin

from split import ensure_directory


class VariantionalAutoencoder(object): #(BaseEstimator, ClusterMixin):
	def __init__(self, n_z=10, insize=784, midsizes=None, learning_rate=1e-3, batch_size=100, num_epoch=100):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_epoch = num_epoch
		
		if midsizes is None:
			midsizes = np.array([512,384,256],dtype=int) # * insize // 784 # scale inner layers to input
		self.midsizes = midsizes
		
		self.n_z = n_z
		self.insize = insize
		

		self.build()

		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())

	# Build the netowrk and the loss functions
	def build(self):
		n_x = self.insize
		n_z = self.n_z
		self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, n_x])

		# Encode
		# x -> z_mean, z_sigma -> z
		# f1 = fc(self.x, 512, scope='enc_fc1', activation_fn=tf.nn.elu)
		# f2 = fc(f1, 384, scope='enc_fc2', activation_fn=tf.nn.elu)
		# f3 = fc(f2, 256, scope='enc_fc3', activation_fn=tf.nn.elu)
		
		prev = self.x
		for li, size in enumerate(self.midsizes, 1):
			fl = fc(prev, size, scope='enc_fc'+str(li), activation_fn=tf.nn.elu)
			prev = fl
			
		li += 1
		
		self.z_mu = fc(prev, n_z, scope='enc_fc%d_mu' % li , activation_fn=None)
		self.z_log_sigma_sq = fc(prev, n_z, scope='enc_fc%d_sigma' % li, activation_fn=None)
		eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),
							   mean=0, stddev=1, dtype=tf.float32)
		self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

		# Decode
		# z -> x_hat
		# g1 = fc(self.z, 256, scope='dec_fc1', activation_fn=tf.nn.elu)
		# g2 = fc(g1, 384, scope='dec_fc2', activation_fn=tf.nn.elu)
		# g3 = fc(g2, 512, scope='dec_fc3', activation_fn=tf.nn.elu)

		prev = self.z
		for li, size in enumerate(reversed(self.midsizes),1):
			prev = fc(prev, size, scope='dec_fc'+str(li), activation_fn=tf.nn.elu)

		li += 1
		
		self.x_hat = fc(prev, n_x, scope='dec_fc%d' % li, activation_fn=tf.sigmoid)

		# Loss
		# Reconstruction loss
		# Minimize the cross-entropy loss
		# H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
		epsilon = 1e-10
		recon_loss = -tf.reduce_sum(
			self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat),
			axis=1
		)
		self.recon_loss = tf.reduce_mean(recon_loss)

		# Latent loss
		# Kullback Leibler divergence: measure the difference between two distributions
		# Here we measure the divergence between the latent distribution and N(0, 1)
		latent_loss = -0.5 * tf.reduce_sum(
			1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
		self.latent_loss = tf.reduce_mean(latent_loss)

		self.total_loss = tf.reduce_mean(recon_loss + latent_loss)
		self.train_op = tf.train.AdamOptimizer(
			learning_rate=self.learning_rate).minimize(self.total_loss)
		return

	# x -> x_hat
	def reconstructor(self, x):
		x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
		return x_hat

	# z -> x
	def generator(self, z):
		x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
		return x_hat

	# x -> z
	def transformer(self, x):
		z = self.sess.run(self.z, feed_dict={self.x: x})
		return z
		
	def fit(self, X):
		num_samples = X.shape[0]
		for epoch in range(self.num_epoch):
			for i in range(0, num_samples, self.batch_size): # for each batch (not each image)
				batch = X[i:i+self.batch_size]
				# Execute the forward and the backward pass and report computed losses
				_, loss, recon_loss, latent_loss = self.sess.run(
					[self.train_op, self.total_loss, self.recon_loss, self.latent_loss],
					feed_dict={self.x: batch}
				)
				
				if np.isnan(loss):
					print('batch range: ', batch.min(), batch.max())
					print('batch size: ', batch.shape)
					G = globals()
					G['evil_batch'] = batch
					sys.exit(1)
		
			if epoch % 10 == 0:
				print('[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
					epoch, loss, recon_loss, latent_loss))
		print('Done!')


flags = tf.app.flags
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string('input_fname_pattern', '*.jpg', 'descriptor for files')
flags.DEFINE_integer("input_height", None, "The size of image to use (will be center cropped). If none, resize model instead.")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer('batches_generated', 500, "Number of batches to generate")

flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS


if __name__ == '__main__':
	import numpy as np
	import scipy.misc
	import os	
	from glob import glob
	#from utils import get_image
	from scipy.misc import imread

	w = FLAGS.input_width
	h = FLAGS.input_height

	if h is None:
		h = w

	filelist = glob(os.path.join("./data", FLAGS.dataset, FLAGS.input_fname_pattern))
	print(next(iter(filelist)))
	
	im0 = imread(next(iter(filelist)))
	
	if w is None:
		h, w = im0.shape[:2]

	
	## HACKY cropping instead of reading the appropriate thing
	border_r = (im0.shape[0]-h)/2
	border_c = (im0.shape[1]-w)/2
	
	model = VariantionalAutoencoder(n_z = 10, insize=w*h)
	images = np.array([imread(sample_file)[border_r:border_r+h,border_r:border_r+w] \
		for sample_file in filelist]).reshape(-1, w*h)

	
			
	print('images of shape:', images.shape)
	images /= 255
	print('range of image values: ', images.min(), images.max())
	model.fit(images)
	
	
	ensure_directory(FLAGS.sample_dir)
	
	n = np.sqrt(model.batch_size).astype(np.int32)
	
	counter = 0
	
	
	for kk in range(FLAGS.batches_generated):
		# Test the trained model: generation
		# Sample noise vectors from N(0, 1)
		z = np.random.normal(size=[model.batch_size, model.n_z])
		x_generated = (model.generator(z)*255).astype(int)
		I_generated = np.empty((h*n, w*n))

		for i in range(n):
			for j in range(n):
				im = x_generated[i*n+j, :].reshape(w, h)
				scipy.misc.imsave(FLAGS.sample_dir+'/split/im%d.png'%counter, im) 
				I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = im
				#'./samples/vae/%s-%s/split/im%d.jpg' % (flags.dataset, label, counter)
				counter += 1
				
		scipy.misc.imsave(FLAGS.sample_dir+'/amalg%d.pngx'%kk, I_generated)

	print("********** done *********")