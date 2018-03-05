# -*- coding: utf-8 -*-
"""
Classifiers

Created on Fri Mar	2 05:18:46 2018

@author: Oliver
""" 
import sys
import os
from sklearn import svm, metrics   
#from sklearn.decomposition import PCA
#from sklearn.pipeline import Pipeline
#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegression

#from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
#from sklearn.multiclass import OneVsRestClassifier

import numpy as np
import tensorflow as tf

from attack import attacks
import scipy.misc

#import datetime
flags = tf.app.flags
flags.DEFINE_string("models", "linsvc,cnet", "the models to run")
flags.DEFINE_bool("adversarial", False, "run adversarial attacks")
flags.DEFINE_integer("examples", 20000, "number of examples")
flags.DEFINE_string("log", 'record.txt' , "log file")
flags.DEFINE_integer("epoch", -1, "number of epochs")
FLAGS = flags.FLAGS


def report(expected, predicted, message='', outfile = './results/record.txt') :
	creport = metrics.classification_report(expected, predicted)
	print(message)
	print("Classification report:\n%s\n" % (creport))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)) 
	
	with open(outfile, 'a') as log:
		log.write('\n' + ('*'*50) +'\n'+message+'\n\n'+ creport)
 
	return metrics.accuracy_score(expected, predicted)
	
def build(pre, modeler, post, name):
	def train(x_train_raw, y_train_raw, train_descr):	
		x_train,y_train, params = pre(x_train_raw, y_train_raw)
		print(name, ' -- train data ready. ', x_train.shape, y_train.shape)
		
		clf = modeler(x_train, y_train, **params)
		print(name, ' -- train data fit.')
		
		def test(x_test_raw, y_test_raw, test_descr):
			x_test, y_test, params2 = pre(x_test_raw, y_test_raw)
			
			
			if( FLAGS.adversarial ) :
				accuracy = attacks(clf, x_test, y_test)
				# require params = params2
				adv = "\n\nvulnerability: "+str(accuracy)
			else:
				adv = ''
	
			predict = post(clf.predict(x_test))
			expect = post(y_test)
			
			print(name, ' -- test data predicted')
			
			report(expect, predict, "   Model: "+name+"\nTraining: "+train_descr+
				"\n Testing: "+test_descr + adv, './results/'+FLAGS.log+'.txt')
			return test
			
		
		test.test = test
		return test
		
	train.train = train
	return name, train
	
def linsvc():
	def pre(X, Y):
		assert(X.shape[0] == Y.shape[0])
		return X.reshape((Y.shape[0], -1)), Y, {}

	def post(Y):
		return Y

	def modeler(X, Y):	
		classifier = svm.LinearSVC()	
		#classifier = Pipeline([('pca', PCA(n_components = dim_latent)), ('log', LogisticRegression())], verbose=True);
		#####  too long
		#classifier = OneVsRestClassifier(BaggingClassifier(svm.SVC(kernel='linear'),
		#				max_samples=1.0 / n_estimators, n_estimators=n_estimators))
		classifier.fit(X, Y)
		return classifier
	
	
	return build(pre, modeler, post, "linsvc")


def net():
	import keras
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten
	from keras.layers import Conv2D, MaxPooling2D
	from keras import backend as K

	
	def pre(X, Y):	
		num_classes = len(set(Y))
		# input image dimensions
		#img_rows, img_cols = 28, 28
		#chan = 
		#input_shape = (chan, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, chan)
		input_shape = X.shape[1:]
		# X = X.reshape(X.shape[0], *input_shape)
		X = X.astype('float32')
		X /= 255
		
		Y = keras.utils.to_categorical(Y, num_classes)
		return X, Y, dict(num_classes=num_classes, input_shape=input_shape)
	
	
	def modeler(X, Y, **params):
		epochs = 12
		batch_size = 128

		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),
						 activation='relu',
						 input_shape=params['input_shape']))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(params['num_classes'], activation='softmax'))
		
		model.compile(loss=keras.losses.categorical_crossentropy,
					  optimizer=keras.optimizers.Adadelta(),
					  metrics=['accuracy'])
	
		model.fit(X, Y,
				  batch_size=batch_size,
				  epochs=epochs,
				  verbose=1,
				  validation_split=0.1)
		return model
	
	def post(Y):
		return Y.argmax(axis=1)

	return build(pre, modeler, post, "cnet")
	
def sample(dataset, number):
	xs, ys, name = dataset
	indices = np.array(np.random.randint(xs.shape[0], size=number))
	return xs[indices], ys[indices], name+'-crop['+str(number)+']'
	
def ttsplit(X,Y, name, prc):
	n = int(prc * X.shape[0])
	return (X[:n], Y[:n], name+'-train'), (X[n:], Y[n:], name+'-test')

def both(data1, data2):
	X1, Y1, n1 = data1
	X2, Y2, n2 = data2
	
	sfx = ['-train', '-test']
	
	name = n1 +'&'+ n2
	for a, b in (sfx, reversed(sfx)):
		if n1.endswith(a):
			begin = n1[:-len(a)]
			if n2 == begin+b:
				name = begin+'-all'
	
	return np.concatenate((X1, X2), axis=0), np.concatenate((Y1, Y2), axis=0), name

def shuffle(data):
	X, Y, n = data
	idx = np.arange(X.shape[0])
	np.random.shuffle(idx)
	return X[idx], Y[idx], '§'+n

	
global VARS;
if __name__ == '__main__':
	import scipy.misc
	
	N_EXAMPLES = FLAGS.examples
	
	# Questions
	# HOw well trained on this test on original
	# train on this, test on thsi
	# train on original test on this.

	# trained on part this part original, test on original


	# samples are all in
	# /sampledir/fashion-n/split/...

	datasetname = sys.argv[1]
	gen_method = sys.argv[2]
	# First load original dataset
	
	def getoriginaldata():
		# I suspsect these might be faster which is why I've left them in.		
#		if datasetname == 'fashion':
#			from keras.datasets import fashion_mnist
#			return fashion_mnist.load_data()
#			
#		elif datasetname == 'mnist':
#			from keras.datasets import mnist
#			return mnist.load_data()
#		elif datasetname == 'cifar':
#			from keras.datasets import cifar10
#			return cifar10.load_data()

		from scipy.misc import imread

		Xs = {'train': [], 'test':[]}
		Ys = {'train': [], 'test':[]}
		
		base = './data/'+datasetname
		for slabel in os.listdir(base):
			for mode in 'train', 'test':
				base2 = base + '/' + slabel+ '/' + mode
				for imgfn in os.listdir(base2):
					x = imread(base2+"/"+imgfn)
					Xs[mode].append(x)
					Ys[mode].append(int(slabel))
		
		
		
		return (np.array(Xs['train']), np.array(Ys['train'])), (np.array(Xs['test']), np.array(Ys['test']))

		#raise ValueError('what is '+datasetname+'??')
		
	(xtrain, ytrain), (xtest, ytest) = getoriginaldata()
	nlabels = len(set(ytest))
	
	stand_train = xtrain, ytrain, 'standard-'+datasetname+'-train'
	stand_test = xtest, ytest, 'standard-'+datasetname+'-test'

	X_stand = np.concatenate((xtrain, xtest), axis=0)
	Y_stand = np.concatenate((ytrain, ytest), axis=0)
	stand_all = X_stand, Y_stand, 'standard-'+datasetname+'-all'
	
	stand_train_small = sample(stand_train, N_EXAMPLES) if N_EXAMPLES < stand_train[0].shape[0] else stand_train
	stand_test_small = sample(stand_test, N_EXAMPLES) if N_EXAMPLES < stand_test[0].shape[0] else stand_test
  
	Xs = []
	Ys = []
	
	#datapath = './samples' 
	datapath = './samples/'+gen_method
	folders = [f for f in os.listdir(datapath) if f.startswith(datasetname+'-')]
	
	for f in folders:
		label = int(f[len(datasetname)+1:])
		
		extra = ''
		if FLAGS.epoch > 0:
			extra += '/epoch-%d'
			
		if 'split' in  os.listdir(datapath+'/'+f + extra) :
			extra += '/split'			

		for idx, imagename in enumerate(os.listdir(datapath+'/'+f+extra)):
			# silly test optimization, force smaller data.			
			if idx > N_EXAMPLES / nlabels:
				break;

			# VERY IMPORTANT. This next line makes sure shitty training things from 
			# early on in the GAN process are not reused.
#			if ('test' in imagename):
			dataX = scipy.misc.imread(datapath+'/'+f+extra+'/'+imagename)
			Xs.append(dataX)
			Ys.append(label)
			
	X_gen = np.array(Xs)
	Y_gen = np.array(Ys)
	gen_inorder = X_gen, Y_gen, datasetname + '-gen-' + gen_method
	gen = shuffle(gen_inorder)

	#gen_small = sample(gen, N_EXAMPLES)
	gen_small = gen
	
	gen_small_train, gen_small_test = ttsplit(*gen_small, 0.7)

	print("ALL DATA LOADED\n" + '*'*50)

	learners = dict([linsvc(), net()])
	
	global VARS
	VARS = locals()
	
	for name in FLAGS.models.split(','):
		if name not in learners:
			print("NO such learner: "+name)
			continue
		
		learner = learners[name]
		print("learner: ", name)
			   
		# train on gen, test on standard_all 
		learner.train(*gen_small_train) \
			.test(*stand_train_small)(*stand_test_small)(*gen_small_test)
			
		learner.train(*stand_train_small) \
			.test(*stand_test_small)(*gen_small_test)(*gen_small)
			
		learner.train(*gen_small) \
			.test(*stand_test)(*stand_all)
		
		  
		
		
		
		

		
		
		
		
		
		
		
		
