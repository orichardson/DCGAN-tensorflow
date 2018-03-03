# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 03:16:14 2018

@author: Oliver
"""

from split import ensure_directory, imsave


"""
This method creates datasets segregated by label in the folders
<basedir>/<label>/<mode>/im<>.jpg

e.g.
data/cifar/0/train/im32.jpg

:param: dataset ( (x_train, y_train), (x_test, y_test) )

"""

#from keras import datasets as kdat
#import pkgutil

# this is the package we are inspecting -- for example 'email' from stdlib

def make_dataset_from_karas(name) :
	pass
	#exec("import keras.datasets.%s as adataset" % name)
	#print("import keras.datasets.%s as adataset" % name)
	#dataset = adataset.load_data()
	#print(dataset[0][0].shape)


	#make_dataset_files(dataset, './make_dataset_files')	
	#for importer, modname, ispkg in pkgutil.iter_modules(kdat.__path__):
#		kdat.cifar10.load_data(name)

def make_dataset_files( dataset ,  base_directory) :
    train, test = dataset
    data = {"train" :  train, "test": test}  
    x_train, y_train = train
    x_test, y_test = test
    
    labelset = set(y_train) | set(y_test)
    counters = {}
    
    ensure_directory(base_directory)
    
    for label in labelset:
        counters[label] = 0
        labeldir = base_directory+"/"+str(label)
        ensure_directory(labeldir)
        ensure_directory(labeldir+"/train")
        ensure_directory(labeldir+"/test")
    
    def save(x, y, mode):
        imsave(x, base_directory+"/"+str(y) +"/"+mode+"/im"+str(counters[y])+".jpg")
        counters[y] += 1
    
    for mode, (X,Y) in data.items() :
        for x,y in zip(X,Y):
            save(x, y, mode)


