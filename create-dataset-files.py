# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 03:16:14 2018

@author: Oliver
"""

from split import ensure_directory, imsave

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


