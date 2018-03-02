# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 04:36:44 2018

@author: Oliver
"""

import sys
import os, errno
import scipy.misc
import math
import itertools

def imsave(image, path):
    return scipy.misc.imsave(path, image)

def ensure_directory(directory):        
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
def insert_into_filename(fname, suffix):
    start, end = fname.split('.',1)
    return start + suffix + '.' + end

def patches(image, side):
    dx,dy = map(int, (image.shape[0] / side, image.shape[1] / side))
    square = itertools.product(range(0,image.shape[0], dx), range(0,image.shape[1], dy))

    if len(image.shape) == 2:        
        return [ image[i:i+dx, j:j+dy] for (i,j) in square]

    return [ image[i:i+dx, j:j+dy, :] for (i,j) in square]
        

if __name__ == '__main__':
    folderpath = sys.argv[1]
    batchsize = int(sys.argv[2])
    
    side_length = int(math.sqrt(batchsize))
    assert(side_length * side_length == batchsize)
    
    ensure_directory(os.path.join(folderpath, 'split'))
    
    for imgname in os.listdir(folderpath) :
        if os.path.isfile(os.path.join(folderpath, imgname)):
            image = scipy.misc.imread(os.path.join(folderpath, imgname))
            
            for count,patch in enumerate(patches(image, side_length)) :       
                name = os.path.join(folderpath, 'split', insert_into_filename(imgname, "-split"+str(count)))
                print(name)
                imsave(patch, name)