__doc__="""cat preprocessing function"""

import sys

import os
import csv
import numpy as np
from PIL import Image,ImageOps 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle





def load_cats_dogs():
    """
    downloads dataset from A.Tvorozhkov's dropbox
    """
    urls = ['https://yadi.sk/d/0tqVcm9HsxyAu']
    filenames = ['dogs_vs_cats.train.zip']
    for u, f in zip(urls,filenames):
        os.system("wget "+u+" -O "+f)
    return True


# Convert all the image files in the given path into np arrays with dimensions suitable for DL with Theano
def jpg_to_nparray(path,img_names, img_size, grayscale = False):
    X = []
    Y = []
    img_colors = 3

    for counter,img_dir in enumerate(img_names):

        # X
        img = Image.open(path+img_dir)
        img = ImageOps.fit(img, img_size, Image.ANTIALIAS)
        
        if grayscale:
            img = ImageOps.grayscale(img)
            img_colors = 1


        img = np.asarray(img, dtype = 'float32') / 255.
        img = img.reshape([img_colors]+list(img_size))
        X.append(img)

        # Y: 0 for cat, 1 for dog
        if "cat" in img_dir:
            Y.append(0)
        else:
            Y.append(1)


        # Printing
        counter+=1
        if counter%1000 == 0:
            print'processed images: ', counter

    X = np.asarray(X)
    Y = np.asarray(Y,dtype='int32')

    return (X,Y)



# Get ids of the images: we'll need them for generating the submission file for Kaggle
def get_ids(path):
    ids = np.array([],dtype = int)
    for str in os.listdir(path):
        ids = np.append(ids, int(str.partition(".")[0]))
    
    ids = np.array(ids, dtype = int)[...,None]
    return ids


