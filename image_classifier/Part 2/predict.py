# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 15:42:39 2022

@author: bhong2
"""


# This is an image classifier for flowers
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from PIL import Image
import json


def process_image(image, image_size = 224):
    image_tf = tf.convert_to_tensor(image)
    image_tf = tf.cast(image_tf, tf.float32)
    image_tf = tf.image.resize(image_tf, (image_size, image_size))
    image_tf /= 255
    image    = image_tf.numpy()
    return image


def predict(image_path, model_path, top_k=5, category_names='label_map.json'):

    im = Image.open(image_path)
    test_image = np.asarray(im)
    
    processed_test_image = process_image(test_image)
    model_input_image    = np.expand_dims(processed_test_image,axis=0)

    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    
    ps = model.predict(model_input_image)
    probs = np.sort(ps,axis = 1)[0][-1:-top_k-1:-1]     #numpy array
    classes = (np.argsort(ps,axis = 1)[0][-1:-top_k-1:-1] + 1).tolist()   # class label start from 1; conver to list in order to have ','
    classes = [str(classes[i]) for i in range(top_k)]   
        
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    
    classnames = [class_names[classes[i]] for i in range(top_k)]
    
    print('\nTop',top_k,'probabilities are:',probs)
    print('Corresponding classe labels are:', classes)
    print('Corresponding classe names are:', classnames)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(8,12), ncols=2)
    ax1.imshow(test_image)
    ax1.axis('off')
    ax1.set_title(image_path.split("/")[-1][0:-4])
    ax2.barh(np.arange(top_k), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels(classnames, size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

    return probs, classes

# Run inside python
#image_path = './test_images/hard-leaved_pocket_orchid.jpg'
#image_path = './test_images/cautleya_spicata.jpg'
#image_path = './test_images/orange_dahlia.jpg'
#image_path = './test_images/wild_pansy.jpg'
#model_path = 'my_model_py.h5'
#probs, classes = predict(image_path, model_path)


# Run through command lines

import argparse

parser = argparse.ArgumentParser(description='This is an image classifier for flowers.')
parser.add_argument('image_file', type=str, help='path and file of the image')
parser.add_argument('model_file', type=str, help = 'path and file of the model')
parser.add_argument('--top_k',dest='topk', type=int, help='top K probabilities')
parser.add_argument('--category_names', dest='category', type=str, help ='path and file of the categories')
args = parser.parse_args()


if (args.topk is None) and (args.category is None):
    probs, classes = predict(args.image_file, args.model_file)
elif (args.topk is not None) and (args.category is None):
    probs, classes = predict(args.image_file, args.model_file, top_k = args.topk)
elif (args.topk is None) and (args.category is not None):
    probs, classes = predict(args.image_file, args.model_file, category_names = args.category)
else:
    probs, classes = predict(args.image_file, args.model_file, args.topk, args.category)

