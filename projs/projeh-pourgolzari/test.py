#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from glob import glob

import cv2
import skimage
from skimage import morphology
from skimage import measure


# In[25]:


def load_dataset(path, phase):
    files = glob(f'{path}/{phase}/*/*.bmp')
    images, labels, filenames = [], [], []
    for i, file in enumerate(files):
        image = cv2.imread(file, 0) #load gray mode
        images.append(image)
    
    return images


# In[26]:


images_train = load_dataset("data/shapes dataset/", "test")


# In[27]:


def feature_extraction(img):
    # thresholding
    img[img < img.max()] = 0
    img[img == img.max()] = 1.
    img = img.astype(np.float32)
    img = 1 - img
    
    # hole filling
    seed, mask = img.copy(), img.copy()
    seed[1:-1, 1:-1] = img.max()
    filled = morphology.reconstruction(seed, mask, method='erosion')

    # feature description
    props = measure.regionprops(filled.astype(np.uint8))
    eccentricity = props[0].eccentricity
    area = props[0].area
    perimeter = props[0].perimeter
    compactness = perimeter ** 2 / area
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    return eccentricity, circularity


# In[28]:


feats_train = []

for image in images_train:
    feats = feature_extraction(image)
    feats_train.append(feats)

X_train = np.array(feats_train)

X_train

