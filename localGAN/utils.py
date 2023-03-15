import tensorflow as tf
import keras
import keras.backend as K

import os
import sys
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import skimage
import cv2
import pathlib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
import scipy

import matplotlib.pyplot as plt

from model_settings import GAN, Generator, Discriminator


class DataGenerator:
    def __init__(self,files_list,shape,num_classes=0,categorize=False,mean=0.5):
        self.files = copy.deepcopy(files_list)
        self.shape = shape
        self.categorize = categorize
        self.mean = mean
        self.num_classes = num_classes
        random.shuffle(self.files)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,idx):
        file = self.files[idx]
        image = parse_image(file)
        if self.categorize == False:
            return image
        else:
            onehot = tf.one_hot(range(self.num_classes), self.num_classes)
            category = int(os.path.split(os.path.split(file)[0])[1].split('_')[1])
            hot_label = onehot[int(category)]
            return image, hot_label

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__()-1:
                self.on_epoch_end()
    
    def parse_image(self,filename,mean=0.5):
        img = skimage.io.imread(filename)[:,:,:3]
        img = tf.convert_to_tensor(img)
        img = tf.image.per_image_standardization(img)
        img = tf.image.random_crop(img, (self.shape[0], self.shape[1],3)).numpy()
        if mean == 0.5:
            img = np.divide(np.add(img,np.abs(img.min())),np.add(img.max(),np.abs(img.min()))+0.00001)
        return img

    def on_epoch_end(self):
        reidx = random.sample(population = list(range(self.__len__())),k = self.__len__())
        self.imgarr = self.imgarr[reidx]

    
def get_smallest_img(files_list):
    result = [100000,100000,4]
    for file in tqdm(files_list):
        img = skimage.io.imread(file)
        arr = np.array(img)
        for i in range(0,3):
            if arr.shape[i]<result[i]:
                result[i] = arr.shape[i]
    return result


def normalize_images(image):
    image = tf.image.per_image_standardization(image)
    return tf.math.divide(tf.math.add(image,tf.math.abs(tf.math.reduce_min(image))),
                          tf.math.add(tf.math.reduce_max(image),tf.math.abs(tf.math.reduce_min(image)))+0.00001)


def process_unknown(unknown_data,model_path,size=476,threshold=2):
    
    def normalize_image_only(image):
        image = np.asarray(Image.open(image))
        image = tf.convert_to_tensor(image)
        image = tf.keras.layers.CenterCrop(size,size)(image)
        image = tf.image.per_image_standardization(image)
        image = tf.math.divide(tf.math.add(image,tf.math.abs(tf.math.reduce_min(image))),tf.math.add(tf.math.reduce_max(image),tf.math.abs(tf.math.reduce_min(image)))+0.00001)
        if len(image.shape) == 3:
            return np.expand_dims(image,0)
        else:
            return image
    
    if type(unknown_data) is str: 
        image_list = {os.path.split(unknown_data)[-1]: normalize_image_only(unknown_data)}
    else:
        image_list = {os.path.split(item)[-1]: normalize_image_only(item) for item in unknown_data}
    
    model = tf.keras.models.load_model(model_path)
    
    output = {}
    for name,image in image_list.items():
        arr = model.predict(image,verbose=0).flatten()
        class_matches = np.argpartition(arr,-3)
        class_matches = class_matches[::-1]
        filtered_matches = {key:np.log(arr[key]) for key in class_matches if arr[key]>threshold}
        output[name] = {}
        for c,v in filtered_matches.items():
            output[name][c] = v
        # print(output[name])
            
    return output


def predict_unknown(img_path,model,shape):
    img = np.asarray(Image.open(img_path))
    img = tf.convert_to_tensor(img)
    img = tf.keras.layers.CenterCrop(shape[0], shape[1])(img)
    img = tf.image.per_image_standardization(img)
    img = tf.math.divide(tf.math.add(img,tf.math.abs(tf.math.reduce_min(img))),tf.math.add(tf.math.reduce_max(img),tf.math.abs(tf.math.reduce_min(img)))+0.00001)
    if len(img.shape) == 3:
        img = np.expand_dims(img,0)
    generator = Generator(shape)
    discriminator = Discriminator(shape)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005,beta_1=0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                             discriminator_optimizer=discriminator_optimizer,
                             generator=generator,
                             discriminator=discriminator)
    checkpoint.restore(model).expect_partial()
    model = GAN(generator,discriminator,512)
    model.compile(generator_optimizer, discriminator_optimizer, loss_fn)
    model.built = True
    evaluation = model.predict(img,verbose=0)
    return evaluation[0][0]


def unsup_and_sup(unknown_data,model_path,size=476):
    
    unsup_class = int(unknown_data.split('\\')[-2].split('_')[-1])
    
    def normalize_image_only(image):
        image = np.asarray(Image.open(image))
        image = tf.convert_to_tensor(image)
        image = tf.keras.layers.CenterCrop(size,size)(image)
        image = tf.image.per_image_standardization(image)
        image = tf.math.divide(tf.math.add(image,tf.math.abs(tf.math.reduce_min(image))),tf.math.add(tf.math.reduce_max(image),tf.math.abs(tf.math.reduce_min(image)))+0.00001)
        if len(image.shape) == 3:
            return np.expand_dims(image,0)
        else:
            return image
    
    if type(unknown_data) is str: 
        image_list = {os.path.split(unknown_data)[-1]: normalize_image_only(unknown_data)}
    else:
        image_list = {os.path.split(item)[-1]: normalize_image_only(item) for item in unknown_data}
    
    model = tf.keras.models.load_model(model_path)
    
    output = {}
    for name,image in image_list.items():
        arr = model.predict(image,verbose=0).flatten()
        class_matches = np.argpartition(arr,-3)
        class_matches = class_matches[::-1]
        filtered_matches = {key:np.log(arr[key]) for key in class_matches if arr[key]>2}
        output[name] = {}
        for c,v in filtered_matches.items():
            output[name][c] = v
        # print(output[name])
    top_3 = list(output[list(image_list.keys())[0]].keys())[:3]
            
    if unsup_class in top_3:
        return 1
    else:
        return 0
    
    
if __name__ == "__main__":
    pass
else:
    pass
