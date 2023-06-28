# Configure Python environment for Dataset pipline
# --------------
import os
import cv2
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import process as P

import numpy as np
from PIL import Image
import tensorflow as tf
from typing import List
import warnings
warnings.filterwarnings('ignore')


# Global variables
# ----------------
IMAGE_PATH = 'ToBeDefined'
N_CHANNEL = 3
BATCH_SIZE = 32
HEIGHT = 256
WIDTH = 256
N_CLASSES = 3
RANDOM_STATE = 2023


class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, list_IDs, df, base_path: str, mode='fit', 
                 HEIGHT: int = 256, WIDTH: int = 256, 
                 batch_size: int = 32, n_channels: int = 3, 
                 n_classes: int = 3, random_state: int = 2023, shuffle=True):
        
        self.dim = (HEIGHT, WIDTH)
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        
        X = self.__generate_X(list_IDs_batch)
        
        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            return X, y
        
        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
    
    def __generate_X(self, list_IDs_batch):
        '''Retrieve image'''
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Get image
        for i, ID in enumerate(list_IDs_batch):
            image_df = self.df[self.df['name'] == ID]
            img_path = f"{self.base_path}/{image_df.iloc[0,5]}"
            img = self.__load_grayscale(img_path)
            # Store samples
            X[i,] = img 
        return X

    def __generate_y(self, list_IDs_batch):
        '''Restore image label'''
        # Initialization
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        # Get annotations        
        for i, ID in enumerate(list_IDs_batch):
            image_df = self.df[self.df['name'] == ID]
            original_dims = (image_df.iloc[0,3], image_df.iloc[0,2])            
            rles = image_df['label'].values
            masks = P.build_masks(rles, original_dims)
            masks = cv2.resize(masks, self.dim)
            masks = np.expand_dims(masks, axis=-1)
            y[i, ] = masks
        return y
    
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # resize image
        img = cv2.resize(img, self.dim, interpolation=cv2.INTER_AREA)
        # normalize
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)
        return img
    
    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        return img
    

class ImageSequence(tf.keras.utils.Sequence): 

    def __init__(self, contents: List[str], HEIGHT: int = 256, WIDTH: int = 256, 
                 batch_size: int = 1, n_channels: int = 3):
        self.contents = contents
        self.dim = (HEIGHT, WIDTH)
        self.batch_size = batch_size if len(self.contents) >= batch_size else 1
        self.n_channels = n_channels

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.contents) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Initiate with empty numpy array
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        low = index * self.batch_size
        # The last batch may be smaller if the length of contents is not a multiple of batch size
        high = min(low + self.batch_size, len(self.contents))
        for i, path in enumerate(self.contents[low:high]):
            image = self.read_image(path)
            X[i, ] = image

        return X
    
    def read_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.dim, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.
        image = np.expand_dims(image, axis=-1)

        return image
    

class ImageSequenceFromBytes(tf.keras.utils.Sequence): 

    def __init__(self, contents: List[bytes], HEIGHT: int = 256, WIDTH: int = 256, 
                 batch_size: int = 10, n_channels: int = 3):
        self.contents = contents
        self.dim = (HEIGHT, WIDTH)
        self.batch_size = batch_size if len(self.contents) >= batch_size else 1
        self.n_channels = n_channels

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.contents) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Initiate with empty numpy array
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        low = index * self.batch_size
        # The last batch may be smaller if the length of contents is not a multiple of batch size
        high = min(low + self.batch_size, len(self.contents))
        for i, _bytes in enumerate(self.contents[low:high]):
            image = self.read_bytes(_bytes)
            X[i,] = image

        return X

    def read_bytes(self, _bytes):
        """Read image from in-memory buffer"""
        image = Image.open(_bytes).convert('L')
        image = np.array(image, dtype=np.float32)
        image = cv2.resize(image, self.dim, interpolation=cv2.INTER_AREA)
        image = image / 255.
        image = np.expand_dims(image, axis=-1)

        return image