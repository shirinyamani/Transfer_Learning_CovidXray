import numpy as np
import nibabel as nib
import matplotlib.pylab as plt
from tensorflow import keras
from tensorflow.keras import backend as K
import h5py
import glob

class DataGeneratorUnet(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, imgs_list, masks_list,  patch_size = (128,128), batch_size = 32, shuffle = True):

        self.imgs_list = imgs_list 
        self.masks_list = masks_list
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.nsamples = len(imgs_list)
        self.shuffle = True
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.imgs_list)//self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, Y = self.__data_generation(batch_indexes)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nsamples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        'Generates data containing batch_size samples'

        # Initialization
        X = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], 1))
        Y = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], 1))
        
        for (jj,ii) in enumerate(batch_indexes):
            
            aux_img = np.load(self.imgs_list[ii])
            aux_mask = np.load(self.masks_list[ii]) 
            
            # Implement data augmentation function
            
            aux_img_patch,aux_mask_patch = self.__extract_patch(aux_img,aux_mask)
            
            X[jj,:,:,0] = aux_img_patch
            Y[jj,:,:,0] = aux_mask_patch

        return X,Y
     
    def __extract_patch(self, img, mask):
        crop_idx = [None]*2
        crop_idx[0] = np.random.randint(0, img.shape[0] - self.patch_size[0])
        crop_idx[1] = np.random.randint(0, img.shape[1] - self.patch_size[1])
        img_cropped =  img[crop_idx[0]:crop_idx[0] + self.patch_size[0],\
                              crop_idx[1]:crop_idx[1] + self.patch_size[1]]
        mask_cropped = mask[ crop_idx[0]:crop_idx[0] + self.patch_size[0], \
                          crop_idx[1]:crop_idx[1] + self.patch_size[1]]
        return img_cropped,mask_cropped
