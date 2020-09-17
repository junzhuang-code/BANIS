#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Bidirectional Adversarial Networks for microscopic Image Synthesis (BANIS)    
@topic: Utils functions
@author: junzhuang, daliwang
"""

import os
import time
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_pickle(file_name):
    """Reload the dataset"""
    with open (file_name,'rb') as file:
        return pickle.load(file)

def dump_pickle(file_name, data):
    """Export the dataset"""
    with open (file_name,'wb') as file:
        pickle.dump(data, file)

def batch_select(file_in, file_out, types="right"):
    """
    @topic: Select and crop the iamges in batch
    @input: file_in/file_out: the file name of input&output file; types: left/right (str)
    """
    path_total = os.getcwd() # Get current pwd
    path_split = path_total+"/{0}".format(file_in)
    path_selected = path_total+"/{0}".format(file_out)
    r0, size = 0, 512 # The parameters of cropping 
    if types == "right": # Select right side image
        c0 = 512
    elif types == "left": # # Select left side image
        c0 = 0
    else:
        return "Please select 'left' or 'right'. "
    for file in os.listdir(path_split):
        if file != ".DS_Store":
            path_sub = path_split+"/{0}".format(file)
            path_out = path_selected+"/{0}".format(file)
            for image in os.listdir(path_sub):
                image_id = int(image[10:-4]) # the number before .jpg
                if image_id >=6 and image_id <= 20:
                    path_img = path_sub+'/{0}'.format(image)
                    img = plt.imread(path_img)
                    #img = color.rgb2gray(img)
                    img_crop = img[r0:(r0 + size), c0:(c0 + size)]
                    plt.imsave(path_out+"/{0}".format(image), img_crop, cmap='gray')

def make_split_dataset(file_X, img_CropCodn, DATA_TYPE='train', CUT_RATIO=1.0, FACTOR=4):
    """
    @topic: Preprocess, split the image and generate the dataset
    @input:
        file_X (string): the file name of dataset
        img_CropCodn (list: Num_SubImage x 4): image coordinate for cropping
        DATA_TYPE (string): select to make training/testing set
        CUT_RATIO (float): the percentage of training set
        FACTOR (int): the rescaled factor for shrinking the image
    """
    path_total = os.getcwd() # Get current pwd
    path_X = path_total+"/{0}".format(file_X) # Path of dataset
    dirs = os.listdir(path_X) # The list of directory for dataset
    cut = int(len(dirs)*CUT_RATIO) # Decide the position of cutting
    if DATA_TYPE == 'train':
        file_out = 'Data_Train'
        dirs = dirs[:cut]
    elif DATA_TYPE == 'test':
        file_out = 'Data_Test'
        dirs = dirs[cut:]
    path_out = path_total+"/Data/{0}".format(file_out) # Path of output
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    data = []
    for i, file in enumerate(dirs): # Assume that file_X & file_Y have the same organization
        if file != ".DS_Store":
            path_X_file = path_X+"/{0}".format(file)
            for image in os.listdir(path_X_file):
                if image != ".DS_Store":
                    path_X_img = path_X_file+'/{0}'.format(image)
                    img_X = cv2.imread(path_X_img)
                    X_max_half = 0.5 * img_X.max()
                    img_X = cv2.cvtColor(img_X, cv2.COLOR_RGB2GRAY)
                    # Employ Gaussian Filter
                    img_X = cv2.GaussianBlur(img_X, ksize=(5,5), sigmaX=1.5)
                    # Resize image to 128 x 128
                    img_X_row, img_X_col = img_X.shape[0], img_X.shape[1] # size=512
                    img_X = cv2.resize(img_X, (img_X_row//FACTOR, img_X_col//FACTOR), \
                                       interpolation=cv2.INTER_LANCZOS4)
                    # Crop, pad & normalize image respectively
                    img_X_row, img_X_col = img_X.shape[0], img_X.shape[1]
                    for i in range(len(img_CropCodn)):
                        # Crop the image
                        r_start, r_end, c_start, c_end = img_CropCodn[i]
                        img_X_i = img_X[r_start:r_end, c_start:c_end]
                        if img_X_i.shape[0] < img_X_i.shape[1]:
                            img_X_i = img_X_i.T
                        # Pad the image
                        img_X_i = pad_image(img_X_i, \
                                            row_new=img_X_row//2, col_new=img_X_col//2)
                        img_X_i = (img_X_i.astype(np.float32) - X_max_half) / X_max_half
                        img_X_i = np.expand_dims(img_X_i, axis=2)
                        data.append(img_X_i)
    dump_pickle(path_out+"/{0}_{1}.pkl".format(file_X, DATA_TYPE), np.array(data))

def pad_image(img, row_new=64, col_new=64):
    """Pad the cropped image"""
    row_old, col_old = img.shape[0], img.shape[1]
    img_new = np.zeros((row_new, col_new))
    # Compute the padding offset
    row_offset = (row_new - row_old) // 2
    col_offset = (col_new - col_old) // 2
    # Copy the old image into center of new image
    img_new[row_offset:row_offset+row_old, col_offset:col_offset+col_old] = img
    # Fill the background color
    bg_px = np.mean(img[:3])
    for i in range(len(img_new)):
        for j in range(len(img_new[i])):
            if img_new[i][j] == 0:
                img_new[i][j] = bg_px
    return img_new

def resize_coordinate(cor_list, FACTOR):
    """Given a list of coordinate and resizing factor, adjust the scale."""
    return [cor_list[i]//FACTOR for i in range(len(cor_list))]

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed Time: %s " % self.elapsed(time.time() - self.start_time))

def convert2binary(img, threshold):
    """
    @topic: Get binary mask of image
    @input: img(2D-array), threshold(float).
    @return: img_bi(2D-array).
    """
    img_bi = np.zeros(img.shape, dtype=np.int32)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > threshold:
                img_bi[i][j] = 1
            else:
                img_bi[i][j] = 0
    return img_bi

def dice_coefficient(gt, seg):
    """
    @topic: Compute dice coefficient
    @input: gt/seg(2D-array); return: DSC(float).
    """
    assert gt.shape == seg.shape
    return np.sum(seg[gt==seg])*2.0 / (np.sum(seg) + np.sum(gt))

def plot_samples(x_rec_list, name='gen'):
    """
    @topic: Plot all reconstructed samples along epochs
    @input: x_rec_list, size=(num_sampling, num_img, rows, cols, channel).
    """
    cur_path = os.getcwd()
    if not os.path.exists('{0}/Images'.format(cur_path)):
        os.makedirs('Images')
    plt.figure(figsize=(5,5))
    for s in range(len(x_rec_list)):
        for i in range(x_rec_list[s].shape[0]):
            plt.subplot(1, x_rec_list[s].shape[0], i+1)
            #x_rec_s_i = x_rec_list[s][i][:,:,0]
            x_rec_s_i = np.squeeze(x_rec_list[s][i], axis=2)
            plt.imshow(x_rec_s_i, interpolation='nearest', cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig("Images/Img_baait_{0}_{1}.png".format(name, s+0))
        plt.close()
