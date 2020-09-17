#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Bidirectional Adversarial Networks for microscopic Image Synthesis (BANIS)    
@topic: Preprocessing
@author: junzhuang, daliwang
"""

from utils import batch_select, resize_coordinate, make_split_dataset

# Split images to left/right side
# Need to get "Data_Split" data
# Need to prepare empty dir Data_Right and Data_Left.
batch_select('Data_Split', 'Data_Right', 'right')
batch_select('Data_Split', 'Data_Left', 'left')
# Generate the dataset
cut_ratio = 1.0
rescaled_factor = 4 # shrink image to 1/(2*rescaled_factor)
# Three coordinates for cropping image
img01_CropCodn = [60, 310, 80, 260] # r_start, r_end, c_start, c_end
img02_CropCodn = [80, 330, 250, 440]
img03_CropCodn = [300, 480, 160, 400]
img_CropCodn = [resize_coordinate(img01_CropCodn, rescaled_factor), \
            resize_coordinate(img02_CropCodn, rescaled_factor), \
            resize_coordinate(img03_CropCodn, rescaled_factor)]
# Only name for "training"
make_split_dataset('Data_Left', img_CropCodn, 'train', cut_ratio, rescaled_factor)
make_split_dataset('Data_Right', img_CropCodn, 'train', cut_ratio, rescaled_factor)
