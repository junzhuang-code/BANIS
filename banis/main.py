#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Bidirectional Adversarial Networks for microscopic Image Synthesis (BANIS)    
@topic: main function
@author: junzhuang, daliwang
@references:
    1. https://github.com/notenot/mnist-keras-dcgan/blob/master/mnist-keras-dcgan.ipynb
    2. https://github.com/vwrs/dcgan-mnist
    3. https://www.tensorflow.org/tutorials/generative/dcgan?hl=zh-cn
    4. https://keras.io/zh/initializers/
"""

import sys
import numpy as np
from utils import read_pickle, ElapsedTimer, plot_samples
from banis_model import BANIS


#if __name__ == '__main__':
# Initialize the arguments
try:
    n_epochs = int(sys.argv[1])
    n_step = int(sys.argv[2])
    is_trainable = bool(sys.argv[3])
except:
    n_epochs = 100
    n_step = 20
    is_trainable = True

if is_trainable:
    # Read the pickle file
    Data_A = read_pickle('./Data/Data_Train/Data_Left_train.pkl')
    Data_B = read_pickle('./Data/Data_Train/Data_Right_train.pkl')
    print("Data A/B: ", Data_A.shape, Data_B.shape)
    # Initialize the model
    assert Data_A.shape == Data_B.shape
    if len(Data_A.shape) == 4 and len(Data_B.shape) == 4:
        img_shape = (Data_A.shape[1], Data_A.shape[2], Data_A.shape[3])
        banis = BANIS(img_shape)
    else:
        print("The shape of input dataset don't match!!!")
    # Train the model and record the runtime
    timer = ElapsedTimer()
    banis.train(Data_A, Data_B, EPOCHS=n_epochs, BATCH_SIZE=128, WARMUP_STEP=n_step, NUM_IMG=5)
    timer.elapsed_time()
else:
    # Plotting the sampling images
    A_gen_list = np.load("./A_gen_baait.npy")
    plot_samples(A_gen_list, name='Agen')
    B_gen_list = np.load("./B_gen_baait.npy")
    plot_samples(B_gen_list, name='Bgen')
    AB_rec_list = np.load("./AB_rec_baait.npy")
    plot_samples(AB_rec_list, name='ABrec')
