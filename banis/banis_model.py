#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Bidirectional Adversarial Networks for microscopic Image Synthesis (BANIS)    
@topic: BANIS model
@author: junzhuang, daliwang
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, \
    Conv2DTranspose, Conv2D,  Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError as loss_mse
from utils import convert2binary, dice_coefficient


class BANIS():
    def __init__(self, img_shape):
        # Initialize input shape
        self.img_shape = img_shape # (64, 64, 1) (img_rows, img_cols, channels)
        self.latent_dim = 100

        # Optimizer
        self.lr = 1e-5
        self.beta1 = 0.5
        self.D_optA = Adam(2*self.lr, self.beta1)
        self.G_optA = Adam(2*self.lr, self.beta1)
        self.E_optA = SGD(5*self.lr)
        self.D_optB = Adam(2*self.lr, self.beta1)
        self.G_optB = Adam(self.lr, self.beta1)
        self.E_optB = SGD(5*self.lr)

        # Build the encoder
        self.encoderA = self.encoder_model()
        self.encoderB = self.encoder_model()

        # Build the generator
        self.generatorA = self.generator_model()
        self.generatorB = self.generator_model()

        # Build and compile the discriminator
        self.discriminatorA = self.discriminator_model()
        self.discriminatorA.trainable = True
        self.discriminatorA.compile(loss=['binary_crossentropy'],
                                   optimizer=self.D_optA,
                                   metrics=['accuracy'])
        self.discriminatorB = self.discriminator_model()
        self.discriminatorB.trainable = True
        self.discriminatorB.compile(loss=['binary_crossentropy'],
                                   optimizer=self.D_optB,
                                   metrics=['accuracy'])

        # Build the pioneer model
        self.discriminatorA.trainable = False
        self.pioneerA = Sequential([self.generatorA, self.discriminatorA])
        self.pioneerA.compile(loss=['binary_crossentropy'],
                              optimizer=self.G_optA,
                              metrics=['accuracy'])
        self.discriminatorB.trainable = False
        self.pioneerB = Sequential([self.generatorB, self.discriminatorB])
        self.pioneerB.compile(loss=['binary_crossentropy'],
                              optimizer=self.G_optB,
                              metrics=['accuracy'])

        # Build the successor model
        self.successorA = Sequential([self.encoderA, self.generatorA])
        self.successorA.compile(loss=loss_mse(),
                                optimizer=self.E_optA)
        self.successorB = Sequential([self.encoderB, self.generatorB])
        self.successorB.compile(loss=loss_mse(),
                                optimizer=self.E_optB)

        # Build the coordinator model
        self.coordinatorA = Sequential([self.successorA, self.successorB])
        self.coordinatorA.compile(loss=loss_mse(),
                                  optimizer=self.E_optA)
        self.coordinatorB = Sequential([self.successorB, self.successorA])
        self.coordinatorB.compile(loss=loss_mse(),
                                  optimizer=self.E_optB)

        # The dir for logs and checkpoint
        self.log_dir = "./TB_logs_baait"
        self.checkpoint_dir = './Train_Checkpoints_baait'


    def encoder_model(self, depth=128):
        model = Sequential()
        model.add(Conv2D(depth, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                         input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(2*depth, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(4*depth, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(self.latent_dim))
        model.summary()
        return model

    def generator_model(self, dim=8, depth=256):
        model = Sequential()
        model.add(Dense(dim*dim*depth, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((dim, dim, depth)))
        assert model.output_shape == (None, dim, dim, depth)
        model.add(Conv2DTranspose(depth, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal'))
        assert model.output_shape == (None, dim, dim, depth)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(int(depth//2), (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal'))
        assert model.output_shape == (None, 2*dim, 2*dim, int(depth//2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(int(depth//4), (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal'))
        assert model.output_shape == (None, 4*dim, 4*dim, int(depth//4)) # for 64x64
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(self.img_shape[2], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        model.summary()
        return model

    def discriminator_model(self, depth=64, drop_rate=0.3):
        model = Sequential()
        model.add(Conv2D(depth, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                         input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(drop_rate))
        model.add(Conv2D(2*depth, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(drop_rate))
        #model.add(Conv2D(4*depth, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_normal'))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(drop_rate)) # ---
        # Out: 1-dim probability
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        return model


    def train(self, A, B, EPOCHS=100, BATCH_SIZE=128, WARMUP_STEP=20, NUM_IMG=5):
        # Define the groundtruth
        Y_real = np.ones((BATCH_SIZE, 1))
        Y_rec = np.zeros((BATCH_SIZE, 1)) # Reconstructed Label
        Y_both = np.concatenate((Y_real, Y_rec), axis=0)

        # Log for TensorBoard
        summary_writer = tf.summary.create_file_writer(self.log_dir)

        # Initialize the checkpoint
        interval = int(EPOCHS // 5) if EPOCHS >= 10 else 5
        checkpoint_path = os.path.join(self.checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(E_optimizerA=self.E_optA,
                                         G_optimizerA=self.G_optA,
                                         D_optimizerA=self.D_optA,
                                         E_optimizerB=self.E_optB,
                                         G_optimizerB=self.G_optB,
                                         D_optimizerB=self.D_optB,
                                         encoderA=self.encoderA,
                                         generatorA=self.generatorA,
                                         discriminatorA=self.discriminatorA,
                                         pioneerA=self.pioneerA,
                                         successorA=self.successorA,
                                         coordinatorA=self.coordinatorA,
                                         encoderB=self.encoderB,
                                         generatorB=self.generatorB,
                                         discriminatorB=self.discriminatorB,
                                         pioneerB=self.pioneerB,
                                         successorB=self.successorB,
                                         coordinatorB=self.coordinatorB)

        # Restore the latest checkpoint in checkpoint_dir
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        A_gen_list, B_gen_list, AB_rec_list = [], [], []
        match_cnt, total_cnt = 0, 0 # Initialize the counting for matching DSC
        NUM_BATCH = len(A) // BATCH_SIZE  # np.ceil()
        for epoch in range(EPOCHS):
            for nb in range(NUM_BATCH-1):
                # ---Pretrain Stage ---
                # Select real instances batch by batch
                step = int(epoch * NUM_BATCH + nb)
                idx = np.arange(nb*BATCH_SIZE, nb*BATCH_SIZE+BATCH_SIZE)
                A_real = A[idx, :, :, :]

                # Generate a batch of latent variables based on uniform distribution
                z_A = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, self.latent_dim])
                A_gen = self.generatorA.predict(z_A)
                # Train the discriminator (real for 1 and rec for 0) ------
                A_both = np.concatenate((A_real, A_gen))
                Dloss_A = self.discriminatorA.train_on_batch(A_both, Y_both)
                # Train the pioneer model to fool the discriminator ------
                Gloss_A = self.pioneerA.train_on_batch(z_A, Y_real)

                # Repeat the same procedure as above for B
                B_real = B[idx, :, :, :]
                z_B = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, self.latent_dim])
                B_gen = self.generatorB.predict(z_B)  
                B_both = np.concatenate((B_real, B_gen))
                Dloss_B = self.discriminatorB.train_on_batch(B_both, Y_both)
                Gloss_B = self.pioneerB.train_on_batch(z_B, Y_real)

                # Train the successor & coordinator when epoch > WARMUP_STEP ------
                mse = -1
                if epoch > WARMUP_STEP:
                    mseA_gen = self.successorA.train_on_batch(B_real, A_gen)
                    mseB_gen = self.successorB.train_on_batch(A_real, B_gen)
                    mseA_real = self.successorA.train_on_batch(B_real, A_real)
                    mseB_real = self.successorB.train_on_batch(A_real, B_real)
                    mse_B2B = self.coordinatorA.train_on_batch(B_real, B_real)
                    mse_A2A = self.coordinatorB.train_on_batch(A_real, A_real)
                    mse_A = 0.5 * np.add(mseA_real, mseA_gen)
                    mse_B = 0.5 * np.add(mseB_real, mseB_gen)
                    identity_loss = 0.5 * np.add(mse_A, mse_B)
                    pair_matched_loss = 0.5 * np.add(mse_A2A, mse_B2B)
                    mse = np.mean([identity_loss, pair_matched_loss], axis=0)

                # For Experiments and Visualization ---------------------
                # Save scalars into TensorBoard
                with summary_writer.as_default():
                    tf.summary.scalar('D_loss_A', Dloss_A[0], step=step)
                    tf.summary.scalar('G_loss_A', Gloss_A[0], step=step)
                    tf.summary.scalar('D_acc_A', Dloss_A[1], step=step)
                    tf.summary.scalar('D_loss_B', Dloss_B[0], step=step)
                    tf.summary.scalar('G_loss_B', Gloss_B[0], step=step)
                    tf.summary.scalar('D_acc_B', Dloss_B[1], step=step)
                    if mse != -1:
                        tf.summary.scalar('MSE_A', mse_A, step=step)
                        tf.summary.scalar('MSE_B', mse_B, step=step)
                        tf.summary.scalar('Identity_Loss', identity_loss, step=step)
                        tf.summary.scalar('Pair_Matched_Loss', pair_matched_loss, step=step)
                        tf.summary.scalar('MSE', mse, step=step)

                # Save the checkpoint at given interval
                if (step + 1) % int(interval*BATCH_SIZE) == 0:
                    checkpoint.save(file_prefix=str(checkpoint_path))

                # Schedule the learning rate
                if (epoch + 1) % 100000 == 0:
                    self.lr = self.lr_scheduler(self.lr, Type="Periodic", epoch=epoch, period=100000)
                    # Kears callback: https://keras.io/zh/callbacks/

                # Store the generated/reconstructed samples
                if (step + 1) % 100 == 0:
                    z_gen = np.random.normal(size=(int(NUM_IMG), self.latent_dim))
                    A_gen_list.append(self.prediction(self.generatorA, z_gen))
                    B_gen_list.append(self.prediction(self.generatorB, z_gen))
                if mse != -1 and (step + 1) % 100 == 0:
                    # Prediction
                    A_rec = self.prediction(self.successorA, B_real)
                    B_rec = self.prediction(self.successorB, A_real)
                    for i in range(len(idx)):
                        total_cnt += 1
                        # Reshape image to 2D size
                        A_rec_i = A_rec[i].reshape(self.img_shape[0], self.img_shape[1])
                        B_rec_i = B_rec[i].reshape(self.img_shape[0], self.img_shape[1])
                        # Get the binary masks of images
                        A_rec_i_bi = convert2binary(A_rec_i, A_rec_i.max()*0.6)
                        B_rec_i_bi = convert2binary(B_rec_i, B_rec_i.max()*0.3)
                        # Compute the DSC
                        DSC = dice_coefficient(A_rec_i_bi, B_rec_i_bi)
                        # Compute matching_index & select rec samples
                        if DSC < 0.2:
                            match_cnt += 1
                            AB_rec_list.append([A_rec[i], B_rec[i]])

                # Plot the progress
                print("A: No.{0}: D_loss: {1}; D_acc: {2}; G_loss: {3}."\
                      .format(step, Dloss_A[0], Dloss_A[1], Gloss_A[0]))
                print("B: No.{0}: D_loss: {1}; D_acc: {2}; G_loss: {3}."\
                      .format(step, Dloss_B[0], Dloss_B[1], Gloss_B[0]))
                if mse != -1:
                    print("Total MSE: {0}.".format(mse))
                print("----------")

        # Save file
        np.save("./A_gen_baait.npy", A_gen_list)
        np.save("./B_gen_baait.npy", B_gen_list)
        np.save("./AB_rec_baait.npy", AB_rec_list)
        checkpoint.save(file_prefix = str(checkpoint_path))

        # Evaluation
        print("Evaluation:")
        if total_cnt != 0:
            matching_index = match_cnt/total_cnt
            print("Matching Index: ", matching_index)


    def prediction(self, model, inputs):
        # Prediction with given model and inputs
        x_pred = model.predict(inputs)
        x_pred = 127.5 * x_pred + 127.5
        return x_pred

    def lr_scheduler(self, lr, Type, epoch, period=100):
        # Schedule the learning rate
        if Type == "Periodic":
            if epoch < int(period):
                lr = lr
            elif epoch % int(period) == 0:
                lr = lr*0.5
        return lr
