from __future__ import print_function, division

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dropout, Concatenate, Lambda, Flatten, GRU, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, UpSampling1D
from keras.models import Model
from keras.optimizers import Adam, Nadam, RMSprop
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import keras.backend as K
from sklearn.preprocessing import scale

from Utils.DataUtils import DataUtils
from deeplearning.AttentionDecoder import AttentionDecoder


class CycleGAN:
    def __init__(self, row, col):
        # Input shape
        self.dataUtils = DataUtils()
        self.img_rows = row
        self.img_cols = col
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols)

        # Configure data loader
        self.dataset_name = 'ECG2FECG'

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (200, 1)

        # Number of filters in the first layer of G and D
        self.gf = 6
        self.df = 12

        # Loss weights
        self.lambda_cycle = 10.0  # Cycle-consistency loss
        self.lambda_id = 0.2 * self.lambda_cycle  # Identity loss

        optimizer = Nadam()

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mae',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss='mae',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator("relu")
        self.g_BA = self.build_generator("relu")

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=[self.custom_loss(), self.custom_loss(),
                                    self.custom_loss(), self.custom_loss(),
                                    self.custom_loss(), self.custom_loss()],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

    # Define custom loss
    def custom_loss(self):

        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true, y_pred):
            return K.mean(K.abs(y_pred - y_true), axis=-1)

        # Return a function
        return loss

    def build_generator(self, outputLayer="sigmoid"):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=3):
            """Layers used during downsampling"""
            d = Conv1D(filters, kernel_size=f_size, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling1D()(layer_input)
            u = Conv1D(filters, kernel_size=f_size, strides=2, padding='same', activation='sigmoid')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # # Downsampling
        d1 = conv2d(d0, self.gf)
        # d2 = GRU(16, return_sequences=True)(d1)

        d3 = AttentionDecoder(32, 16)(d1)
        # Upsampling
        u3 = deconv2d(d3, d1, self.gf)
        # u2 = deconv2d(u3, d3, self.gf)
        # u4 = GRU(16, return_sequences=True)(u3)
        # output_img = Conv2D(1, 3, padding="same",str activation=outputLayer)(u4)
        output_img = Conv1D(4, kernel_size=2, padding='same', activation=outputLayer)(u3)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv1D(filters, kernel_size=f_size, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv1D(1, kernel_size=4, padding='same')(d4)

        return Model(img, validity)

    def train(self, x_train, y_train, epochs, batch_size=1, sample_interval=20):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, imgs_A in enumerate(x_train):
                imgs_B = y_train[batch_i]
                imgs_B = np.reshape(imgs_B, (-1, x_train.shape[1], x_train.shape[2]))
                imgs_A = np.reshape(imgs_A, (-1, x_train.shape[1], x_train.shape[2]))
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                      [valid, valid,
                                                       imgs_A, imgs_B,
                                                       imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                      % (epoch, epochs,
                         batch_i, 1,
                         d_loss[0], 100 * d_loss[1],
                         g_loss[0],
                         np.mean(g_loss[1:3]),
                         np.mean(g_loss[3:5]),
                         np.mean(g_loss[5:6]),
                         elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, imgs_A, imgs_B)
        self.g_AB.save("ECG2FECG.h5", overwrite=True)
        self.g_BA.save("FECG2ECG.h5", overwrite=True)

    def sample_images(self, epoch, batch_i, imgs_A, imgs_B):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        # Demo (for GIF)
        # imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        # imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                for bias in range(4):
                    gen_imgs[cnt][:, bias] = scale(self.dataUtils.butter_bandpass_filter(gen_imgs[cnt][:, bias], 10, 50, 200, axis =0), axis=0)
                    axs[i, j].plot(gen_imgs[cnt][:, bias] / np.max(gen_imgs[cnt][:, bias]) + bias)
                axs[i, j].set_title(titles[j])
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()
