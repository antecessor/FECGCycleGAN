from __future__ import print_function, division

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras import Input, Model
from keras.layers import Conv1D, UpSampling1D, LeakyReLU, Dropout, Lambda, Embedding, Bidirectional, LSTM, Dense, Flatten, Layer, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.optimizers import Nadam
from keras_contrib.layers import InstanceNormalization
from keras_self_attention import ScaledDotProductAttention
from keras_self_attention.backend import regularizers
from sklearn.preprocessing import scale

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Normalization

from Utils.DataUtils import DataUtils


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # self.ffn = Sequential(
        #     [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
        # )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        # self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        # self.dropout2 = Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs * attn_output)
        # ffn_output = self.ffn(out1)
        # ffn_output = self.dropout2(ffn_output)
        return out1


class CycleGAN:
    def __init__(self, row, col):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Restrict TensorFlow to only use the fourth GPU
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
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
        self.lambda_cycle = 4.0  # Cycle-consistency loss
        self.lambda_id = 0.01 * self.lambda_cycle  # Identity loss

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
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

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
                                       fake_B, fake_A,
                                       reconstr_A, reconstr_B])
        self.combined.compile(loss=['logcosh', 'logcosh',
                                    'logcosh', 'logcosh', 'logcosh',
                                    'logcosh'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

    # Define custom loss
    def custom_loss(self):

        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true, y_pred):
            return K.mean(y_true * K.log(y_true / y_pred + K.epsilon()))

        # Return a function
        return loss

    def build_generator(self):
        """U-Net Generator"""

        def conv1DWithSINE(layer_input, filters, f_size=7):
            """Layers used during downsampling"""
            d = Conv1D(filters, kernel_size=f_size, padding='same', activation=tf.math.sin)(layer_input)
            d = InstanceNormalization()(d)
            return d

        def multiply(x):
            mask,image  = x
            return image* K.clip(mask,0.8,1)

        input = Input(shape=self.img_shape)

        value = conv1DWithSINE(input, 1, f_size=30)


        att = TransformerBlock(200, 2)(value)
        att = Normalization(axis=1)(att)

        remainedInput = Lambda(multiply)([att, value])

        output_img = conv1DWithSINE(remainedInput, 13, f_size=3)
        output_img = conv1DWithSINE(output_img, 7, f_size=5)
        output_img = conv1DWithSINE(output_img, 5, f_size=13)
        output_img = conv1DWithSINE(output_img, 1, f_size=3)

        return Model(input, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv1D(filters, kernel_size=f_size, padding='same',activation=tf.math.sin)(layer_input)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df)
        d2 = d_layer(d1, 7)
        d3 = d_layer(d2, 3)
        validity = d_layer(d3, 1)


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

                # Translate images to the other domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)
                # Translate images back to original domain
                reconstr_A = self.g_BA.predict(fake_B)
                reconstr_B = self.g_AB.predict(fake_A)
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
                                                       imgs_B, imgs_A,
                                                       reconstr_A, reconstr_B])

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

        # gen_imgs[1] = gen_imgs[0]-gen_imgs[1]
        try:
            for i in range(r):
                for j in range(c):
                    for bias in range(1):
                        gen_imgs[cnt][:, bias] = scale(self.dataUtils.butter_bandpass_filter(gen_imgs[cnt][:, bias], 10, 50, 200, axis=0), axis=0)
                        if np.max(gen_imgs[cnt][:, bias]) != 0:
                            gen_imgs[cnt][:, bias] = gen_imgs[cnt][:, bias] / np.max(gen_imgs[cnt][:, bias])
                        axs[i, j].plot(gen_imgs[cnt][:, bias] + bias)
                    axs[i, j].set_title(titles[j])
                    cnt += 1
            fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
            plt.close()
        except:
            pass
