# BSD 3-Clause License
#
# Copyright (c) 2021, Manuel Kraus
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as bk
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, \
    AveragePooling2D, Flatten, Dropout, Activation, Dense
from tensorflow.keras.regularizers import l2
import numpy as np
import os
import datetime


class TriggerModel:
    """ Trigger bot model used for inference """

    def __init__(self, config, image_shape, model_path):
        """ Initialize the model

        :param config configuration object
        :param image_shape (height,width) of used fov
        :param model_path path to the model file
        """

        self._config = config
        self._image_shape = image_shape
        self._model_path = model_path

        self._model = None
        self._aim_mask = _AimMask(self._image_shape, self._config.trigger_aim_size)

    def init_inference(self):
        """ Initialize trigger bot inference"""

        self._model = keras.models.load_model(self._model_path)

    def predict_is_on_target(self, image):
        """ Returns true if aim is on-target """

        image = self._preprocess_image(image)
        prediction = self._model(image)

        return np.argmax(prediction) == 1

    def shutdown_inference(self):

        self._model = None

    @tf.function
    def _preprocess_image(self, image):

        image = tf.cast(image, tf.float32)
        image = tf.multiply(image, self._aim_mask.mask)
        image = (image - tf.reduce_mean(image)) / tf.math.reduce_std(image)

        return image


class TrainableTriggerModel(TriggerModel):
    """ Trigger bot training model, builds on top of the inference model """

    def __init__(self, config, image_shape, model_path):
        """ Initialize the model

        :param config configuration object
        :param image_shape (height,width) of used fov
        :param model_path path to the model file
        """

        super(TrainableTriggerModel, self).__init__(config, image_shape, model_path)

        self._model = self._create_trigger_model(image_shape)
        self._optimizer = keras.optimizers.Adam(lr=0.001)
        self._model.compile(optimizer=self._optimizer,
                            loss='categorical_crossentropy', metrics=['accuracy'])
        self._current_epoch = 0  # relative to initialize time

        #####################################
        #        Training Callbacks         #
        #####################################

        self._training_callbacks = []
        self._training_callbacks.append(keras.callbacks.ModelCheckpoint(self._model_path, monitor='val_loss',
                                                                        verbose=1, save_best_only=True, mode='min'))
        self._training_callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                                          patience=10, min_lr=1e-7, verbose=1))

        logdir = os.path.join(os.path.dirname(self._model_path),
                              "logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._training_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))

        self._image_logger = ImageLogger(name='Images', logdir=logdir, max_images=2)

    def fit_one_epoch(self, data_train, data_test):
        """ Fit for one epoch. Model will be reloaded if existing to keep on training """

        try:
            self._model = keras.models.load_model(self._model_path)
            bk.set_value(self._model.optimizer.learning_rate,
                         float(self._config.trigger_train_lr))
        except (ImportError, IOError):
            self._current_epoch = 0

        self._model.fit(x=data_train, validation_data=data_test, initial_epoch=self._current_epoch,
                        epochs=self._config.trigger_train_epochs+self._current_epoch,
                        callbacks=self._training_callbacks)
        self._current_epoch += self._config.trigger_train_epochs
        self._config.trigger_train_lr = bk.get_value(self._model.optimizer.learning_rate)

    def create_dataset(self, image_paths_labels, augment=True, shuffle=True):
        """ Create the a tensorflow data set which can be used for training/testing

            :param image_paths_labels array of (image_path, label) tuples
            :param augment flag whether to perform image augmentation
            :param shuffle flag whether to perform shuffling
            :returns tf.data.Dataset
        """

        dataset = tf.data.Dataset.from_generator(lambda: image_paths_labels, (tf.string, tf.int32))

        # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Augmentation
        if augment:
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Set batch size
        dataset = dataset.batch(self._config.trigger_train_batch_size)

        dataset = dataset.map(self._image_logger, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Set the number of datapoints you want to load and shuffle
        if shuffle:
            dataset = dataset.shuffle(self._config.trigger_train_shuffle_size)

        dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    @tf.function
    def _parse_function(self, image_path, label):
        """ Load an image using a path and provide the image + label """

        bits = tf.io.read_file(image_path)
        image = tf.image.decode_png(bits, channels=3)
        image = self._preprocess_image(image)

        label = tf.one_hot(label, 2)
        label = tf.reshape(label, (2,))

        return image, label

    @tf.function
    def _augment(self, image, label):
        """ Perform image augmentation, I think we could do more here """

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)

        return image, label

    def _create_trigger_model(self, image_shape):
        """ Create the keras model itself (not compiled yet)

            The model itself is inspired by GoogleNet.
            I take the input block + the first two inception layers.
        """

        # Input block of GoogleNet
        # Get rid of first 7x7 convolution since trigger bot fov is already quite small
        image_input = keras.Input(shape=(image_shape[0], image_shape[1], 3))

        # Input block of GoogleNet
        # Get rid of first 7x7 convolution and replace with 3x3 since we are already quite small
        input_block_conv_1 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid',
                                    activation='relu', kernel_regularizer=l2())(image_input)
        input_block_pool_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid',)(input_block_conv_1)
        input_block_conv_3 = Conv2D(filters=192, kernel_size=3, strides=(1, 1), padding='same',
                                    activation='relu', kernel_regularizer=l2())(input_block_pool_2)
        input_block_pool_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_block_conv_3)

        # First inception block
        inception_1_1x1 = Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding='same',
                                 activation='relu', kernel_regularizer=l2())(input_block_pool_4)
        inception_1_3x3_red = Conv2D(filters=96, kernel_size=1, strides=(1, 1), padding='same',
                                     activation='relu', kernel_regularizer=l2())(input_block_pool_4)
        inception_1_3x3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
                                 activation='relu', kernel_regularizer=l2())(inception_1_3x3_red)
        inception_1_5x5_red = Conv2D(filters=16, kernel_size=1, strides=(1, 1), padding='same',
                                     activation='relu', kernel_regularizer=l2())(input_block_pool_4)
        inception_1_5x5 = Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same',
                                 activation='relu', kernel_regularizer=l2())(inception_1_5x5_red)
        inception_1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_block_pool_4)
        inception_1_pool_proj = Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding='same',
                                       activation='relu', kernel_regularizer=l2())(inception_1_pool)
        inception_1_output = Concatenate(axis=3)([inception_1_1x1, inception_1_3x3,
                                                  inception_1_5x5, inception_1_pool_proj])
        max_pool_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_1_output)

        # Second inception block
        inception_2_1x1 = Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding='same',
                                 activation='relu', kernel_regularizer=l2())(max_pool_1)
        inception_2_3x3_red = Conv2D(filters=128, kernel_size=1, strides=(1, 1), padding='same',
                                     activation='relu', kernel_regularizer=l2())(max_pool_1)
        inception_2_3x3 = Conv2D(filters=192, kernel_size=3, strides=(1, 1), padding='same',
                                 activation='relu', kernel_regularizer=l2())(inception_2_3x3_red)
        inception_2_5x5_red = Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding='same',
                                     activation='relu', kernel_regularizer=l2())(max_pool_1)
        inception_2_5x5 = Conv2D(filters=96, kernel_size=5, strides=(1, 1), padding='same',
                                 activation='relu', kernel_regularizer=l2())(inception_2_5x5_red)
        inception_2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(max_pool_1)
        inception_2_pool_proj = Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding='same',
                                       activation='relu', kernel_regularizer=l2())(inception_2_pool)
        inception_2_output = Concatenate(axis=3)([inception_2_1x1, inception_2_3x3,
                                                  inception_2_5x5, inception_2_pool_proj])

        max_pool_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_2_output)

        # Output block
        output_pool = AveragePooling2D(pool_size=(6, 6), strides=(1, 1), padding='same')(max_pool_2)
        output_flat = Flatten()(output_pool)
        output_dropout = Dropout(rate=0.2)(output_flat)
        output_class = Dense(2, kernel_regularizer=l2(), activation='softmax')(output_dropout)

        # Construct the model itself
        model = keras.models.Model(inputs=image_input, outputs=output_class)
        print(model.summary())

        return model


class _AimMask:
    """ Utility class for marking the aim dot (the user has to set this!) """

    def __init__(self, image_shape, aim_size):
        """ Initialize mask itself, returns tensorflow tensor """

        mask = np.ones((image_shape[0], image_shape[1], 3))
        mask[int(0.5 * (image_shape[0] - aim_size)): int(0.5 * (image_shape[0] - aim_size) + aim_size),
             int(0.5 * (image_shape[1] - aim_size)): int(0.5 * (image_shape[1] - aim_size) + aim_size), :] = 0
        self._tf_mask = tf.cast(tf.convert_to_tensor(mask), tf.float32)

    @property
    def mask(self):

        return self._tf_mask


class ImageLogger:
    """ Utility class for logging images in tensorboard """

    def __init__(self, name, logdir, max_images=2):
        """ Initialize it, creates image tab named "name" in tensorboard """

        self.file_writer = tf.summary.create_file_writer(logdir)
        self.name = name
        self._max_images = max_images

    def __call__(self, image, label):
        """ Will be performed for every batch call (but only two images will be saved) """

        with self.file_writer.as_default():
            tf.summary.image(self.name, image,
                             step=0,  # Always overwrite images, do not save
                             max_outputs=self._max_images)

        return image, label
