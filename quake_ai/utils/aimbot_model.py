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

from yolov3_tf2.models import YoloV3Tiny, YoloLoss, yolo_tiny_anchors, yolo_tiny_anchor_masks
from yolov3_tf2.dataset import load_tfrecord_dataset, transform_targets

import tensorflow as tf
import tensorflow.keras as keras
import os
import datetime
from yolov3_tf2.utils import draw_outputs
import cv2
import numpy as np
from PIL import Image
from quake_ai.utils.model_utils import ImageLogger


class AimbotModel:
    """ aimbot model used for inference """

    def __init__(self, config, image_shape, model_path):
        """ Initialize the model

        :param config configuration object
        :param image_shape (height,width) of used fov
        :param model_path path to the model file
        """

        self._config = config
        self._image_shape = image_shape
        self._model_path = model_path
        self._class_file_path = config.aimbot_class_file
        self._class_names = [c.strip() for c in open(self._class_file_path).readlines()]
        self._training = False

        self._model = None

    def init_inference(self):
        """ Initialize aimbot inference"""

        self._model = YoloV3Tiny(size=self._config.aimbot_train_image_size, channels=3, classes=1, training=False)
        self._model.load_weights(self._model_path)

    def predict(self, image):
        """ Returns true if aim is on-target """

        image_proc = np.expand_dims(image, axis=0)
        image_proc = self._preprocess_image(image_proc)
        image_proc2 = self._preprocess_image(image)
        boxes, scores, classes, nums = self._model(image_proc)
        #print(image_proc)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_proc2 = cv2.cvtColor(image_proc2.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), self._class_names)
        #print(img)
        cv2.imshow("Bla", img)
        #cv2.imshow("Bla2", image_proc2)
        cv2.waitKey(0)
        print(scores)
        #print(classes)
        print(nums)


    def shutdown_inference(self):

        self._model = None

    @tf.function
    def _preprocess_image(self, image):

        # Rectangular images are not supported at the moment --> change this
        if not self._training and (self._config.aimbot_inference_image_size != self._config.aimbot_train_image_size):
            image = tf.image.resize(image, (self._config.aimbot_train_image_size, self._config.aimbot_train_image_size))
        image = tf.cast(image, tf.float32)
        image = (image - tf.reduce_mean(image)) / tf.math.reduce_std(image)

        return image


class TrainableAimbotModel(AimbotModel):
    """ Aimbot training model, builds on top of the inference model """

    def __init__(self, config, image_shape, model_path):
        """ Initialize the model

        :param config configuration object
        :param image_shape (height,width) of used fov
        :param model_path path to the model file
        """

        super(TrainableAimbotModel, self).__init__(config, image_shape, model_path)

        self._training = True
        self._model = YoloV3Tiny(size=self._config.aimbot_train_image_size, channels=3, classes=1, training=True)
        #self._model.load_weights(self._config.aimbot_train_darknet_weights_path)
        print("Weigths loaded")
        print("Weigths loaded")
        print("Weigths loaded")
        print("Weigths loaded")
        print("Weigths loaded")
        print("Weigths loaded")
        print("Weigths loaded")
        print("Weigths loaded")
        print("Weigths loaded")
        self._optimizer = keras.optimizers.Adam(lr=0.001)
        self._loss = [YoloLoss(yolo_tiny_anchors[mask], classes=1) for mask in yolo_tiny_anchor_masks]
        self._model.compile(optimizer=self._optimizer, loss=self._loss)
        self._current_epoch = 0  # relative to initialize time
        self._training_fov = (config.aimbot_train_image_size, config.aimbot_train_image_size)
        self._output_weights = os.path.join(os.path.dirname(model_path), 'aimbot_weights.hdf5')

        #####################################
        #        Training Callbacks         #
        #####################################

        self._training_callbacks = []
        self._training_callbacks.append(keras.callbacks.ModelCheckpoint(self._model_path, monitor='val_loss',
                                                                        verbose=1, save_best_only=True, mode='min'))
        self._training_callbacks.append(keras.callbacks.ModelCheckpoint(self._output_weights, monitor='val_loss',
                                                                        verbose=1, save_weights_only=True,
                                                                        save_best_only=True, mode='min'))
        self._training_callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                                          patience=10, min_lr=1e-7, verbose=1))

        logdir = os.path.join(os.path.dirname(self._model_path),
                              "logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._training_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))

        self._image_logger = ImageLogger(name='Images', logdir=logdir, max_images=2)

    def fit_num_epochs(self, data_train, data_test):
        """ Fit for a number of epochs (see config). Model will be reloaded if existing to keep on training.
            TODO: Resuming training is not working correctly after training is stopped, GUI closed, restarted
        """

        try:
            self._model = keras.models.load_model(self._model_path, custom_objects={'yolo_loss': self._loss})
        except (ImportError, IOError):
            self._current_epoch = 0

        self._model.fit(x=data_train, validation_data=data_test, initial_epoch=self._current_epoch,
                        epochs=self._config.aimbot_train_epochs+self._current_epoch,
                        callbacks=self._training_callbacks)
        self._current_epoch += self._config.aimbot_train_epochs

    def create_tfrecords(self, output_path, image_labels):
        """ Create tfrecords for training/testing data

            :param output_path path to .tfrecord file
            :param image_labels image paths and boxes/labels
        """
        class_map = {idx: name for idx, name in enumerate(
            open(self._class_file_path).read().splitlines())}

        writer = tf.io.TFRecordWriter(output_path)

        for element in image_labels:
            tf_example = build_example(element[0], element[1], class_map, self._training_fov)
            writer.write(tf_example.SerializeToString())

        writer.close()

    def create_dataset(self, tfrecord_path, augment=False, shuffle=False):
        """ Create the a tensorflow data set which can be used for training/testing

            :param image_paths_labels array of (image_path, label) tuples
            :param augment flag whether to perform image augmentation
            :param shuffle flag whether to perform shuffling
            :returns tf.data.Dataset
        """

        dataset = load_tfrecord_dataset(tfrecord_path, self._class_file_path)
        # Augmentation
        # TODO: Properly implement augmentation!
        #if augment:
           # dataset = dataset.map(self._augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Set the number of datapoints you want to load and shuffle
        if shuffle:
            dataset = dataset.shuffle(self._config.aimbot_train_shuffle_size)

        # Set batch size
        dataset = dataset.batch(self._config.aimbot_train_batch_size)

        dataset = dataset.map(lambda x, y: (
            self._preprocess_image(x),
            transform_targets(y, yolo_tiny_anchors, yolo_tiny_anchor_masks, self._config.aimbot_train_image_size)))

        dataset = dataset.map(self._image_logger, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    @tf.function
    def _augment(self, image, label):
        """ Perform image augmentation, I think we could do more here """

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)

        return image, label


def build_example(image_path, label_list, class_map, image_size):

    # Reading binary and writing to tfrecord leads to corrupt images....Why?
    img_raw = np.array(Image.open(image_path).convert("RGB"))

    width = image_size[1]
    height = image_size[0]

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    for obj in label_list:

        xmin.append(float(obj[0]) / width)
        ymin.append(float(obj[1]) / height)
        xmax.append(float(obj[0] + obj[2]) / width)
        ymax.append(float(obj[1] + obj[3]) / height)
        classes_text.append(class_map[0].encode('utf8'))
        classes.append(0)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(img_raw).numpy()])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes))
    }))

    return example

