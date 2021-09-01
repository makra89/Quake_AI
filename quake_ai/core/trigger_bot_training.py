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

import os
import re
import keyboard
import numpy as np
from sklearn.model_selection import train_test_split

from quake_ai.utils.trigger_model import TrainableTriggerModel


class TriggerBotTrainer:
    """ Main class for training the trigger bot """

    def __init__(self, config, screenshot_func):
        """ Initializes the data structure needed for training the model

            :param config configuration object
            :param screenshot_func screenshot function reference (from ImageCapturer)
        """

        self._config = config
        self._fov = (config.trigger_fov[0], config.trigger_fov[1])
        self._screenshot_func = screenshot_func
        # Do not initialize the model here, prevent tensorflow from loading
        self._model_root_path = os.path.join(self._config.training_env_path, 'triggerbot_model')
        self._model = None

        self._hook_pos = None
        self._hook_neg = None

        # Setup image paths
        self._image_path = os.path.join(self._config.training_env_path, 'triggerbot_images')
        self._image_path_pos = os.path.join(os.path.abspath(self._image_path), 'pos')
        self._image_path_neg = os.path.join(os.path.abspath(self._image_path), 'neg')
        # Will be initialized once during training startup
        self._curr_image_neg_inc = None
        self._curr_image_pos_inc = None

        self._train_data_set = None
        self._test_data_set = None

    def startup_capture(self):
        """ Startup capturing, hooks for keyboard keys will be placed """

        # Check if paths for 'positive' images already exist
        if not os.path.exists(self._image_path_pos):
            os.makedirs(self._image_path_pos)
            self._curr_image_pos_inc = 0
        else:
            self._curr_image_pos_inc = _get_latest_inc(self._image_path_pos) + 1

        # Check if paths for 'negative' images already exist
        if not os.path.exists(self._image_path_neg):
            os.makedirs(self._image_path_neg)
            self._curr_image_neg_inc = 0
        else:
            self._curr_image_neg_inc = _get_latest_inc(self._image_path_neg) + 1

        self._hook_pos = keyboard.on_press_key(self._config.trigger_train_capture_pos,
                                               self._save_pos_screenshot_callback)
        self._hook_neg = keyboard.on_press_key(self._config.trigger_train_capture_neg,
                                               self._save_neg_screenshot_callback)

    def shutdown_capture(self):
        """ Release all hooks, stop capturing """

        if self._hook_pos is not None:
            keyboard.unhook(self._hook_pos)
        if self._hook_neg is not None:
            keyboard.unhook(self._hook_neg)

    def init_training(self):
        """ Initialize training, collects image paths and creates data sets"""

        if not os.path.exists(self._model_root_path):
            os.makedirs(self._model_root_path)

        # Only initialize once!
        if self._model is None:
            self._model = TrainableTriggerModel(self._config, self._fov,
                                                os.path.join(self._model_root_path, 'trigger_model.tf'))

        pos_path_labels = [(os.path.join(self._image_path_pos, image), 1) for image in os.listdir(self._image_path_pos)]
        neg_path_labels = [(os.path.join(self._image_path_neg, image), 0) for image in os.listdir(self._image_path_neg)]
        all_paths_labels = pos_path_labels + neg_path_labels
        # Get test and train sets
        paths_labels_train, paths_labels_test = train_test_split(all_paths_labels, shuffle=True, test_size=0.20)

        self._train_data_set = self._model.create_dataset(paths_labels_train, augment=True, shuffle=True)
        self._test_data_set = self._model.create_dataset(paths_labels_test)

    def train_epoch(self):
        """ Train for a number of epochs (see config) """

        if self._train_data_set is not None and self._train_data_set is not None:
            self._model.fit_num_epochs(self._train_data_set, self._test_data_set)
        else:
            raise RuntimeError("[Triggerbot]: No training or test set available")

    def shutdown_training(self):
        """ Shutdown training process, nothing to do here """

        self._train_data_set = None
        self._test_data_set = None

    def _save_screenshot(self, path, inc):
        """ Save a screenshot done using the screenshot handle """

        image = self._screenshot_func()
        print("Captured image of shape", np.shape(image))
        print("Current positive labels:", self._curr_image_pos_inc)
        print("Current negative labels:", self._curr_image_neg_inc)

        image.save(os.path.join(path, str(inc) + '.png'))

    def _save_pos_screenshot_callback(self, _):
        """ Callback for 'positive images' hook """

        self._save_screenshot(self._image_path_pos, self._curr_image_pos_inc)
        self._curr_image_pos_inc += 1

    def _save_neg_screenshot_callback(self, _):
        """ Callback for 'negative images' hook """

        self._save_screenshot(self._image_path_neg, self._curr_image_neg_inc)
        self._curr_image_neg_inc += 1


def _get_latest_inc(path):
    """ Search through images in path and get 'oldest' one.
        If there are missing image numbers in between, this will be wrong
    """

    images = [os.path.join(path, image) for image in os.listdir(path)]
    if not images:
        return 0
    else:
        return int(re.search('(?P<inc>\d+).png$', max(images, key=os.path.getctime)).group('inc'))





