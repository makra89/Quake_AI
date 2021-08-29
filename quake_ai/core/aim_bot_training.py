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
import keyboard
import numpy as np
import re


class AimbotTrainer:
    """ Main class for training the aim bot """

    def __init__(self, config, screenshot_func):
        """ Initializes the data structure needed for training the model

            :param config configuration object
            :param screenshot_func screenshot function reference (from ImageCapturer)
        """

        self._config = config
        self._fov = (config.trigger_fov[0], config.trigger_fov[1])
        self._screenshot_func = screenshot_func

        self._hook_screenshot = None

        # Setup image paths
        self._image_path = os.path.join(self._config.training_env_path, 'aimbot_images')
        self._curr_image_inc = None

    def startup_capture(self):
        """ Startup capturing, hook for keyboard key will be placed """

        # Check if path for images already exists
        if not os.path.exists(self._image_path):
            os.makedirs(self._image_path)
            self._curr_image_inc = 0
        else:
            self._curr_image_inc = _get_latest_inc(self._image_path) + 1

        self._hook_screenshot = keyboard.on_press_key(self._config.aimbot_train_capture_key,
                                                      self._save_screenshot_callback)

    def shutdown_capture(self):
        """ Release hook, stop capturing """

        if self._hook_screenshot is not None:
            keyboard.unhook(self._hook_screenshot)

    def _save_screenshot_callback(self, _):
        """ Save a screenshot done using the screenshot handle """

        self._curr_image_inc += 1
        image = self._screenshot_func()
        print("Captured image of shape", np.shape(image))
        print("Current number of images:", self._curr_image_inc)

        image.save(os.path.join(self._image_path, str(self._curr_image_inc) + '.png'))


def _get_latest_inc(path):
    """ Search through images in path and get 'oldest' one.
        If there are missing image numbers in between, this will be wrong
    """

    images = [os.path.join(path, image) for image in os.listdir(path)]
    if not images:
        return 0
    else:
        return int(re.search('(?P<inc>\d+).png$', max(images, key=os.path.getctime)).group('inc'))





