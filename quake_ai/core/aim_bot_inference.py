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

import numpy as np
import keyboard
import pydirectinput

from quake_ai.utils.aimbot_model import AimbotModel


class Aimbot:
    """ Main class for aimbot inference """

    def __init__(self, config, screenshot_func=None, coord_trafo_func=None):
        """ Initializes the aimbot

            :param model_path path to the trained(!) model
            :param config configuration object
            :param screenshot_func screenshot function reference (from ImageCapturer)
            :param coord_trafo_func gets relative pos. to aim position (from ImageCapturer)
        """

        self._config = config
        self._activation_hook = None
        self._model_path = config.aimbot_model_path
        self._screenshot_func = screenshot_func
        self._coord_trafo_func = coord_trafo_func
        self._fov = (config.aimbot_inference_image_size, config.aimbot_inference_image_size)
        # Do not do it here, prevent tensorflow from loading
        self._model = None
        # Aimbot is switched off by default
        self._active = False

    def init_inference(self):
        """ Initialize the model for inference (tries to load the saved model) """

        # Only initialize once!
        if self._model is None:
            self._model = AimbotModel(self._config, self._fov, self._model_path)

        self._model.init_inference()
        pydirectinput.PAUSE = 0.0
        self._activation_hook = keyboard.on_press_key(self._config.aimbot_activation_key,
                                                      self._activate_aimbot)

    def run_inference(self):
        """ Run the trigger bot for one screenshot, this will perform mouse_events! """

        if self._active:

            screenshot = np.array(self._screenshot_func())
            boxes, scores, classes, nums = self._model.predict(screenshot)

            current_min_dist = 1000
            current_shortest_rel = (0, 0)
            for element_id in np.arange(nums):
                box = boxes[0, element_id, :]

                mean_x = self._fov[1] * 0.5 * (box[0] + box[2])
                mean_y = self._fov[0] * 0.5 * (box[1] + box[3])

                rel_x, rel_y = self._coord_trafo_func(mean_x, mean_y)
                dist = np.sqrt(rel_x**2 + rel_y**2)
                if dist < current_min_dist:
                    current_shortest_rel = (rel_x, rel_y)

            pydirectinput.moveRel(-current_shortest_rel[0], -current_shortest_rel[1], 0.05, relative=True)

    def shutdown_inference(self):
        """ Stop the inference """

        self._model.shutdown_inference()
        pydirectinput.PAUSE = 0.1
        if self._activation_hook is not None:
            keyboard.unhook(self._activation_hook)

    def _activate_aimbot(self, _):

        self._active = not self._active



