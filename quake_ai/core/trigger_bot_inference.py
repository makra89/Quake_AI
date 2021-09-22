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
import time

from quake_ai.utils.trigger_model import TriggerModel
from quake_ai.utils.input import Mouse


class TriggerBot:
    """ Main class for trigger bot inference """

    def __init__(self, config, screenshot_func):
        """ Initializes the trigger bot

            :param model_path path to the trained(!) model
            :param config configuration object
            :param screenshot_func screenshot function reference (from ImageCapturer)
        """

        self._config = config
        self._model_path = config.trigger_model_path
        self._screenshot_func = screenshot_func
        self._fov = (config.trigger_fov[0], config.trigger_fov[1])
        self._mouse = Mouse(config)
        # Do not do it here, prevent tensorflow from loading
        self._model = None

    def init_inference(self):
        """ Initialize the model for inference (tries to load the saved model) """

        # Only initialize once!
        if self._model is None:
            self._model = TriggerModel(self._config, self._fov, self._model_path)

        self._model.init_inference()
        self._mouse.set_timeout(0.0)

    def run_inference(self):
        """ Run the trigger bot for one screenshot, this will perform mouse_events! """

        screenshot = np.expand_dims(np.array(self._screenshot_func()), axis=0)

        if self._model.predict_is_on_target(screenshot):
            self._mouse.left_mouse_down()
            time.sleep(0.2)
            self._mouse.left_mouse_up()

    def shutdown_inference(self):
        """ Stop the inference """

        self._model.shutdown_inference()
        self._mouse.set_timeout(0.1)



