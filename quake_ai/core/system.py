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

from quake_ai.core.trigger_bot_training import TriggerBotTrainer
from quake_ai.core.trigger_bot_inference import TriggerBot
from quake_ai.utils.screen_capturing import ScreenCapturer
from quake_ai.utils.threading import NonBlockingTask, BlockingTask


class System:

    def __init__(self, system_root_path):

        self._system_root_path = os.path.abspath(system_root_path)

        # It is important that the ScreenCapturer is initialized as first component!
        self._screen_capturer = ScreenCapturer()

        self._trigger_trainer = TriggerBotTrainer(image_path=os.path.join(self._system_root_path, 'triggerbot_images'),
                                                  model_path=os.path.join(self._system_root_path, 'triggerbot_model'),
                                                  image_shape=self._screen_capturer.get_image_shape(),
                                                  screenshot_func=self._screen_capturer.make_screenshot)

        self._trigger_inference = TriggerBot(model_path=os.path.join(self._system_root_path, 'triggerbot_model'),
                                             image_shape=self._screen_capturer.get_image_shape(),
                                             screenshot_func=self._screen_capturer.make_screenshot)

        self._trigger_capture_task = NonBlockingTask(self._trigger_trainer.startup_capture,
                                                     shutdown_task=self._trigger_trainer.shutdown_capture)

        self._trigger_training_task = BlockingTask(self._trigger_trainer.train_epoch,
                                                   init_task=self._trigger_trainer.init_training,
                                                   shutdown_task=self._trigger_trainer.shutdown_training)

        self._trigger_inference_task = BlockingTask(self._trigger_inference.run_inference,
                                                    init_task=self._trigger_inference.init_inference,
                                                    shutdown_task=self._trigger_inference.shutdown_inference)

    @property
    def trigger_capture_task(self):
        return self._trigger_capture_task

    @property
    def trigger_training_task(self):
        return self._trigger_training_task

    @property
    def trigger_inference_task(self):
        return self._trigger_inference_task


