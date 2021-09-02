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

from quake_ai.core.trigger_bot_training import TriggerBotTrainer
from quake_ai.core.trigger_bot_inference import TriggerBot
from quake_ai.core.aim_bot_training import AimbotTrainer
from quake_ai.core.aim_bot_inference import Aimbot
from quake_ai.core.image_annotation import ImageAnnotator
from quake_ai.utils.screen_capturing import ScreenCapturer
from quake_ai.utils.threading import NonBlockingTask, BlockingTask
from quake_ai.core.config import QuakeAiConfig


class System:
    """ Defines tasks that can be used by the QuakeAiGui

        There are two types of tasks:
        - NonBlocking: "Immediately" return
        - Blocking: Might run for a while, example is the training process of the trigger bot
    """

    def __init__(self, config_path):
        """ Startup the system. Also loads tensorflow/CUDA libraries """

        self._config = QuakeAiConfig(config_path)

        #######################################
        #             MEMBERS                 #
        #######################################

        self._screen_capturer = ScreenCapturer(self._config)

        self._trigger_trainer = TriggerBotTrainer(config=self._config,
                                                  screenshot_func=self._screen_capturer.make_trigger_screenshot)

        self._trigger_inference = TriggerBot(config=self._config,
                                             screenshot_func=self._screen_capturer.make_trigger_screenshot)

        self._aimbot_trainer = AimbotTrainer(config=self._config,
                                             screenshot_func=self._screen_capturer.make_aimbot_train_screenshot)

        self._aimbot = Aimbot(config=self._config,
                              screenshot_func=self._screen_capturer.make_aimbot_inference_screenshot)

        self._image_annotator = ImageAnnotator(config=self._config)

        #####################################
        #             TASKS                 #
        #####################################

        self._trigger_capture_task = NonBlockingTask(self._trigger_trainer.startup_capture,
                                                     init_task_list=[self._config.update_from_file,
                                                                     self._screen_capturer.startup],
                                                     shutdown_task_list=[self._trigger_trainer.shutdown_capture,
                                                                         self._screen_capturer.shutdown])

        self._trigger_training_task = BlockingTask(self._trigger_trainer.train_epoch,
                                                   init_task_list=[self._config.update_from_file,
                                                                   self._trigger_trainer.init_training],
                                                   shutdown_task_list=[self._trigger_trainer.shutdown_training])

        self._trigger_inference_task = BlockingTask(self._trigger_inference.run_inference,
                                                    init_task_list=[self._config.update_from_file,
                                                                    self._screen_capturer.startup,
                                                                    self._trigger_inference.init_inference],
                                                    shutdown_task_list=[self._trigger_inference.shutdown_inference,
                                                                        self._screen_capturer.shutdown])

        self._aimbot_capture_task = NonBlockingTask(self._aimbot_trainer.startup_capture,
                                                    init_task_list=[self._config.update_from_file,
                                                                    self._screen_capturer.startup],
                                                    shutdown_task_list=[self._aimbot_trainer.shutdown_capture,
                                                                        self._screen_capturer.shutdown])

        self._aimbot_annotation_task = BlockingTask(self._image_annotator.run_annotation,
                                                    init_task_list=[self._config.update_from_file,
                                                                    self._image_annotator.startup_annotation],
                                                    shutdown_task_list=[self._image_annotator.shutdown_annotation])

        self._aimbot_training_task = BlockingTask(self._aimbot_trainer.train_epoch,
                                                  init_task_list=[self._config.update_from_file,
                                                                  self._aimbot_trainer.init_training],
                                                  shutdown_task_list=[self._aimbot_trainer.shutdown_training])

    @property
    def trigger_capture_task(self):
        return self._trigger_capture_task

    @property
    def aimbot_capture_task(self):
        return self._aimbot_capture_task

    @property
    def aimbot_training_task(self):
        return self._aimbot_training_task

    @property
    def aimbot_annotation_task(self):
        return self._aimbot_annotation_task

    @property
    def trigger_training_task(self):
        return self._trigger_training_task

    @property
    def trigger_inference_task(self):
        return self._trigger_inference_task


