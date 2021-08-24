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
from quake_ai.utils.screen_capturing import ScreenCapturer
from quake_ai.utils.threading import NonBlockingTask


class System:

    def __init__(self, system_root_path):

        self.system_root_path = os.path.abspath(system_root_path)
        self.screen_capturer = ScreenCapturer()
        self.trigger_trainer = TriggerBotTrainer(image_path=os.path.join(self.system_root_path, 'triggerbot_images'),
                                                 screenshot_func=self.screen_capturer.make_screenshot)

        self._trigger_capture_task = NonBlockingTask(self._start_capture_images_trigger,
                                                     self._stop_capture_images_trigger)

    @property
    def trigger_capture_task(self):
        return self._trigger_capture_task

    def _start_capture_images_trigger(self):

        self.screen_capturer.startup()
        self.trigger_trainer.startup()

    def _stop_capture_images_trigger(self):

        self.screen_capturer.shutdown()
        self.trigger_trainer.shutdown()


