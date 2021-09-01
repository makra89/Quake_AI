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

from quake_ai import QuakeAiGui
from quake_ai.core.aim_bot_training import AimbotTrainer
from quake_ai.core.aim_bot_inference import Aimbot
from quake_ai.utils.screen_capturing import ScreenCapturer
from quake_ai.core.config import QuakeAiConfig
#quake_ai = QuakeAiGui()



config = QuakeAiConfig('../test/config.ini')

# Training
model = AimbotTrainer(config)
model.init_training()
model.train_epoch()
model.shutdown_training()
import os

#capturer = ScreenCapturer(config)
#capturer.startup()
#model = Aimbot(config, capturer.make_aimbot_inference_screenshot)
#model.init_inference()
#while True:
#    model.run_inference_screen()

#model = Aimbot(config)
#model.init_inference()
#images = [file for file in os.listdir(os.path.abspath('../test/training/aimbot_images')) if '.png' in file]
#print(images)
#for image in images:
#    model.run_inference(os.path.join('../test/training/aimbot_images', image))



