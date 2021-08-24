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

_CAPTURE_KEY_POS = 'e'
_CAPTURE_KEY_NEG = 'r'


class TriggerBotTrainer:

    def __init__(self, image_path, screenshot_func):

        self._image_path = image_path
        self._screenshot_func = screenshot_func

        self._hook_pos = None
        self._hook_neg = None
        
        # Setup image paths
        self._image_path_pos = os.path.join(os.path.abspath(image_path), 'pos')
        self._image_path_neg = os.path.join(os.path.abspath(image_path), 'neg')
        print(self._image_path_pos)

        if not os.path.exists(self._image_path_pos):
            os.makedirs(self._image_path_pos)
            self._curr_image_pos_inc = 0
        else:
            self._curr_image_pos_inc = _get_latest_inc(self._image_path_pos) + 1

        if not os.path.exists(self._image_path_neg):
            os.makedirs(self._image_path_neg)
            self._curr_image_neg_inc = 0
        else:
            self._curr_image_neg_inc = _get_latest_inc(self._image_path_neg) + 1

    def startup(self):

        self._hook_pos = keyboard.on_press_key(_CAPTURE_KEY_POS, self._save_pos_screenshot_callback)
        self._hook_neg = keyboard.on_press_key(_CAPTURE_KEY_NEG, self._save_neg_screenshot_callback)

    def shutdown(self):

        if self._hook_pos is not None:
            keyboard.unhook(self._hook_pos)
        if self._hook_neg is not None:
            keyboard.unhook(self._hook_neg)

    def _save_screenshot(self, path, inc):
        self._screenshot_func().save(os.path.join(path, str(inc) + '.png'))

    def _save_pos_screenshot_callback(self, _):
        self._save_screenshot(self._image_path_pos, self._curr_image_pos_inc)
        self._curr_image_pos_inc += 1

    def _save_neg_screenshot_callback(self, _):
        self._save_screenshot(self._image_path_neg, self._curr_image_neg_inc)
        self._curr_image_neg_inc += 1


def _get_latest_inc(path):
    images = [os.path.join(path, image) for image in os.listdir(path)]
    if not images:
        return 0
    else:
        return int(re.search('(?P<inc>\d+).png$', max(images, key=os.path.getctime)).group('inc'))





