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
from Bezier import Bezier
import keyboard
import pydirectinput
import time


TIME_UNTIL_TRAJ_OUTDATED = 0.05  # 50 ms should be okay


class Trajectory:

    def __init__(self, start_point, end_point):

        self._start = start_point
        self._end = end_point
        self._init_time = time.time()


class Mouse:

    def __init__(self, config):

        self._config = config
        self._current_trajectory = None
        pydirectinput.PAUSE = 0.1  # Timeout for input

    def set_timeout(self, timeout):

        pydirectinput.PAUSE = timeout

    def move_rel(self, x, y):

        current_time = time.time()

    def left_mouse_down(self):

        pydirectinput.mouseDown()

    def left_mouse_up(self):

        pydirectinput.mouseUp()


class Keyboard:

    def __init__(self, config):

        self._config = config

    def register_hook(self, key, callback):

        pass

    def remove_hook(self, hook):

        pass