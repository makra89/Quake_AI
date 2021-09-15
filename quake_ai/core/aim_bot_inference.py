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
import win32gui
import win32con
import win32api
import pygame
from ctypes import windll
from quake_ai.utils.aimbot_model import AimbotModel
import time


class Aimbot:
    """ Main class for aimbot inference """

    def __init__(self, config, screenshot_func=None, aim_pos_func=None):
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
        self._aim_pos_func = aim_pos_func

        self._fov = (config.aimbot_inference_image_size, config.aimbot_inference_image_size)
        # Do not do it here, prevent tensorflow from loading
        self._model = None
        # Aimbot is switched off by default
        self._active = False
        self._overlay = None

    def init_inference(self):
        """ Initialize the model for inference (tries to load the saved model) """

        # Only initialize once!
        if self._model is None:
            self._model = AimbotModel(self._config, self._fov, self._model_path)

        self._model.init_inference()
        pydirectinput.PAUSE = 0.0
        self._activation_hook = keyboard.on_press_key(self._config.aimbot_activation_key,
                                                      self._activate_aimbot)

        self._overlay = Overlay(self._fov, self._aim_pos_func())

    def run_inference(self):
        """ Run the trigger bot for one screenshot """

        if self._active:
            screenshot = np.array(self._screenshot_func())
            boxes, scores, classes, nums = self._model.predict(screenshot)

            current_min_dist = 1000
            current_shortest_rel = (0, 0)
            current_top_left = (0, 0)
            current_bottom_right = (0, 0)
            found = False

            # Naive approach: Just take the box which is closest to the aim
            # Results in jumping around a lot
            for element_id in np.arange(nums):
                box = boxes[0, element_id, :]

                left = self._fov[1] * box[0].numpy()
                top = self._fov[0] * box[1].numpy()
                width = self._fov[1] * (box[2] - box[0]).numpy()
                height = self._fov[0] * (box[3] - box[1]).numpy()

                mean_x = left + 0.5 * width
                mean_y = top + 0.5 * height

                rel_x, rel_y = (mean_x - self._fov[1]/2., mean_y - self._fov[0]/2.)
                dist = np.sqrt(rel_x**2 + rel_y**2)

                # Pick closest one
                if dist < current_min_dist:
                    current_min_dist = dist
                    current_shortest_rel = (int(rel_x), int(rel_y))
                    current_left_top = (left, top)
                    current_width_height = (width, height)
                    found = True

            if found:
                pydirectinput.move(current_shortest_rel[0], current_shortest_rel[1])
                self._overlay.activate_rectangle(current_left_top, current_width_height)
            else:
                self._overlay.deactivate_rectangle()

        else:
            self._overlay.deactivate_rectangle()

        self._overlay.update()

    def shutdown_inference(self):
        """ Stop the inference """

        self._model.shutdown_inference()
        pydirectinput.PAUSE = 0.1
        if self._activation_hook is not None:
            keyboard.unhook(self._activation_hook)

        self._overlay = None

    def _activate_aimbot(self, _):

        self._active = not self._active


class Overlay:
    """ Simple overlay for displaying the (tracked) yolo bounding box"""
    def __init__(self, fov, aim_pos):

        self._rect_active = False

        pygame.init()

        # Create transparent, clickthrough window
        self._window = pygame.display.set_mode((fov[1], fov[0]), pygame.NOFRAME)
        self._trans_color = (255, 255, 255)  # Transparency color
        self._window.fill(self._trans_color)
        self._hwnd = pygame.display.get_wm_info()["window"]
        _set_transparent_clickthrough(self._hwnd, self._trans_color, int(aim_pos[0] - fov[1]/2.),
                                      int(aim_pos[1] - fov[0]/2.))

        self._rect = pygame.Rect(50, 50, 50, 50)
        self._rect_color = self._trans_color

        self.update()

    def activate_rectangle(self, left_top, width_heigth):

        self._rect = pygame.Rect(left_top, width_heigth)
        self._rect_color = (255, 0, 0)

    def deactivate_rectangle(self):

        self._rect_color = self._trans_color

    def update(self):

        self._window.fill(self._trans_color)
        pygame.draw.rect(self._window, self._rect_color, self._rect, 3)
        pygame.display.flip()
        pygame.event.pump()
        time.sleep(0.01)


def _set_transparent_clickthrough(hwnd, color, x_pos, y_pos):
    """ Define as transparent, clickthrough window """

    styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    styles |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*color), 0, win32con.LWA_COLORKEY)
    windll.user32.SetWindowPos(hwnd, -1, x_pos, y_pos, 0, 0, 0x0001)



