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

import win32gui
import win32ui
import win32con
import win32api
from PIL import Image


class _WindowRect:
    """ Utility class for defining the edges of a rectangular window """

    def __init__(self, rect_tuple_ltrb):
        """ input argument defined by win32gui.GetWindowRect(process_handle) """

        self.left = rect_tuple_ltrb[0]
        self.top = rect_tuple_ltrb[1]
        self.right = rect_tuple_ltrb[2]
        self.bottom = rect_tuple_ltrb[3]

    def calc_center_crop_rect(self, height, width):
        """ Calculate the _WindowRect object that corresponds to cropping from the center of this rect """

        old_height = self.bottom - self.top
        old_width = self.right - self.left

        new_top = int(0.5 * (old_height - height)) + self.top
        new_left = int(0.5 * (old_width - width)) + self.left

        return _WindowRect((new_left, new_top, new_left + width, new_top + height))


class ScreenCapturer:
    """ Attaches to Quake Live process and provides function references for making screenshots """

    def __init__(self, config):
        """ Initialize, nothing done here since we cannot be sure that the Quake Live process exists """

        self._config = config
        self._process_handle = None
        self._window_rect = None  # Window of main process (without bars)
        self._trigger_fov_rect = None  # Window for trigger bot fov
        self._aimbot_train_fov_rect = None  # Window for aimbot trainining fov
        self._aimbot_inf_fov_rect = None  # Window for aimbot trainining fov
        self._aim_shift = None

    def startup(self):
        """ Attach ScreenCapturer to the Quake Live handle (if existing)
            and provide screenshot function
        """

        self._process_handle = win32gui.FindWindow(None, "Quake Live")
        if self._process_handle == 0:
            raise RuntimeError('Cannot find Quake Live window on primary screen')
        else:
            win32gui.SetForegroundWindow(self._process_handle)

            self._window_rect = _WindowRect((win32gui.GetClientRect(self._process_handle)))
            # Calculate window for trigger bot fov
            trigger_fov_height = self._config.trigger_fov[0]
            trigger_fov_width = self._config.trigger_fov[1]
            self._trigger_fov_rect = self._window_rect.calc_center_crop_rect(trigger_fov_height,
                                                                             trigger_fov_width)

            # Calculate window for aimbot training fov
            aimbot_train_height = self._config.aimbot_train_image_size
            aimbot_train_width = self._config.aimbot_train_image_size
            self._aimbot_train_fov_rect = self._window_rect.calc_center_crop_rect(aimbot_train_height,
                                                                                  aimbot_train_width)

            # Calculate window for aimbot inference fov
            aimbot_inf_height = self._config.aimbot_inference_image_size
            aimbot_inf_width = self._config.aimbot_inference_image_size
            self._aimbot_inf_fov_rect = self._window_rect.calc_center_crop_rect(aimbot_inf_height,
                                                                                aimbot_inf_width)

    def shutdown(self):
        """ Is there anything to do with the process handle? """

        self._process_handle = None
        self._window_rect = None

    def make_trigger_screenshot(self):
        """ Make and return screenshot of the trigger bot fov """

        return self._screenshot(self._trigger_fov_rect.bottom - self._trigger_fov_rect.top,
                                self._trigger_fov_rect.right - self._trigger_fov_rect.left,
                                self._trigger_fov_rect.left,
                                self._trigger_fov_rect.top)

    def make_aimbot_train_screenshot(self):
        """ Make and return screenshot of the aimbot training fov """

        return self._screenshot(self._aimbot_train_fov_rect.bottom - self._aimbot_train_fov_rect.top,
                                self._aimbot_train_fov_rect.right - self._aimbot_train_fov_rect.left,
                                self._aimbot_train_fov_rect.left,
                                self._aimbot_train_fov_rect.top)

    def make_aimbot_inference_screenshot(self):
        """ Make and return screenshot of the aimbot inference fov """

        return self._screenshot(self._aimbot_inf_fov_rect.bottom - self._aimbot_inf_fov_rect.top,
                                self._aimbot_inf_fov_rect.right - self._aimbot_inf_fov_rect.left,
                                self._aimbot_inf_fov_rect.left,
                                self._aimbot_inf_fov_rect.top)

    def get_aim_pos(self):
        """ Get the position of the aim in screen coordinates"""

        left_top_client = (self._window_rect.left, self._window_rect.top)
        right_bottom_client = (self._window_rect.right, self._window_rect.bottom)
        left_top_screen = win32gui.ClientToScreen(self._process_handle, left_top_client)
        right_bottom_screen = win32gui.ClientToScreen(self._process_handle, right_bottom_client)

        x_pos = 0.5 * (right_bottom_screen[0] + left_top_screen[0])
        y_pos = 0.5 * (right_bottom_screen[1] + left_top_screen[1])

        return x_pos, y_pos

    # Reference: https://stackoverflow.com/questions/66384468/how-to-record-my-computer-screen-with-high-fps
    def _screenshot(self, height, width, left, top):
        """ Do a screenshot, window may be inactive """

        top = top + win32api.GetSystemMetrics(win32con.SM_CYCAPTION) + self._config.screen_fine_tune_y
        left = left + win32api.GetSystemMetrics(win32con.SM_CYBORDER) + self._config.screen_fine_tune_x
        hwindc = win32gui.GetWindowDC(self._process_handle)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()

        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

        signed_ints_array = bmp.GetBitmapBits(True)
        img_out = Image.frombuffer(
            'RGB',
            (width, height),
            signed_ints_array, 'raw', 'BGRX', 0, 1)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(self._process_handle, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return img_out
