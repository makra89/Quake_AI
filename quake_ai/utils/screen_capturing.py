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
import mss
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
        self._mss_handle = None

    def startup(self):
        """ Attach ScreenCapturer to the Quake Live handle (if existing)
            and provide screenshot function
        """

        self._process_handle = win32gui.FindWindow(None, "Quake Live")
        if self._process_handle == 0:
            raise RuntimeError('Cannot find Quake Live window on primary screen')
        else:
            win32gui.SetForegroundWindow(self._process_handle)
            # Get Client rect --> this is in client coordinates
            client_rect = _WindowRect((win32gui.GetClientRect(self._process_handle)))

            # Convert to screen coordinates
            left_top_client = (client_rect.left, client_rect.top)
            right_bottom_client = (client_rect.right, client_rect.bottom)
            left_top_screen = win32gui.ClientToScreen(self._process_handle, left_top_client)
            right_bottom_screen = win32gui.ClientToScreen(self._process_handle, right_bottom_client)

            self._window_rect = _WindowRect((left_top_screen[0], left_top_screen[1],
                                             right_bottom_screen[0], right_bottom_screen[1]))

            # Calculate window for trigger bot fov
            trigger_fov_height = self._config.trigger_fov[0]
            trigger_fov_width = self._config.trigger_fov[1]
            self._trigger_fov_rect = self._window_rect.calc_center_crop_rect(trigger_fov_height,
                                                                             trigger_fov_width)

            self._mss_handle = mss.mss()

    def shutdown(self):
        """ Is there anything to do with the process handle? """

        self._process_handle = None
        self._window_rect = None
        self._mss_handle = None

    def make_window_screenshot(self):
        """ Make and return screenshot of the complete process window """

        monitor = {"top": self._window_rect.top, "left": self._window_rect.left,
                   "width": self._window_rect.right - self._window_rect.left,
                   "height": self._window_rect.bottom - self._window_rect.top}

        return pil_frombytes(self._mss_handle.grab(monitor))

    def make_trigger_screenshot(self):
        """ Make and return screenshot of the trigger bot fov """

        monitor = {"top": self._trigger_fov_rect.top, "left": self._trigger_fov_rect.left,
                   "width": self._trigger_fov_rect.right - self._trigger_fov_rect.left,
                   "height": self._trigger_fov_rect.bottom - self._trigger_fov_rect.top}

        return pil_frombytes(self._mss_handle.grab(monitor))


# From https://python-mss.readthedocs.io/examples.html#bgra-to-rgb
# Converts BGRA to RGB
def pil_frombytes(image):
    """ Efficient Pillow version. """
    
    return Image.frombytes('RGB', image.size, image.bgra, 'raw', 'BGRX')
