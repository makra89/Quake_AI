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

import pyautogui as pgui
import win32gui


class _WindowRect:

    def __init__(self, rect_tuple_ltrb):
        self.left = rect_tuple_ltrb[0]
        self.top = rect_tuple_ltrb[1]
        self.right = rect_tuple_ltrb[2]
        self.bottom = rect_tuple_ltrb[3]


class ScreenCapturer:

    def __init__(self):
        self._process_handle = None
        self._process_window_rect = None

    def startup(self):

        self._process_handle = win32gui.FindWindow(None, "Quake Live")
        if self._process_handle == 0:
            raise RuntimeError('Cannot find Quake Live window on primary screen')
        else:
            win32gui.SetForegroundWindow(self._process_handle)
            self._process_window_rect = _WindowRect((win32gui.GetWindowRect(self._process_handle)))

    def shutdown(self):

        self._process_handle = None
        self._process_window_rect = None

    def make_screenshot(self):
        return pgui.screenshot(region=(self._process_window_rect.left,self._process_window_rect.top,
                                       self._process_window_rect.right-self._process_window_rect.left,
                                       self._process_window_rect.bottom-self._process_window_rect.top))
