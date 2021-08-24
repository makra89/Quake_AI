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

import threading
import tkinter as tk

from quake_ai.core.system import System


class QuakeAiGui:

    def __init__(self, system_root_path):

        self._system = System(system_root_path)
        self._running_thread = None
        self._running_task = None

        root = tk.Tk()
        tk.Label(root, text="Quake AI").pack()

        self._status_text = tk.StringVar(root)
        self._status_text.set('Idle')
        tk.Label(root, textvariable=self._status_text).pack()

        tk.Button(root, text="Capture Images for Triggerbot",
                  command=self._run_trigger_capture_task).pack(side="left")

        tk.Button(root, text="Stop",
                  command=self._stop_current_task).pack(side="right")

        root.mainloop()

    def _run_trigger_capture_task(self):

        self._running_task = self._system.trigger_capture_task
        self._running_thread = threading.Thread(target=self._running_task.start)
        self._status_text.set('TriggerBot - Capturing')
        self._running_thread.start()

    def _stop_current_task(self):

        if self._running_task is not None:
            self._running_task.stop()

        if self._running_thread is not None:
            self._running_thread.join()

        self._status_text.set('Idle')
        self._running_thread = None
        self._running_task = None





