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
from tkinter import filedialog

from quake_ai.core.system import System


class QuakeAiGui:
    """ GUI for Quake AI

        Main internal building blocks are "tasks". Tasks will be run in a separate thread since
        some of them might be running for a while and we don't want to disturb the GUI.
        There may only be one task active at a time
    """

    def __init__(self):
        """ Initialize and start the main GUI system """

        self._system = None
        self._running_thread = None
        self._running_task = None

        self._root = tk.Tk(className='Quake AI')
        self._logo = tk.PhotoImage(file='./resources/logo.png')
        tk.Label(self._root, image=self._logo).pack()

        # Start frame
        self._start_frame = tk.Frame(self._root)
        self._start_frame.pack(side=tk.BOTTOM)

        self._config_path = tk.StringVar()
        tk.Label(self._start_frame, text='Please choose a config to startup the system', bg='white').pack()
        tk.Button(self._start_frame, text="Choose config", command=self._startup_system).pack()

        # Main frame
        self._main_frame = tk.Frame(self._root)
        self._status_text = tk.StringVar(self._main_frame)
        self._status_text.set('Idle')
        tk.Label(self._main_frame, textvariable=self._status_text).pack()
        tk.Button(self._main_frame, text="Capture Images for Triggerbot",
                  command=self._run_trigger_capture_task).pack(side="top")
        tk.Button(self._main_frame, text="Start Training for Triggerbot",
                  command=self._run_trigger_training_task).pack(side="top")
        tk.Button(self._main_frame, text="Start Triggerbot",
                  command=self._run_trigger_inference_task).pack(side="top")
        tk.Button(self._main_frame, text="Stop",
                  command=self._stop_current_task).pack(side="top")

        self._root.mainloop()

    def _startup_system(self):
        """ Start the internal system """

        self._config_path.set(filedialog.asksaveasfilename())
        self._system = System(self._config_path.get())

        self._start_frame.pack_forget()
        self._main_frame.pack()

    def _run_trigger_capture_task(self):
        """ Run image capturing task for trigger bot """

        if self._running_task is None and self._running_thread is None:

            self._running_task = self._system.trigger_capture_task
            self._running_thread = threading.Thread(target=self._running_task.start)
            self._status_text.set('TriggerBot - Capturing')
            self._running_thread.start()

    def _run_trigger_training_task(self):
        """ Run training task for trigger bot """

        if self._running_task is None and self._running_thread is None:

            self._running_task = self._system.trigger_training_task
            self._running_thread = threading.Thread(target=self._running_task.start)
            self._status_text.set('TriggerBot - Training')
            self._running_thread.start()

    def _run_trigger_inference_task(self):
        """ Run the trigger bot """

        if self._running_task is None and self._running_thread is None:

            self._running_task = self._system.trigger_inference_task
            self._running_thread = threading.Thread(target=self._running_task.start)
            self._status_text.set('TriggerBot - Running')
            self._running_thread.start()

    def _stop_current_task(self):
        """ Stop the current task and join its thread """

        if self._running_task is not None:
            self._running_task.stop()

        if self._running_thread is not None:
            self._running_thread.join()

        self._status_text.set('Idle')
        self._running_thread = None
        self._running_task = None







