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
from tkinter import ttk
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
        self._root.configure(background='white')
        self._logo = tk.PhotoImage(file='./resources/logo.png')
        logo_frame = tk.Frame(self._root, bg='white')
        tk.Label(logo_frame, image=self._logo, bd=0).pack()
        logo_frame.pack()

        # Start frame
        self._start_frame = tk.Frame(self._root, bg='white')

        self._config_path = tk.StringVar()

        tk.Label(self._start_frame, text='Quake AI needs config to startup the system. Either choose an\n '
                                         'existing configuration file or provide a file path and a default one\n '
                                         'will be generated.',
                 bg='white', font=('Helvetica', 10)).pack(fill='x')

        tk.Button(self._start_frame, text="Choose config", font=('Helvetica', 10),
                  command=self._startup_system).pack(pady=10)
        self._start_frame.pack(side=tk.BOTTOM)

        # Main frame
        self._main_frame = tk.Frame(self._root, bg='white')

        self._status_text = tk.StringVar(self._main_frame)
        self._status_text.set('Idle')
        tk.Label(self._main_frame, textvariable=self._status_text, font=('Helvetica', 16, 'bold'), bg='white',
                 relief=tk.SUNKEN).pack(pady=10)

        trigger_training = tk.LabelFrame(self._main_frame, text="Triggerbot Training", font=('Helvetica', 10), bg='white')
        tk.Button(trigger_training, text="Capture Images for Triggerbot", font=('Helvetica', 10),
                  command=self._run_trigger_capture_task).pack(pady=10)
        tk.Button(trigger_training, text="Start Training for Triggerbot", font=('Helvetica', 10),
                  command=self._run_trigger_training_task).pack(pady=10)
        trigger_training.pack(ipadx=110)

        triggerbot = tk.LabelFrame(self._main_frame, text="Triggerbot", font=('Helvetica', 10), bg='white')
        tk.Button(triggerbot, text="Start Triggerbot", font=('Helvetica', 10),
                  command=self._run_trigger_inference_task).pack(pady=10)
        triggerbot.pack(ipadx=150)

        aimbot_training = tk.LabelFrame(self._main_frame, text="Aimbot Training", font=('Helvetica', 10), bg='white')
        tk.Button(aimbot_training, text="Capture Images for Aimbot", font=('Helvetica', 10),
                  command=self._run_aimbot_capture_task).pack(pady=10)
        tk.Button(aimbot_training, text="Annotate Images for Aimbot", font=('Helvetica', 10),
                  command=self._run_aimbot_annotation_task).pack(pady=10)
        tk.Button(aimbot_training, text="Train Aimbot", font=('Helvetica', 10),
                  command=self._run_aimbot_training_task).pack(pady=10)
        tk.Button(aimbot_training, text="Start Aimbot", font=('Helvetica', 10),
                  command=self._run_aimbot_inference_task).pack(pady=10)
        aimbot_training.pack(ipadx=118)

        tk.Button(self._main_frame, text="Stop Current Task", font=('Helvetica', 10),
                  command=self._stop_current_task).pack(pady=10)

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

    def _run_aimbot_capture_task(self):
        """ Run image capturing task for aimbot """

        if self._running_task is None and self._running_thread is None:

            self._running_task = self._system.aimbot_capture_task
            self._running_thread = threading.Thread(target=self._running_task.start)
            self._status_text.set('Aimbot - Capturing')
            self._running_thread.start()

    def _run_aimbot_annotation_task(self):
        """ Run image annotation for aimbot """

        if self._running_task is None and self._running_thread is None:

            self._running_task = self._system.aimbot_annotation_task
            self._running_thread = threading.Thread(target=self._running_task.start)
            self._status_text.set('Aimbot - Annotating')
            self._running_thread.start()

    def _run_aimbot_training_task(self):
        """ Run aimbot training """

        if self._running_task is None and self._running_thread is None:

            self._running_task = self._system.aimbot_training_task
            self._running_thread = threading.Thread(target=self._running_task.start)
            self._status_text.set('Aimbot - Training')
            self._running_thread.start()

    def _run_aimbot_inference_task(self):
        """ Run aimbot """

        if self._running_task is None and self._running_thread is None:
            self._running_task = self._system.aimbot_inference_task
            self._running_thread = threading.Thread(target=self._running_task.start)
            self._status_text.set('Aimbot - Running')
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







