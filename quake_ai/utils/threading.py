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

from abc import ABC, abstractmethod


class Task(ABC):

    def __init__(self, task, init_tasks=None, shutdown_tasks=None):

        self._task = task
        self._init_tasks = init_tasks
        self._shutdown_tasks = shutdown_tasks
        self._running = False

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class NonBlockingTask(Task):

    def __init__(self, task, init_tasks=None, shutdown_tasks=None):

        super(NonBlockingTask, self).__init__(task, init_tasks, shutdown_tasks)

    def start(self):

        if not self._running:
            if self._init_tasks:
                for init_task in self._init_tasks:
                    init_task()
            self._running = True
            self._task()

    def stop(self):

        if self._running:
            if self._shutdown_tasks:
                for shutdown_task in self._shutdown_tasks:
                    shutdown_task()
            self._running = False


class BlockingTask(Task):

    def __init__(self, task, init_tasks=None, shutdown_tasks=None):

        super(BlockingTask, self).__init__(task, init_tasks, shutdown_tasks)

    def start(self):

        if not self._running:
            if self._init_tasks:
                for init_task in self._init_tasks:
                    init_task()

            self._running = True
            while self._running:
                self._task()

            if self._shutdown_tasks:
                for shutdown_task in self._shutdown_tasks:
                    shutdown_task()

    def stop(self):

        self._running = False
