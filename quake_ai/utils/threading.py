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
    """ Base class for all types of tasks """

    def __init__(self, main_task, init_task_list=[], shutdown_task_list=[]):
        """ Set tasks to be done

        :param main_task: Main tasks
        :type main_task: function reference
        :param init_task_list: List of tasks to be done once before the main task
        :type init_task_list: list of function references
        :param shutdown_task_list: List of tasks to be done once after the main task
        :type shutdown_task_list: list of function references
        """
        self._main_task = main_task
        self._init_task_list = init_task_list
        self._shutdown_task_list = shutdown_task_list
        self._running = False

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class NonBlockingTask(Task):
    """ This type of task should return immediately """

    def __init__(self, main_task, init_task_list=[], shutdown_task_list=[]):
        """ Initialize base class

        :param main_task: Main tasks
        :type main_task: function reference
        :param init_task_list: List of tasks to be done once before the main task
        :type init_task_list: list of function references
        :param shutdown_task_list: List of tasks to be done once after the main task
        :type shutdown_task_list: list of function references
        """

        super(NonBlockingTask, self).__init__(main_task, init_task_list, shutdown_task_list)

    def start(self):
        """ Start the task """

        if not self._running:
            # When state switches to running --> do init tasks once
            for init_task in self._init_task_list:
                init_task()
            self._running = True
            self._main_task()

    def stop(self):
        """ Stop the task (since it returns immediately this is not really necessary) """

        if self._running:
            # State switches from running to stopped --> do shutdown tasks
            for shutdown_task in self._shutdown_task_list:
                shutdown_task()
            self._running = False


class BlockingTask(Task):
    """ This type of task is designed to deal with long-running processes.
        The idea is to partition every long-running process into smaller steps.
        For example only train one epoch at a time and than do this step in a loop that
        can be interrupted.
    """

    def __init__(self, main_task, init_task_list=None, shutdown_task_list=None):
        """ Initialize the task

        :param main_task: Main tasks
        :type main_task: function reference
        :param init_task_list: List of tasks to be done once before the main task
        :type init_task_list: list of function references
        :param shutdown_task_list: List of tasks to be done once after the main task
        :type shutdown_task_list: list of function references
        """

        super(BlockingTask, self).__init__(main_task, init_task_list, shutdown_task_list)

    def start(self):
        """ Start the task """

        if not self._running:
            # Do init tasks once when we switch from stopped --> running
            for init_task in self._init_task_list:
                init_task()

            self._running = True
            while self._running:
                self._main_task()

            # If we reach this point the main loop has been interrupted --> shutdown tasks
            for shutdown_task in self._shutdown_task_list:
                shutdown_task()

    def stop(self):
        """ Stop the task (I think this is not really thread-safe what I am doing here """

        self._running = False
