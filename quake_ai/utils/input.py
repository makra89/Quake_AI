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
import pydirectinput
import time


class Trajectory:
    """ Encapsulates bezier trajectory """

    def __init__(self, start_point, end_point, short_traj_length, move_per_step,
                 steps_per_cycle, time_until_outdated, short_traj_smooth_fac):

        self._start = start_point
        self._end = end_point
        self._init_time = time.time()
        self._time_until_outdated = time_until_outdated
        self._short_traj_smooth_fac = short_traj_smooth_fac
        self._steps_per_cycle = steps_per_cycle

        # Two kinds of trajectories:
        # - Bezier curves used to move to targets that are rather far away
        # - Short trajectories, linear movement is used (use case: stay locked on to target)
        length = np.linalg.norm((end_point[0] - start_point[0], end_point[1] - start_point[1]))
        if length <= short_traj_length:
            self._short_traj = True
            self._current_idx = 0
            self._max_idx = 1
        else:
            self._short_traj = False
            # Some heuristic to get a nice looking curve
            arc_point_x = start_point[0] * 0.75 + end_point[0] * 0.25
            arc_point = (arc_point_x, end_point[1])

            num_segments = int(length / move_per_step)  # underestimated since we are moving along a Bezier curve!
            step_size = 1./num_segments

            t_points = np.arange(0, 1.0, step_size)
            points = np.array([start_point, arc_point, end_point])
            self._curve = Bezier.Curve(t_points, points)
            self._max_idx = np.shape(self._curve)[0] - 1
            self._current_idx = 0

    def is_outdated(self):
        """ Trajectory can be outdated due to:
            - Time
            - Length exceeded
        """

        current_time = time.time()
        outdated_time = (current_time - self._init_time) > self._time_until_outdated
        outdated_length = self._current_idx >= self._max_idx

        return outdated_time or outdated_length

    def move_along(self):
        """ Move next step along trajectory
            Note: For short trajectories there is only one step!
        """

        if self._short_traj:
            move_x = (self._end[0] - self._start[0]) / self._short_traj_smooth_fac
            move_y = (self._end[1] - self._start[1]) / self._short_traj_smooth_fac
            pydirectinput.moveTo(int(self._start[0] + move_x),
                                 int(self._start[1] + move_y))
            self._current_idx += 1
        else:
            for step in range(self._steps_per_cycle):
                if not self.is_outdated():
                    pydirectinput.moveTo(int(self._curve[self._current_idx, 0]),
                                         int(self._curve[self._current_idx, 1]))
                self._current_idx += 1


class Mouse:
    """ Abstracts mouse input """

    def __init__(self, config=None):

        self._config = config
        self._current_trajectory = None
        pydirectinput.PAUSE = 0.1  # Timeout for input

    def set_timeout(self, timeout):

        pydirectinput.PAUSE = timeout

    def move_rel(self, x, y):

        if self._current_trajectory and not self._current_trajectory.is_outdated():
            self._current_trajectory.move_along()
        else:
            current_pos = pydirectinput.position()
            move_to_pos = (current_pos[0] + x, current_pos[1] + y)
            self._current_trajectory = Trajectory(current_pos, move_to_pos,
                                                  self._config.mouse_short_traj_max_length,
                                                  self._config.mouse_move_per_step,
                                                  self._config.mouse_steps_per_cycle,
                                                  self._config.mouse_time_until_traj_outdated,
                                                  self._config.mouse_short_traj_smooth_fac)
            self._current_trajectory.move_along()

        return self._current_trajectory

    def left_mouse_down(self):

        pydirectinput.mouseDown()

    def left_mouse_up(self):

        pydirectinput.mouseUp()
