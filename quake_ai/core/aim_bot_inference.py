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
from threading import Thread, Event, Lock
from queue import Queue
import cv2
import imgaug

OVERLAY_HEARTBEAT_SEC = 0.02  # Do not update/query the overlay too often
SCORE_THRESH = 0.95  # Do not track targets with scores lower than this one
MAX_MOVE = 10  # Maximum number of pixels moved in one action
NO_PREDICT_UPDATE_CYCLES = 5  # Maximum number of cycles the tracker goes on tracking without predictor updates
PREDICTOR_SLEEP = 0.05
TRACKER_SLEEP = 0.01
MAX_MOVEMENT_FREQ = 100  # Hz


class BoundingBox:

    def __init__(self, left, top, width, height):

        self.left = int(left)
        self.top = int(top)
        self.width = int(width)
        self.height = int(height)
        self._imgaug_box = imgaug.augmentables.bbs.BoundingBox(x1=left, x2=left+width, y1=top, y2=top+height)

    def as_tuple(self):
        return self.left, self.top, self.width, self.height

    def as_open_cv_rect(self):
        return (self.left, self.top), (self.left + self.width, self.top + self.height)

    def iou(self, box):
        return self._imgaug_box.iou(box._imgaug_box)


class Aimbot:
    """ Main class for aimbot inference """

    def __init__(self, config, screenshot_func=None, aim_pos_func=None, tracker_screenshot_func=None):
        """ Initializes the aimbot """

        self._config = config
        self._activation_hook = None
        self._model_path = config.aimbot_model_path
        self._screenshot_func = screenshot_func
        self._tracker_screenshot_func = tracker_screenshot_func
        self._aim_pos_func = aim_pos_func

        self._fov = (config.aimbot_inference_image_size, config.aimbot_inference_image_size)
        # Do not do it here, prevent tensorflow from loading
        self._model = None
        # Aimbot is switched off by default
        self._active = False
        self._overlay = None

        # Threading members
        self._shared_predictor = None
        self._shared_box_buffer = SharedBoxBuffer()
        self._stop_event = Event()
        self._active_event = Event()
        self._box_queue = Queue()
        self._box_eval_thread = None
        self._overlay_thread = None

    def init_inference(self):
        """ Initialize the model for inference (tries to load the saved model) """

        # Only initialize once!
        if self._model is None:
            self._model = AimbotModel(self._config, self._fov, self._model_path)

        self._shared_predictor = ThreadSafePredictor(predict_func=self._model.predict,
                                                     screenshot_func=self._screenshot_func,
                                                     fov=self._fov)

        self._model.init_inference()
        pydirectinput.PAUSE = 0.0
        self._activation_hook = keyboard.on_press_key(self._config.aimbot_activation_key,
                                                      self._activate_aimbot)

        self._box_eval_thread = Thread(target=eval_boxes_worker, args=(self._stop_event, self._active_event,
                                                                       self._shared_predictor,
                                                                       self._shared_box_buffer, self._fov,
                                                                       self._tracker_screenshot_func))
        self._box_eval_thread.start()

        self._overlay_thread = Thread(target=overlay_worker, args=(self._stop_event, self._shared_box_buffer,
                                                                   self._aim_pos_func(), self._fov))
        self._overlay_thread.start()

    def run_inference(self):
        """ Run the aimbot for one screenshot """

        if self._active:

            self._shared_predictor.update()
            time.sleep(PREDICTOR_SLEEP)

    def shutdown_inference(self):
        """ Stop the inference """

        self._model.shutdown_inference()
        pydirectinput.PAUSE = 0.1
        if self._activation_hook is not None:
            keyboard.unhook(self._activation_hook)

        self._stop_event.set()
        self._box_eval_thread.join()
        self._box_eval_thread = None
        self._overlay_thread.join()
        self._overlay_thread = None

    def _activate_aimbot(self, _):

        self._active = not self._active

        if self._active:
            self._active_event.set()
        else:
            self._active_event.clear()


class ThreadSafePredictor:

    def __init__(self, screenshot_func, predict_func, fov):

        self._screen_func = screenshot_func
        self._predict_func = predict_func
        self._fov = fov
        self._mutex = Lock()

        self._predict_screen = None
        self._predict_time = None
        self._boxes = None
        self._updated = False

    def update(self):

        screen = np.array(self._screen_func())
        # Tensorflow releases the GIL!
        boxes, scores, classes, nums = self._predict_func(screen)
        # No reason to forward boxes if there aren't any
        nums = int(nums.numpy())
        if nums > 0:
            boxes = boxes[0].numpy()[:nums, :]
            self._mutex.acquire()
            self._boxes = [BoundingBox(box[0] * self._fov[1], box[1] * self._fov[0],
                                       self._fov[1] * (box[2] - box[0]),
                                       self._fov[0] * (box[3] - box[1])) for box in boxes]

            self._predict_time = time.time()
            self._predict_screen = screen
            self._updated = True
            self._mutex.release()

        else:
            self._mutex.acquire()
            self._updated = False
            self._mutex.release()

    def has_update(self):

        with self._mutex:
            return self._updated

    def get_updates(self):

        with self._mutex:
            self._updated = False
            return self._boxes, self._predict_screen, self._predict_time


class SharedBoxBuffer:

    def __init__(self):

        self._mutex = Lock()

        self._reset_flag = True
        self._predict_box = None
        self._tracked_box = None
        self._updated = False

    def update(self, reset_flag, predict_box=None, tracked_box=None):

        self._mutex.acquire()
        self._reset_flag = reset_flag
        if predict_box:
            self._predict_box = predict_box
        if tracked_box:
            self._tracked_box = tracked_box
        self._updated = True

        self._mutex.release()

    def has_update(self):

        with self._mutex:
            return self._updated

    def get_update(self):

        with self._mutex:
            self._updated = False
            return self._reset_flag, self._predict_box, self._tracked_box


def eval_boxes_worker(stop_event, active_event, shared_predictor, shared_box_buffer, fov, screenshot_func):
    """ Receives boxes, picks one box to target and performs mouse movements """
    # TODO: Try to encapsulate this mess!
    pydirectinput.PAUSE = 0.0

    # Some function-global variables
    cycles_without_update = 0
    last_move_time = 0.0
    last_predict_time = 0.0
    tracked_box = None

    def move(x, y, last_time):
        """ Perform the mouse movement """
        if active_event.is_set() and time.time() - last_time > 1./MAX_MOVEMENT_FREQ:
            pydirectinput.move(min(MAX_MOVE, x), min(MAX_MOVE, y))
            last_time = time.time()
        return last_time

    def calc_rel(in_box):
        """ Calculate the relative distance in pixels to the aim"""
        mean_x = in_box.left + 0.5 * in_box.width
        mean_y = in_box.top + 0.5 * in_box.height
        return mean_x - fov[1] / 2., mean_y - fov[0] / 2.

    def check_tracked_candidate(predict_boxes, tracked_candidate):
        """ Check all predicted boxes for our tracked candidate """
        if tracked_candidate:
            best_iou = 0.0
            for predict_box in predict_boxes:
                iou = tracked_candidate.iou(predict_box)
                if iou > best_iou:
                    best_iou = iou
                    tracked_candidate = predict_box

            # We only need some evidence that the tracked and predicted boxes are the same
            if tracked_candidate and best_iou < 0.4:
                tracked_candidate = None

        return tracked_candidate

    tracker = None
    while not stop_event.is_set():
        # Check for predictor update
        if shared_predictor.has_update():

            boxes, screen, last_predict_time = shared_predictor.get_updates()

            # Approach 1: Use tracked box (if existing) and only update its location
            candidate = check_tracked_candidate(boxes, tracked_box)

            # Fallback approach: Just take the box which is closest to the aim
            if not candidate:

                current_min_dist = 1000
                current_shortest_rel = (0, 0)
                for box in boxes:

                    rel_x, rel_y = calc_rel(box)
                    dist = np.sqrt(rel_x ** 2 + rel_y ** 2)

                    # Pick closest one
                    if dist < current_min_dist:
                        current_min_dist = dist
                        current_shortest_rel = (int(rel_x), int(rel_y))
                        candidate = box

            if candidate:
                # Init tracker, MedianFlow is a good compromise, CSRT is better, but way slower
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(screen, candidate.as_tuple())

                shared_box_buffer.update(False, candidate, candidate)
                last_move_time = move(current_shortest_rel[0], current_shortest_rel[1], last_move_time)
                cycles_without_update = 0
                tracked_box = candidate
            else:
                # Always increase counter, We cannot trust the tracker to report a failure in tracking
                cycles_without_update += 1
                if tracker and cycles_without_update <= NO_PREDICT_UPDATE_CYCLES:
                    ret, bbox_cv = tracker.update(np.array(screenshot_func()))
                    if ret:
                        box = BoundingBox(bbox_cv[0], bbox_cv[1], bbox_cv[2], bbox_cv[3])
                        tracked_box = box
                        rel_x, rel_y = calc_rel(box)
                        last_move_time = move(int(rel_x), int(rel_y), last_move_time)
                        shared_box_buffer.update(False, tracked_box=box)
                else:
                    shared_box_buffer.update(True)
                    tracked_box = None
        # No update from predictor --> try to track
        else:
            time.sleep(TRACKER_SLEEP)  # Aim at ~ 100 FPS
            # Only track for a certain time (use case: in between to predictions)
            if tracker and time.time() - last_predict_time < (2 * PREDICTOR_SLEEP):
                ret, bbox_cv = tracker.update(np.array(screenshot_func()))
                if ret:
                    box = BoundingBox(bbox_cv[0], bbox_cv[1], bbox_cv[2], bbox_cv[3])
                    tracked_box = box
                    rel_x, rel_y = calc_rel(box)
                    last_move_time = move(int(rel_x), int(rel_y), last_move_time)
                    shared_box_buffer.update(False, tracked_box=box)
            else:
                shared_box_buffer.update(True)
                tracked_box = None

    pydirectinput.PAUSE = 0.1


def overlay_worker(stop_event, shared_box_buffer, aim_pos, fov):
    """ Overlay process, draws boxes coming from the eval process """

    overlay = Overlay(fov, aim_pos)

    while not stop_event.is_set():
        if shared_box_buffer.has_update():
            reset, predict_box, tracked_box = shared_box_buffer.get_update()

            if not reset:
                overlay.activate_predict_rect(predict_box, tracked_box)
            else:
                overlay.deactivate_predict_rect()

            overlay.update()
            time.sleep(OVERLAY_HEARTBEAT_SEC)

        else:
            overlay.update()
            time.sleep(OVERLAY_HEARTBEAT_SEC)


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

        self._predict_rect = pygame.Rect(50, 50, 50, 50)
        self._predict_color = self._trans_color

        self._track_rect = pygame.Rect(50, 50, 50, 50)
        self._track_color = self._trans_color

        self.update()

    def activate_predict_rect(self, predict_box, track_box):

        self._predict_rect = pygame.Rect(predict_box.as_tuple())
        self._predict_color = (255, 0, 0)
        self._track_rect = pygame.Rect(track_box.as_tuple())
        self._track_color = (0, 255, 0)

    def deactivate_predict_rect(self):

        self._predict_color = self._trans_color
        self._track_color = self._trans_color

    def update(self):

        self._window.fill(self._trans_color)
        pygame.draw.rect(self._window, self._predict_color, self._predict_rect, 3)
        pygame.draw.rect(self._window, self._track_color, self._track_rect, 3)
        pygame.display.flip()
        pygame.event.pump()


def _set_transparent_clickthrough(hwnd, color, x_pos, y_pos):
    """ Define as transparent, clickthrough window """

    styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    styles |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*color), 0, win32con.LWA_COLORKEY)
    windll.user32.SetWindowPos(hwnd, -1, x_pos, y_pos, 0, 0, 0x0001)



