import keyboard
import pyautogui as pgui
import win32gui
import os
import re
import time
from tensorflow import keras
import numpy as np
import tensorflow as tf

# \devmap vortexportal
# Add bots
# \bot_pause "1"
# \g_dropinactive 0
# sv_timeout 6000

WINDOW_MARGIN = 50
CAPTURE_KEY_POS = 'e'
CAPTURE_KEY_NEG = 'r'

class WindowRect:

    def __init__(self, rect_tuple_ltrb):
        self.left   = rect_tuple_ltrb[0] + WINDOW_MARGIN
        self.top    = rect_tuple_ltrb[1] + WINDOW_MARGIN
        self.right  = rect_tuple_ltrb[2] - WINDOW_MARGIN
        self.bottom = rect_tuple_ltrb[3] - WINDOW_MARGIN

@tf.function
def _preprocess_image(image):
    image = tf.cast(image, tf.float32)
    image = (image - tf.reduce_mean(image)) / tf.math.reduce_std(image)
    return image

class System:

    def __init__(self):

        self._handle = win32gui.FindWindow(None, "Quake Live")
        if self._handle == 0:
            raise RuntimeError('Cannot find Quake Live window on primary screen')
        else:
            win32gui.SetForegroundWindow(self._handle)
            self._window_rect = WindowRect((win32gui.GetWindowRect(self._handle)))

    def make_screenshot(self):
        return pgui.screenshot(region=(self._window_rect.left,self._window_rect.top,
                                       self._window_rect.right-self._window_rect.left,
                                       self._window_rect.bottom-self._window_rect.top))

    @property
    def window_rect(self):
        return self._window_rect

sys = System()
model = keras.models.load_model('best_model.hdf5')

pgui.PAUSE = 0.02


while(True):
    screenshot = np.expand_dims(np.array(sys.make_screenshot()), axis=0)
    screenshot = _preprocess_image(screenshot)
    predict = model(screenshot)
    if(np.argmax(predict) == 1):
        pgui.mouseDown();
        time.sleep(pgui.PAUSE)
        pgui.mouseUp();
    else:
        time.sleep(0.02)
