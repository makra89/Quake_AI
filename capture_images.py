import keyboard
import pyautogui as pgui
import win32gui
import os
import re
import time

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

class System:

    def __init__(self, image_path):

        self._image_path_pos = os.path.join(os.path.abspath(image_path), 'pos')
        self._image_path_neg = os.path.join(os.path.abspath(image_path), 'neg')

        if not os.path.exists(self._image_path_pos):
            os.makedirs(self._image_path_pos)
            self._curr_image_pos_inc = 0
        else:
            self._curr_image_pos_inc = self._get_latest_inc(self._image_path_pos) + 1

        if not os.path.exists(self._image_path_neg):
            os.makedirs(self._image_path_neg)
            self._curr_image_neg_inc = 0
        else:
            self._curr_image_neg_inc = self._get_latest_inc(self._image_path_neg) + 1

        self._handle = win32gui.FindWindow(None, "Quake Live")
        if self._handle == 0:
            raise RuntimeError('Cannot find Quake Live window on primary screen')
        else:
            win32gui.SetForegroundWindow(self._handle)
            self._window_rect = WindowRect((win32gui.GetWindowRect(self._handle)))

    def _get_latest_inc(self, path):
        images = [os.path.join(path, image) for image in os.listdir(path)]
        return int(re.search('(?P<inc>\d+).png$', max(images, key=os.path.getctime)).group('inc'))

    def make_screenshot(self):
        return pgui.screenshot(region=(self._window_rect.left,self._window_rect.top,
                                       self._window_rect.right-self._window_rect.left,
                                       self._window_rect.bottom-self._window_rect.top))

    def _save_screenshot(self, path, inc):
        self.make_screenshot().save(os.path.join(path, str(inc) + '.png'))

    def save_pos_screenshot(self):
        self._save_screenshot(self._image_path_pos, self._curr_image_pos_inc)
        self._curr_image_pos_inc += 1

    def save_neg_screenshot(self):
        self._save_screenshot(self._image_path_neg, self._curr_image_neg_inc)
        self._curr_image_neg_inc += 1

    @property
    def window_rect(self):
        return self._window_rect

sys = System('../images')

while True:
    if keyboard.read_key() == CAPTURE_KEY_POS:
        sys.save_pos_screenshot()
    elif keyboard.read_key() == CAPTURE_KEY_NEG:
        sys.save_neg_screenshot()

    time.sleep(0.1)
