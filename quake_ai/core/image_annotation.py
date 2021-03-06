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

import os
import numpy as np
import re
from PIL import Image
import cv2 as cv

from quake_ai.utils.trigger_model import TriggerModel
from quake_ai.utils.aimbot_model import AimbotModel


class ImageAnnotator:
    """ Automatically annotates images for the aimbot training by
        sliding the triggerbot fov over an entire image.
        For every image in the path it will create a <image_name>_anno.txt file
    """

    def __init__(self, config):
        """ Initialize the annotator, uses the path to the trained trigger model """

        self._config = config
        self._image_path = os.path.join(self._config.training_env_path, 'aimbot_images')

        # Choose type of annotation, either use trigger model or aimbot
        self._annotation_type = config.annotator_type
        if self._annotation_type == 'aimbot':
            self._model_path = config.aimbot_model_path
        else:
            self._model_path = config.trigger_model_path

        self._trigger_fov = config.trigger_fov
        self._aimbot_training_fov = (config.aimbot_train_image_size, config.aimbot_train_image_size)
        self._step_size_height = config.annotator_step_size_height
        self._step_size_width = config.annotator_step_size_width
        # Do not do it here, prevent tensorflow from loading
        self._model = None
        self._file_list = None
        self._current_image_idx = None
        self._num_images_to_process = config.annotator_num_images_per_step

    def startup_annotation(self):
        """ Initialize the trigger model for annotation """

        # Only initialize once!
        if self._model is None:
            if self._annotation_type == 'aimbot':
                self._model = AimbotModel(self._config, self._aimbot_training_fov, self._model_path)
            else:
                self._model = TriggerModel(self._config, self._trigger_fov, self._model_path)

        self._gather_images()

        self._model.init_inference()

    def run_annotation(self):
        """ Run the annotation, will annotate few images
            every run to give the user the possibility to stop
        """

        max_index = min(self._current_image_idx + self._num_images_to_process, len(self._file_list))
        for image_path in self._file_list[self._current_image_idx: max_index]:
            print('Processing image', image_path)

            boxes = []
            if self._annotation_type == 'aimbot':
                boxes = self._run_annotation_aimbot(image_path)
            else:
                boxes = self._run_annotation_trigger(image_path)

            # Create annotation files
            if boxes:
                anno_file_name = os.path.join(self._image_path, image_path.split('.')[0] + ".txt")
                with open(anno_file_name, 'w') as file:
                    for box in boxes:

                        center_x = box[0] + 0.5 * box[2]
                        center_y = box[1] + 0.5 * box[3]
                        # Now find out if the enemy is on-target
                        aim_x = self._aimbot_training_fov[1] / 2.
                        aim_y = self._aimbot_training_fov[0] / 2.
                        # Very simple heuristic, will lead to false-positives
                        class_id = 0
                        on_target_x = np.abs(center_x - aim_x) <= 0.4 * box[2]
                        on_target_y = np.abs(center_y - aim_y) <= 0.4 * box[3]
                        if on_target_x and on_target_y:
                            class_id = 1

                        # Class, MeanX, MeanY, Width, Height
                        file.write(str(class_id) + ' ' + str(center_x / self._aimbot_training_fov[1]) + ' '
                                   + str(center_y / self._aimbot_training_fov[0])
                                   + ' ' + str(box[2] / self._aimbot_training_fov[1]) + ' '
                                   + str(box[3] / self._aimbot_training_fov[0]) + '\n')

            # If there are none, remove the file
            else:
                os.remove(os.path.join(self._image_path, image_path))

        # Look if there are new ones to process
        self._gather_images()

    def shutdown_annotation(self):
        """ Shutdown the trigger model for annotation """

        self._model.shutdown_inference()

    def _run_annotation_trigger(self, image_path):
        """ Run the annotation using the triggerbot
        """

        radius_x = int(self._trigger_fov[0]/2.)
        radius_y = int(self._trigger_fov[1]/2.)

        # Pad image to be able to slide the triggerbot over the entire image
        image = Image.open(os.path.join(self._image_path, image_path))
        padded_image = Image.new(image.mode, (image.width + self._trigger_fov[1],
                                              image.height + self._trigger_fov[0]), (0, 0, 0))
        padded_image.paste(image, (radius_x, radius_y))

        # Collect the x (height) and y (width) CENTER coordinates
        center_x_list = np.arange(0, self._aimbot_training_fov[0] + self._step_size_height, self._step_size_height)
        center_y_list = np.arange(0, self._aimbot_training_fov[1] - self._step_size_width, self._step_size_width)
        # Empty matrix to collect the triggerbot response
        output = np.zeros((self._aimbot_training_fov[0],
                           self._aimbot_training_fov[1]))

        # Perform predictions in batches along x, much faster!
        for y in center_y_list:

            image_batch = np.zeros((len(center_x_list), self._trigger_fov[0],
                                    self._trigger_fov[1], 3))
            for idx, x in enumerate(center_x_list):
                crop = np.array(padded_image.crop((y - radius_y + radius_y, x - radius_x + radius_x,
                                                   y + radius_y + radius_y, x + radius_x + radius_x)))
                image_batch[idx, :, :, :] = crop

            is_on_target = self._model.predict_is_on_target(image_batch)

            for idx, x in enumerate(center_x_list):
                if is_on_target[idx]:
                    half_step_height = int(self._step_size_height / 2.)
                    half_step_width = int(self._step_size_width / 2.)
                    output[x - half_step_height:x + half_step_height, y - half_step_width:y + half_step_width] = 255

        # Find contours in response map
        cnts, hierarchy = cv.findContours(output.astype(np.uint8), cv.RETR_EXTERNAL,
                                              cv.CHAIN_APPROX_SIMPLE)
        # And create bounding boxes for them
        bounding_boxes = [cv.boundingRect(c) for c in cnts]

        # Filter out very small boxes and images with many boxes
        filtered_boxes = []
        for box in bounding_boxes:
            size_ok = box[2] >= 2 * self._step_size_width and box[3] >= 2 * self._step_size_height
            num_ok = len(bounding_boxes) < 4
            if size_ok and num_ok:
                filtered_boxes.append(box)

        return filtered_boxes

    def _run_annotation_aimbot(self, image_path):
        """ Run the annotation using the aimbot
        """

        # Pad image to be able to slide the triggerbot over the entire image
        image = Image.open(os.path.join(self._image_path, image_path))
        boxes, scores, classes, nums = self._model.predict(np.array(image))

        output_boxes = []
        for element_id in np.arange(nums):
            box = boxes[0, element_id, :]

            left = self._aimbot_training_fov[1] * box[0]
            top = self._aimbot_training_fov[0] * box[1]
            width = self._aimbot_training_fov[1] * (box[2] - box[0])
            height = self._aimbot_training_fov[0] * (box[3] - box[1])

            output_boxes.append([left.numpy(), top.numpy(), width.numpy(), height.numpy()])

        return output_boxes

    def _gather_images(self):

        # Now collect all images(!) that do not have an annotation file yet
        self._file_list = []
        self._current_image_idx = 0
        file_list = os.listdir(self._image_path)
        for file in file_list:
            is_image = '.png' in file
            anno_exists = is_image and os.path.isfile(os.path.join(self._image_path,
                                                                   file.split('.')[0] + ".txt"))

            if is_image and not anno_exists:
                self._file_list.append(file)


def _get_latest_inc(path):
    """ Search through images in path and get 'oldest' one.
        If there are missing image numbers in between, this will be wrong
    """

    images = [os.path.join(path, image) for image in os.listdir(path) if '.png' in image]

    if not images:
        return 0
    else:
        return int(re.search('(?P<inc>\d+).png$', max(images, key=os.path.getctime)).group('inc'))





