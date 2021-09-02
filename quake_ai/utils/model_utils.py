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

import tensorflow as tf
import numpy as np


class _AimMask:
    """ Utility class for marking the aim dot (the user has to set this!) """

    def __init__(self, image_shape, aim_size):
        """ Initialize mask itself, returns tensorflow tensor """

        mask = np.ones((image_shape[0], image_shape[1], 3))
        mask[int(0.5 * (image_shape[0] - aim_size)): int(0.5 * (image_shape[0] - aim_size) + aim_size),
             int(0.5 * (image_shape[1] - aim_size)): int(0.5 * (image_shape[1] - aim_size) + aim_size), :] = 0
        self._tf_mask = tf.cast(tf.convert_to_tensor(mask), tf.float32)

    @property
    def mask(self):

        return self._tf_mask


class ImageLogger:
    """ Utility class for logging images in tensorboard """

    def __init__(self, name, logdir, max_images=2):
        """ Initialize it, creates image tab named "name" in tensorboard """

        self.file_writer = tf.summary.create_file_writer(logdir)
        self.name = name
        self._max_images = max_images

    def __call__(self, image, label):
        """ Will be performed for every batch call (but only two images will be saved) """

        with self.file_writer.as_default():
            tf.summary.image(self.name, image,
                             step=0,  # Always overwrite images, do not save
                             max_outputs=self._max_images)

        return image, label
