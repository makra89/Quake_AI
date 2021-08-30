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

from os.path import join, dirname
import configparser


class QuakeAiConfig:
    """ Configuration object for Quake AI
        Partially this config is used for saving values (i.e. current learning rate).
        Not so nice, but easiest way to do :/
    """

    def __init__(self, config_path):
        """ Initialize the configuration using a path.
            If not existing a default one will be generated.
        """

        self._path = config_path
        self._config = configparser.ConfigParser()

        # Try to load an existing configuration
        try:
            with open(self._path, 'r') as file:
                self._config.read_file(file)
        except:
            # No config available --> create default one and write it
            self._config = self._create_default_config()
            with open(self._path, 'w') as file:
                self._config.write(file)

    def update_from_file(self):
        """ Update configuration object from the file """

        with open(self._path, 'r') as file:
            self._config.read_file(file)

    @property
    def training_env_path(self):
        return self._config['GENERAL']['training_env_path']

    @property
    def trigger_model_path(self):
        return self._config['TRIGGERBOT']['model_path']

    @property
    def trigger_fov(self):
        return int(self._config['TRIGGERBOT']['fov_height']), int(self._config['TRIGGERBOT']['fov_width'])

    @property
    def trigger_aim_size(self):
        return int(self._config['TRIGGERBOT']['aim_size'])

    @property
    def trigger_train_lr(self):
        return float(self._config['TRIGGERBOT-TRAINING']['learning_rate'])

    @trigger_train_lr.setter
    def trigger_train_lr(self, value):
        self._config['TRIGGERBOT-TRAINING']['learning_rate'] = str(value)
        # Write changes to file
        with open(self._path, 'w') as file:
            self._config.write(file)

    @property
    def trigger_train_batch_size(self):
        return int(self._config['TRIGGERBOT-TRAINING']['batch_size'])

    @property
    def trigger_train_shuffle_size(self):
        return int(self._config['TRIGGERBOT-TRAINING']['shuffle_size'])

    @property
    def trigger_train_epochs(self):
        return int(self._config['TRIGGERBOT-TRAINING']['num_epochs'])

    @property
    def trigger_train_capture_pos(self):
        return self._config['TRIGGERBOT-TRAINING']['capture_key_pos']

    @property
    def trigger_train_capture_neg(self):
        return self._config['TRIGGERBOT-TRAINING']['capture_key_neg']

    @property
    def aimbot_train_image_fov(self):
        return int(self._config['AIMBOT-TRAINING']['image_height']), \
               int(self._config['AIMBOT-TRAINING']['image_width'])

    @property
    def aimbot_train_capture_key(self):
        return self._config['AIMBOT-TRAINING']['capture_key']

    @property
    def annotator_num_images_per_step(self):
        return int(self._config['AIMBOT-ANNOTATION']['num_images_per_step'])

    @property
    def annotator_step_size_height(self):
        return int(self._config['AIMBOT-ANNOTATION']['step_size_height'])

    @property
    def annotator_step_size_width(self):
        return int(self._config['AIMBOT-ANNOTATION']['step_size_width'])

    def _create_default_config(self):
        """ Creates default config """

        config = configparser.ConfigParser()

        # Default training environment is parallel to the used configuration
        default_training_environment = join(dirname(self._path), 'training')
        config['GENERAL'] = {'training_env_path': default_training_environment}

        # Default model lies in the repository itself
        default_trigger_model_path = join(dirname(dirname(dirname(__file__))), 'default_models\\trigger_model.hdf5')

        config['TRIGGERBOT'] = {'model_path': default_trigger_model_path,
                                'fov_height': '100',
                                'fov_width': '100',
                                'aim_size': '4'}

        config['TRIGGERBOT-TRAINING'] = {'learning_rate': '0.01',
                                         'batch_size': '30',
                                         'shuffle_size': '100',
                                         'num_epochs': '20',
                                         'capture_key_pos': 'e',
                                         'capture_key_neg': 'r'}

        config['AIMBOT-TRAINING'] = {'image_height': '300',
                                     'image_width': '530',
                                     'capture_key': 'e'}

        config['AIMBOT-ANNOTATION'] = {'num_images_per_step': 2,
                                       'step_size_height': 8,
                                       'step_size_width': 4}

        return config
