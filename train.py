import tensorflow as tf
import tensorflow.keras as keras
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime

BATCH_SIZE = 5
SHUFFLE_BUFFER = 20
NUM_CLASSES = 2
CROP_SIZE = 200

class ImageLogger:

    def __init__(self, name, logdir, max_images=2):
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.name = name
        self._max_images = 2

    def __call__(self, image, label):
        with self.file_writer.as_default():
            tf.summary.image(self.name, image,
                             step=0, # Always overwrite images, do not save
                             max_outputs=self._max_images)

        return image, label

def _preprocess_image(image):
    image = tf.cast(image, tf.float32)
    image = (image - tf.reduce_mean(image)) / tf.math.reduce_std(image)

    return image

def _parse_function(image_path, label):
    bits = tf.io.read_file(image_path)
    image = tf.image.decode_png(bits, channels=3)
    image = _preprocess_image(image)

    label = tf.one_hot(label, NUM_CLASSES)
    label = tf.reshape(label, (2,))

    return image, label

def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)


    # Rotate the image
    #k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    #image = tf.image.rot90(image, k)

    return image, label


def create_dataset(image_paths_labels, augment=True, shuffle=True, image_logger=None):
    # This works with arrays as well
    dataset = tf.data.Dataset.from_generator(lambda: image_paths_labels, (tf.string, tf.int32))

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Augmentation
    if augment:
        dataset = dataset.map(_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Set batch size
    dataset = dataset.batch(BATCH_SIZE)

    if image_logger is not None:
        dataset = dataset.map(image_logger, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Set the number of datapoints you want to load and shuffle
    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER)

    dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def create_model(image_shape):


    image_input = keras.Input(shape=image_shape)
    cropped = keras.layers.experimental.preprocessing.CenterCrop(CROP_SIZE, CROP_SIZE)(image_input)

    conv = keras.layers.Conv2D(filters=16, strides=(2,2), kernel_size=3, activation='relu')(cropped)
    conv = keras.layers.Conv2D(filters=32, strides=(2,2), kernel_size=3, activation='relu')(conv)
    pooling = keras.layers.MaxPooling2D()(conv)
    conv = keras.layers.Conv2D(filters=64, strides=(2,2), kernel_size=3, activation='relu')(pooling)
    pooling = keras.layers.MaxPooling2D()(conv)

    # Dense part
    flat = keras.layers.Flatten()(pooling)
    dense = keras.layers.Dense(units=32, activation='relu')(flat)
    out = keras.layers.Dense(units=NUM_CLASSES, activation='softmax')(dense)

    return keras.models.Model(inputs=image_input, outputs=out)


pos_path = os.path.abspath('../images/pos')
pos_path_labels = [(os.path.join(pos_path, image), 1) for image in os.listdir(pos_path)]

neg_path = os.path.abspath('../images/neg')
neg_path_labels = [(os.path.join(neg_path, image), 0) for image in os.listdir(neg_path)]

all_paths_labels = pos_path_labels + neg_path_labels
paths_labels_train, paths_labels_test = train_test_split(all_paths_labels, shuffle=True, test_size=0.20)

logdir = os.path.join(os.getcwd(),"logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
image_logger = ImageLogger('input', logdir)

data_train = create_dataset(paths_labels_train, image_logger=None)
data_test = create_dataset(paths_labels_test)

model = create_model((956, 1836, 3))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

filepath= os.path.join(os.getcwd(),"best_model.hdf5")
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=1e-7, verbose=1)

logdir = os.path.join(os.getcwd(),"logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model.fit(x=data_train, validation_data=data_test,
          epochs=100, callbacks=[checkpoint, reduce_lr, tensorboard_callback])
