import re
import numpy as np
import pandas as pd
import self
import tensorflow as tf
from functools import partial
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "../data"
IMAGE_SIZE = [1024, 1024]
EPOCHS = 50


def data_files(self):
    self.TRAINING_FILENAMES, self.VALID_FILENAMES = train_test_split(
        tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec'),
        test_size=0.1, random_state=5
    )
    self.TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')
    print('Train TFRecord Files:', len(self.TRAINING_FILENAMES))
    print('Validation TFRecord Files:', len(self.VALID_FILENAMES))
    print('Test TFRecord Files:', len(self.TEST_FILENAMES))


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def set_data_counts(self):
    data_files(self)
    self.NUM_TRAINING_IMAGES = count_data_items(self.TRAINING_FILENAMES)
    self.NUM_VALIDATION_IMAGES = count_data_items(self.VALID_FILENAMES)
    self.NUM_TEST_IMAGES = count_data_items(self.TEST_FILENAMES)
    self.STEPS_PER_EPOCH = self.NUM_TRAINING_IMAGES // self.BATCH_SIZE
    self.VALID_STEPS = self.NUM_VALIDATION_IMAGES // self.BATCH_SIZE
    print(
        'Dataset: {} training images, {} validation images, {} unlabeled test images'.format(
            self.NUM_TRAINING_IMAGES, self.NUM_VALIDATION_IMAGES, self.NUM_TEST_IMAGES)
    )

    train_csv = pd.read_csv(GCS_PATH + '/train.csv')
    test_csv = pd.read_csv(GCS_PATH + '/test.csv')
    self.total_img = train_csv['target'].size

    self.malignant = np.count_nonzero(train_csv['target'])
    self.benign = self.total_img - self.malignant

    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        self.total_img, self.malignant, 100 * self.malignant / self.total_img))


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def read_tfrecord(example, labeled):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    if labeled:
        label = tf.cast(example['target'], tf.int32)
        return image, label
    idnum = example['image_name']
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames,
                                      num_parallel_reads=AUTOTUNE)  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order)  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def augmentation_pipeline(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image, label


def get_training_dataset(self):
    dataset = load_dataset(self.TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(augmentation_pipeline, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(self.BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_validation_dataset(self, ordered=False):
    dataset = load_dataset(self.VALID_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(self.BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_test_dataset(self, ordered=False):
    dataset = load_dataset(self.TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(self.BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title("MALIGNANT")
        else:
            plt.title("BENIGN")
        plt.axis("off")



def prepare_data(self):
    set_data_counts(self)
    self.train_dataset = get_training_dataset(self)
    self.valid_dataset = get_validation_dataset(self)
    self.test_dataset = get_test_dataset(self)

if __name__ == "__main__":
    # prepare_predict()
    prepare_data(self)



