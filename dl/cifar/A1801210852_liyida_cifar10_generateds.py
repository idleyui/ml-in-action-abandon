import tensorflow as tf
from PIL import Image
import numpy as np
import os

image_train_path = './cifar-10/train/'
image_test_path = './cifar-10/test/'
tfRecord_train = './data/mnist_train.tfrecords'
tfRecord_test = './data/mnist_test.tfrecords'
data_path = './data'
resite_height = 32
resize_width = 32


def write_tfRecord(tfRecordName, image_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)

    label_index = 0
    process_cnt = 0
    for label in os.listdir(image_path):
        for img_file in os.listdir(image_path + label):
            img = Image.open(image_path + label + '/' + img_file)
            img = img.convert('L')
            img_raw = img.tobytes()
            labels = [0] * 10
            labels[label_index] = 1

            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
            }))
            writer.write(example.SerializeToString())
            process_cnt += 1
            if process_cnt % 100 == 0:
                print("process cnt:", process_cnt)

        label_index += 1

    writer.close()
    print("write tfrecord successful")


def generate_tfRecord():
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("create data dir")
    else:
        print('data dir already exists')
    write_tfRecord(tfRecord_train, image_train_path)
    write_tfRecord(tfRecord_test, image_test_path)


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([10], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([1024])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)

    return img, label


def get_tfrecord(num, isTrain=True):
    tfRecord_path = tfRecord_train if isTrain else tfRecord_test

    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=num,
                                                    num_threads=2,
                                                    capacity=50000,
                                                    min_after_dequeue=49880)
    return img_batch, label_batch


if __name__ == '__main__':
    generate_tfRecord()
    # get_tfrecord(200)
