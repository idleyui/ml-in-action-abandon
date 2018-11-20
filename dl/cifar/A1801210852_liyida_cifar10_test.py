# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import A1801210852_liyida_cifar10_backward as mnist_backward
import A1801210852_liyida_cifar10_forward as  mnist_forward
import A1801210852_liyida_cifar10_generateds as mnist_generateds

TEST_INTERVAL_SECS = 5
TEST_NUM = 100


def tes():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])  # init x by input node number
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])  # init y_ by output node number
        y = mnist_forward.forward(x, None)  # init y by forward

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)  # get exponential moving average
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)  # init saver which could restore ema

        # calculate accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        img_batch, label_batch = mnist_generateds.get_tfrecord(TEST_NUM, isTrain=False)

        # while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)  # load ckpt model
            # if ckpt and ckpt.model_checkpoint_path:  # if model exist
            if ckpt and ckpt.all_model_checkpoint_paths:  # if model exist
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                for path in ckpt.all_model_checkpoint_paths:
                    # saver.restore(sess, ckpt.model_checkpoint_path)  # restore model
                    saver.restore(sess, path)  # restore model
                    global_step = path.split('/')[-1].split('-')[-1]  # restore global step

                    xs, ys = sess.run([img_batch, label_batch])  # train batch size round

                    accuracy_score = sess.run(accuracy, feed_dict={x: xs,
                                                                   y_: ys})  # calculate accuracy
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                coord.request_stop()
                coord.join(threads)

            else:  # no checkpoint return
                print('No checkpoint file found')
                return
        time.sleep(TEST_INTERVAL_SECS)  # sleep time


def main():
    # mnist = input_data.read_data_sets("./data/", one_hot=True)
    # mnist = input_data.read_data_sets("../../Datasets/mnist", one_hot=True)
    # tes(mnist)
    tes()


if __name__ == '__main__':
    main()
