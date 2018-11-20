import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import A1801210852_liyida_cifar10_forward as mnist_forward
import os
import A1801210852_liyida_cifar10_generateds as mnist_generateds

BATCH_SIZE = 100  # train batch size
LEARNING_RATE_BASE = 0.1  # learning rate
LEARNING_RATE_DECAY = 0.99  # decay
REGULARIZER = 0.0001  # regularizer
STEPS = 50000  # max steps
MOVING_AVERAGE_DECAY = 0.99  # ema decay
MODEL_SAVE_PATH = "./model/"  # model save path
MODEL_NAME = "mnist_model"  # model file name
train_num_examples=60000


def backward():
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])  # init x by input node number
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])  # init y by output node number
    y = mnist_forward.forward(x, REGULARIZER)  # forward propagation
    global_step = tf.Variable(0, trainable=False)

    # loss with y and y_
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))  # cross entropy
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # exponential decay learning rate
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # gradient descent train step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)  # use decay and global step get ema
    ema_op = ema.apply(tf.trainable_variables())  # calculate ema
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()  # init saver
    img_batch, label_batch = mnist_generateds.get_tfrecord(BATCH_SIZE)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)  # check if exist saved model
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # restore saved model

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            # xs, ys = mnist.train.next_batch(BATCH_SIZE)  # train batch size round
            xs, ys = sess.run([img_batch, label_batch])  # train batch size round
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})  # run session
            # every 1000 steps, print training steps and loss and save model
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop()
        coord.join(threads)


def main():
    # mnist = input_data.read_data_sets("data/", one_hot=True)  # load dataset
    # backward(mnist)
    backward()


if __name__ == '__main__':
    main()
