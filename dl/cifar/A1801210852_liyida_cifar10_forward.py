import tensorflow as tf

# INPUT_NODE = 784  # input node number
INPUT_NODE = 1024 # input node number
OUTPUT_NODE = 10  # output node number
LAYER1_NODE = 500  # layer1 node number


def get_weight(shape, regularizer):  # get weight function
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))  # init weight
    if regularizer != None: tf.add_to_collection('losses',
                                                 tf.contrib.layers.l2_regularizer(regularizer)(w))  # regularization
    return w


def get_bias(shape):  # get bias function
    b = tf.Variable(tf.zeros(shape))  # init b by shape
    return b


def forward(x, regularizer):  # forward propagation function
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer) # get weight by input node number, layer1 number and regularizer
    b1 = get_bias([LAYER1_NODE]) #
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
