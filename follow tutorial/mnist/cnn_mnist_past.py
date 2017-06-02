import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse, sys

SEED = 66478
BATCH_SIZE = 64
REGULAR_RATE = 5e-4

def model():
    pass

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    input = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    data = tf.reshape(input, [-1, 28, 28, 1])

    conv1_w = tf.Variable(
        tf.truncated_normal(
            shape=[5, 5, 1, 32], stddev=0.1, seed=SEED, dtype=tf.float32
        )
    )

    conv1_b = tf.Variable(
        tf.truncated_normal(
            shape=[32], stddev=0.1, seed=SEED, dtype=tf.float32
        )
    )

    conv1 = tf.nn.conv2d(data, conv1_w, strides=[1, 1, 1, 1], padding="SAME")
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv2_w = tf.Variable(
        tf.truncated_normal(
            [5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=tf.float32
        )
    )

    conv2_b = tf.Variable(
        tf.truncated_normal(
            shape=[64], stddev=0.1, seed=SEED, dtype=tf.float32
        )
    )

    conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding="SAME")
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    pool_shape = pool2.get_shape().as_list()
    reshape = tf.reshape(
        pool2,
        [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])

    fc1_w = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal(
            [pool_shape[1] * pool_shape[2] * pool_shape[3], 512], stddev=0.1, seed=SEED, dtype=tf.float32
        )
    )

    fc1_b = tf.Variable(
        tf.truncated_normal(
            shape=[512], stddev=0.1, seed=SEED, dtype=tf.float32
        )
    )

    hidden = tf.nn.relu(tf.matmul(reshape, fc1_w) + fc1_b)

    fc2_w = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal(
            [512, 10], stddev=0.1, seed=SEED, dtype=tf.float32
        )
    )

    fc2_b = tf.Variable(
        tf.truncated_normal(
            shape=[10], stddev=0.1, seed=SEED, dtype=tf.float32
        )
    )

    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    result = tf.matmul(hidden, fc2_w) + fc2_b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=result))

    # regularizers = (tf.nn.l2_loss(fc1_w) + tf.nn.l2_loss(fc1_b) +
    #                 tf.nn.l2_loss(fc2_w) + tf.nn.l2_loss(fc2_b))
    #
    # loss += 5e-4 * regularizers

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.arg_max(result, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={input: batch_xs, labels: batch_ys})

        if (_ % 100 == 0):

            print(sess.run(accuracy, feed_dict={
                input: mnist.validation.images, labels: mnist.validation.labels
            }))

    print(sess.run(accuracy, feed_dict={
        input: mnist.validation.images, labels: mnist.validation.labels
    }))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='D:/v-habao/Python/TF_Study/MNIST_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)