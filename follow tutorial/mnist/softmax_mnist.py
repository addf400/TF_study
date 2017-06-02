import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse, sys

SEED = 66478

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    input = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    w = tf.Variable(
        tf.truncated_normal(
            shape=[784, 10], stddev=0.1, seed=SEED, dtype=tf.float32
        )
    )
    b = tf.Variable(
        tf.truncated_normal(
            shape=[10], stddev=0.1, seed=SEED, dtype=tf.float32
        )
    )

    result = tf.matmul(input, w) + b
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=result)
    )

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={input: batch_xs, labels: batch_ys})

        if (_ % 100 == 0):

            correct_prediction = tf.equal(tf.arg_max(result, 1), tf.arg_max(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={
                input: mnist.test.images,labels: mnist.test.labels
            }))

    correct_prediction = tf.equal(tf.arg_max(result, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={
        input: mnist.test.images, labels: mnist.test.labels
    }))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='D:/v-habao/Python/TF_Study/MNIST_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)