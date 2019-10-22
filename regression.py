from collections import namedtuple
import tensorflow as tf

from data_creator import generate_usages, create_food_matrix

TRAIN_PROPORTION = 0.7

CrossData = namedtuple('CrossData', 'x_train x_test y_train y_test'.split())

def create_train_test_data(num, food_matrix):
    usages = [generate_usages(food_matrix) for _ in range(num)]
    x, y = list(zip(*usages))
    idx = int(TRAIN_PROPORTION*num)
    return CrossData(
        x_train=x[:idx],
        x_test=x[idx:],
        y_train=y[:idx],
        y_test=y[idx:]
    )


if __name__ == '__main__':
    num_samples = 100
    learning_rate = 0.01

    food_matrix = create_food_matrix('item_matrix.csv')
    input_dim = len(food_matrix.records.keys())
    out_dim = len(food_matrix.columns)

    train_test = create_train_test_data(num_samples, food_matrix)

    W = tf.Variable(tf.random_uniform([input_dim, out_dim]))
    b = tf.Variable(tf.zeros([out_dim]))
    xs = tf.placeholder('float')
    ys = tf.placeholder('float')

    output = tf.add(tf.matmul(xs, W), b)
    cost = tf.reduce_mean(tf.square(output - ys))

    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    print(train_test)

