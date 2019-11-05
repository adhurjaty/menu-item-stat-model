from collections import namedtuple
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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
    num_samples = 300
    learning_rate = 0.02
    training_epochs = 1000

    food_matrix = create_food_matrix('item_matrix.csv')
    input_dim = len(food_matrix.records.keys())
    out_dim = len(food_matrix.columns)

    train_test = create_train_test_data(num_samples, food_matrix)
    x_train = np.array(train_test.x_train)
    y_train = np.array(train_test.y_train)

    # scaler = MinMaxScaler()

    W = tf.Variable(tf.random.uniform([input_dim, out_dim]), name='W')
    xs = tf.placeholder('float')
    ys = tf.placeholder('float')

    y_pred = tf.matmul(xs, W)
    cost = tf.reduce_mean(tf.square(y_pred - ys))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()

        for epoch in range(training_epochs):
            feed_dict = {xs: x_train, ys: y_train}
            sess.run(optimizer, feed_dict=feed_dict)
            print(cost.eval(feed_dict))

        weight = sess.run(W) 
        print(weight)


