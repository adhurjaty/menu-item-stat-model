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

            # if (epoch + 1) % 50 == 0: 
            #     c = sess.run(cost, feed_dict = {xs: xi, ys: yi}) 
            #     print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 
      
        # Storing necessary values to be used outside the Session 
        # training_cost = sess.run(cost, feed_dict ={xs: x, ys: y}) 
        weight = sess.run(W) 


    print(train_test)


    # with tf.Session() as sess:
    # # Initiate session and initialize all vaiables
    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # #saver.restore(sess,'yahoo_dataset.ckpt')
    # for i in range(100):
    #     for j in range(X_train.shape[0]):
    #         sess.run([cost,train],feed_dict=    {xs:X_train[j,:].reshape(1,3), ys:y_train[j]})
    #         # Run cost and train with each sample
    #     c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
    #     c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    #     print('Epoch :',i,'Cost :',c_t[i])
    # pred = sess.run(output, feed_dict={xs:X_test})
    # # predict output of test data after training
    # print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    # y_test = denormalize(df_test,y_test)
    # pred = denormalize(df_test,pred)
    # #Denormalize data     
    # plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
    # plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
    # plt.legend(loc='best')
    # plt.ylabel('Stock Value')
    # plt.xlabel('Days')
    # plt.title('Stock Market Nifty')
    # plt.show()
    # if input('Save model ? [Y/N]') == 'Y':
    #     saver.save(sess,'yahoo_dataset.ckpt')
    #     print('Model Saved')

