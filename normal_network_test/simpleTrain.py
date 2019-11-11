import numpy as np
import tensorflow as tf
import os
# from mnistDatas import DataManager as dm
import matplotlib.pyplot as plt
from normal_network_test.cifar10Datas import DataManager as dm


def plotting(title, train_acc, train_err, valdiate_acc, validate_err):
    train_acc = np.array(train_acc)
    train_err = np.array(train_err)
    valdiate_acc = np.array(valdiate_acc)
    validate_err = np.array(validate_err)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.set_xlabel("accuracy")
    ax1.plot(train_acc, "r", label='train')
    ax1.plot(valdiate_acc, "g", label='test')
    ax1.legend(loc='lower right')

    ax2.set_xlabel("loss")
    ax2.plot(train_err, "r", label='train')
    ax2.plot(validate_err, "g", label='test')
    ax2.legend(loc='upper right')

    fig.suptitle('learning_rate :{}'.format(title))
    plt.show()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

dataManager = dm()

train_acc_array = []
train_loss_array = []
test_acc_array = []
test_loss_array = []


def getAccuracyAndLoss(output, label):
    accuracy = np.mean(np.equal(np.argmax(output, axis=1),
                                np.argmax(label, axis=1)))
    loss = -np.mean(label * np.log(output + 1e-6))

    return accuracy, loss


def conv_layer(input, outdim):
    w = tf.Variable(tf.random_normal(shape=[3, 3, input.get_shape().as_list()[-1], outdim]))
    b = tf.Variable(tf.random_normal(shape=[outdim]))

    out = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME") + b

    # activation.
    out = tf.nn.relu(out)

    return out


def layer(input, outdim, activation=True):
    # already flattend
    if len(input.get_shape().as_list()) == 2:
        w = tf.Variable(tf.random_normal(shape=[input.get_shape().as_list()[-1], outdim]))
        b = tf.Variable(tf.random_normal(shape=[outdim]))

        out = tf.matmul(input, w) + b
        # activation.
        if activation:
            out = tf.nn.relu(out)
    else:
        # NOTICE:flatten
        out = tf.reshape(input,
                         shape=[-1, input.get_shape().as_list()[1] * input.get_shape().as_list()[2] *
                                input.get_shape().as_list()[3]])

        w = tf.Variable(tf.random_normal(shape=[out.get_shape().as_list()[-1], outdim]))
        b = tf.Variable(tf.random_normal(shape=[outdim]))

        out = tf.matmul(out, w) + b
        # activation.
        if activation:
            out = tf.nn.relu(out)
    return out


# x_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_output = tf.placeholder(tf.float32, shape=[None, 10])

# model.
model = conv_layer(x_input, 16)
model = tf.nn.max_pool(model, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
model = conv_layer(model, 32)
model = tf.nn.max_pool(model, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
model = layer(model, 64)
output = layer(model, 10, False)
softmax_output = tf.nn.softmax(output)

learning_rate = 1e-3
epochs = 90

loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_output)

optimizing = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for epoch in range(epochs):
        print("============== EPOCH {} START ==============".format(epoch + 1))
        for iter in range(50):
            # NOTICE: supervised batch.
            batch_x, batch_y = dataManager.next_batch(1000, batchType=dataManager.SUPERVISED_BATCH)
            _ = sess.run(optimizing, feed_dict={x_input: batch_x, y_output: batch_y})

            if iter % 10 == 0:
                y_train_predict = sess.run(softmax_output, feed_dict={x_input: dataManager.train_X})
                y_test_predict = sess.run(softmax_output, feed_dict={x_input: dataManager.test_X})

                train_acc, train_loss = getAccuracyAndLoss(y_train_predict, dataManager.train_y)
                test_acc, test_loss = getAccuracyAndLoss(y_test_predict, dataManager.test_y)

                train_acc_array.append(train_acc)
                train_loss_array.append(train_loss)
                test_acc_array.append(test_acc)
                test_loss_array.append(test_loss)

        print("============== EPOCH {} END ================".format(epoch + 1))

        output_train = sess.run(softmax_output, feed_dict={x_input: dataManager.train_X})
        accuracy_train, loss_train = getAccuracyAndLoss(output_train, dataManager.train_y)

        # calculate test dataset.
        output_test = sess.run(softmax_output, feed_dict={x_input: dataManager.test_X})
        accuracy_test, loss_test = getAccuracyAndLoss(output_test, dataManager.test_y)

        print("train accuracy : {:.4}; loss : {:.3}, test accuracy : {:.3}; loss : {:.3}".format(accuracy_train,
                                                                                                 loss_train,
                                                                                                 accuracy_test,
                                                                                                 loss_test))

        plotting("", train_acc_array, train_loss_array, test_acc_array, test_loss_array)
