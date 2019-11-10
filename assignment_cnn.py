import numpy as np
# my class
import cnn.model
import cnn.tools as tool
from cnn.model import ConvolutionalNetwork as network
from cnn.datas import DataManager as data_manager

# data stack for plotting.
train_acc = []
train_err = []
valid_acc = []
valid_err = []

# config list.
CONFIG = {'total_epoch': 3000,
          'batch_size': 60000,
          'learning_rate': 1e-4,
          'train_dataset_size': 60000,
          'test_dataset_size': 10000
          }

# define network fcn.
network_model = network(configure=CONFIG, filter_size=3, d1=16, d2=32, h1=64)
dataManager = data_manager()

network_model.train(dataManager.test_X[0:2], dataManager.test_y[0:2])

# using mini-batch
for i in range(network_model.TOTAL_EPOCH):
    print("============== EPOCH {} START ==============".format(i + 1))
    for j in range(dataManager.train_dataset_size // network_model.BATCH_SIZE):
        # print("-------------- batch {} training...".format(j))

        # load batch data.
        batch_x, batch_y = dataManager.next_batch(network_model.BATCH_SIZE)

        # train model.
        network_model.train(batch_x, batch_y)

        if j % 10 == 0:
            # save data.
            # calculate accuracy and loss
            output_train = network_model.predict(dataManager.train_X)
            accuracy_train, loss_train = network_model.getAccuracyAndLoss(output_train, dataManager.train_y)

            # add data to stack
            train_acc.append(accuracy_train)
            train_err.append(loss_train)

            # calculate test dataset.
            output_test = network_model.predict(dataManager.test_X)
            accuracy_test, loss_test = network_model.getAccuracyAndLoss(output_test, dataManager.test_y)

            # add data to stack
            valid_acc.append(accuracy_test)
            valid_err.append(loss_test)
    print("============== EPOCH {} END ================".format(i + 1))

    # shake data when epoch ended.
    # dataManager.shake_data()

    # calculate accuracy and loss
    output_train = network_model.predict(dataManager.train_X)
    accuracy_train, loss_train = network_model.getAccuracyAndLoss(output_train, dataManager.train_y)

    # calculate test dataset.
    output_test = network_model.predict(dataManager.test_X)
    accuracy_test, loss_test = network_model.getAccuracyAndLoss(output_test, dataManager.test_y)

    print("train accuracy : {:.4}; loss : {:.3}, test accuracy : {:.3}; loss : {:.3}".format(accuracy_train, loss_train,
                                                                                             accuracy_test, loss_test))

    if i % 10 == 0 or i == network_model.TOTAL_EPOCH - 1:
        # draw graph.
        tool.plotting(CONFIG['learning_rate'], train_acc, train_err, valid_acc, valid_err)
