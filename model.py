import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

device_name = "/device:GPU:0"
# device_name = "/cpu:0"


# Hyperparameters
epochs = 50
batch_size = 32
learn_rate = 0.0001
regularization = 0.1
num_classes = 10


# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def model_forward(x, weights, biases):
    # First conv2d layer
    x = tf.nn.conv2d(x, filter=weights['conv2d_filter1'], strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, biases['bias1'])

    # First ReLU layer
    x = tf.nn.relu(x)

    # Batch normalization layer
    mean, variance = tf.nn.moments(x, axes=[0])
    x = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-8)

    # 2x2 max pooling layer
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print(x.shape)

    # Flatten Layer
    x = tf.reshape(x, [-1, weights['fc1_weight'].get_shape().as_list()[0]])

    # Fully connected layer 1 (784 output units)
    x = tf.add(tf.matmul(x, weights['fc1_weight']), biases['fc1_bias'])

    # Dropout layer
    x = tf.layers.dropout(x, rate=0.9)

    # Second ReLU layer
    x = tf.nn.relu(x)

    # Fully connected layer 2 (10 output units)
    x = tf.add(tf.matmul(x, weights['out_weight']), biases['out_bias'])

    # Softmax to formulate the output
    res = tf.nn.softmax(x)

    return res


if __name__ == '__main__':

    # with tf.device(device_name):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    # reshape to 4D tensors for input into CNN
    trainData = np.expand_dims(trainData, 3)
    validData = np.expand_dims(validData, 3)
    testData = np.expand_dims(testData, 3)

    x = tf.placeholder("float", [None, 28, 28, 1])
    y = tf.placeholder("float", [None, num_classes])

    weights = {
        'conv2d_filter1': tf.get_variable('W1', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
        'fc1_weight': tf.get_variable('W2', shape=(32 * 14 * 14, 784), initializer=tf.contrib.layers.xavier_initializer()),
        'out_weight': tf.get_variable('W6', shape=(784, num_classes), initializer=tf.contrib.layers.xavier_initializer())
    }

    biases = {
        'bias1': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'fc1_bias': tf.get_variable('B2', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
        'out_bias': tf.get_variable('B3', shape=(num_classes), initializer=tf.contrib.layers.xavier_initializer())
    }

    # Convert to one-hot
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

    pred = model_forward(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)) + \
        tf.multiply(
               tf.reduce_sum(tf.square(weights['conv2d_filter1'])) + tf.reduce_sum(tf.square(weights['fc1_weight']))
               + tf.reduce_sum(tf.square(weights['out_weight'])), regularization / 2)

    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # calculate accuracy across all the given images and average them out.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    test_losses = []
    test_accuracies = []

    batches = []
    counter = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        num_batches = int(trainData.shape[0] / batch_size)
        for epoch in range(epochs):

            shuffle(trainData, newtrain)

            for i in range(num_batches):
                train_Batch = trainData[i * batch_size: min((i+1)*batch_size, len(trainData))]
                train_Target_Batch = newtrain[i * batch_size: min((i+1)*batch_size, len(trainData))]

                sess.run(optimizer, feed_dict={x: train_Batch, y: train_Target_Batch})

                train_loss, train_acc = sess.run([cost, accuracy], feed_dict={x: train_Batch, y: train_Target_Batch})
                # print(train_loss, train_acc)

            batches.append(counter)
            counter += 1

            valid_loss, valid_acc = sess.run([cost, accuracy], feed_dict={x: validData, y: newvalid})
            test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: testData, y: newtest})

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            print("Epoch: {} | Training loss: {:.5f} Training Accuracy: {:.5f} | Validation Loss: {:.5f} Validation Accuracy: {:.5f} | Test Loss: {:.5f} Test Accuracy: {:.5f}  "
                  .format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc))

            # print("Epoch: {} | Training loss: {:.5f} Training Accuracy: {:.5f}  "
            #      .format(epoch + 1, train_loss, train_acc))

    # Plotting
    batches = np.array(batches)
    fig = plt.figure()
    plt.plot(batches, train_losses, label='Training Loss')
    plt.plot(batches, valid_losses, label='Test Loss')
    plt.plot(batches, test_losses, label='Validation Loss')
    ax = fig.add_subplot(1, 1, 1)
    plt.title('alpha={}, epsilon={}, batch_size={}'.format(learn_rate, epochs, batch_size))
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    fig.savefig('plots/p34c_sgd_loss_eps_{}.png'.format(epochs))

    fig = plt.figure()
    plt.plot(batches, train_accuracies, label='Training Accuracy')
    plt.plot(batches, valid_accuracies, label='Validation Accuracy')
    plt.plot(batches, test_accuracies, label='Testing Accuracy')
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlim(-200, relevantepoch.shape[0] + 10)
    plt.title('alpha={}, epsilon={}, batch_size={}'.format(learn_rate, epochs, batch_size))
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    fig.savefig('plots/p34c_sgd_acc_eps_{}.png'.format(epochs))
