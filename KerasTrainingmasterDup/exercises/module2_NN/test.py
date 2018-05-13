import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
#
# # Step 1:Load the Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test,10)

import tensorflow as tf
import numpy as np

model = Sequential()
model.add(Dense(256,input_dim=784,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(1024,activation='relu'))
# model.add(Dense(10,activation='softmax'))
# print(model.summary())

a = np.array([[0,0,1], [1,1,1], [1000,200,300], [-3000,-0.2,0]])

k = tf.placeholder(tf.float32, [None,3])
w = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.ones([1,1]))

model = tf.nn.softmax(tf.matmul(k,w) + b, dim=0)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(model, {k:a}))