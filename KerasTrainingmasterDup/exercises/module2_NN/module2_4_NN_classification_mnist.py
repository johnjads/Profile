# Module 2 Neural Network
# NN Model on MNIST dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# Step 1:Load the Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, 2048)
y_test = keras.utils.to_categorical(y_test,2048)



################################################


# Step 2: Build the Model
model = Sequential()
model.add(Dense(256,input_dim=784,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(2048,activation='relu'))
print(model.summary())



# Step 3: Compile the Model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Step 4: Train the Model
model.fit(X_train,y_train,epochs=5,batch_size=100)

# Step 5: Evaluate the Model
loss,accuracy = model.evaluate(X_test,y_test)
print("Loss = ",loss)
print("Accuracy ",accuracy)

# Step 6: Save the Model
model.save('./models/mnist_nn2.h5')
