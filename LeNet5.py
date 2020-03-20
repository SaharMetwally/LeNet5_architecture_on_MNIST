import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Softmax
import numpy as np
from numpy import expand_dims

'''About DATA'''
#  load our dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# convertion to float 32
x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
# Normalize value to [0, 1]
# normalization helps network to learn better
x_train /= 255
x_test /= 255
#  convert dataset to 4D actually put number of channels equal to one
x_train = expand_dims(x_train, 3)
x_test = expand_dims(x_test, 3)
#convert lables in both training and testing to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)




'''About the model architecture <LeNet-5>'''
'''
model = Sequential()
# layer1: conv layer
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation="tanh", input_shape=(28, 28, 1), padding="same"))
# layer2: pooling layer "average pooling"
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid"))
# layer3: conv layer
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation="tanh", padding="valid"))
# layer4: pooling layer "average pooling"
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
# layer5: is a fully connected convolutional layer  "conv layer"
model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation="tanh", padding="valid"))
# faltten the output of layer5 so we can connect layer5's output with fully connected layer
model.add(Flatten())
# layer6: fully connected layer
model.add(Dense(84, activation="tanh"))
# output layer with softmax
model.add(Dense(10, activation="softmax"))
# compile the model
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
# fit model
hist = model.fit(x=x_train, y=y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), verbose=1)
# model evaluation
validation_loss, validation_accuracy = model.evaluate(x_test, y_test)
print("validation_loss : ", validation_loss)
print("validation_accuracy : ", validation_accuracy)
# save the model
model.save("LeNet-5_model")
'''


new_model = tf.keras.models.load_model("LeNet-5_model")
predictions = new_model.predict(x_test)
print(np.argmax(predictions[0]))
print(np.argmax(predictions[1]))
print(np.argmax(predictions[2]))

print(np.argmax(y_test[0]))
print(np.argmax(y_test[1]))
print(np.argmax(y_test[2]))




# '''method to show a sequence of images not just one :D'''
def rendering(list_of_images):
    count = 0
    for i in list_of_images:
        plt.imshow(list_of_images[count])
        plt.show()
        i += 1
        count += 1


#print(y_train[:3])
# rendering(x_train[:3])
