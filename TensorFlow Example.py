import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import cifar10 #mnist
from matplotlib import pyplot as plt
from keras.models import load_model

#Data set creation and conditioning
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_train.shape)
plt.imshow(x_train[0,:,:,:])

x_train /= 255.
x_test /= 255.

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape((-1, 32, 32, 3))
x_test = x_test.reshape((-1, 32, 32, 3))

#if model exists, load it, otherwise train it
try:
    model = load_model('mnist_weights.h5') #mnist_weights.h5
    model.summary()

    score = model.evaluate(x_test, y_test)
    
    from random import randint
    r_ind = randint(0, x_test.shape[0])
    r_pic = x_test[r_ind,:,:,:]
    plt.imshow(r_pic.reshape((32,32)))
    
    val = model.predict(r_pic.reshape(-1,32,32,3))
    ans = np.argmax(val)
    print(ans)
    
except:
    #Model design
    model = Sequential()
    
    #Input Layer
    model.add(Conv2D(32, (3,3), activation = 'relu', input_shape=(32,32,3)))
    
    #Convolutional layer 1
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    #Convolutional layer 2
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.3))
    
    #Convolutional layer 3
    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.3))
    
    #flatten layer 4
    model.add(Flatten())
    
    #Dense NN Layer 5
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.3))
    
    #Dense NN Layer 6
    model.add(Dense(10, activation='relu'))
    
    #Output later
    model.add(Dense(10, activation='softmax')) #output layer
    
    model.compile(loss='categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    model.summary()
    
    model.fit(x_train, 
              y_train, 
              batch_size = 32,
              epochs = 1,
              verbose = 1,
              shuffle = True,
              validation_data = (x_test, y_test))
    
    model.save('mnist_weights.h5')