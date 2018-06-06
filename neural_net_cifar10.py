from keras.datasets import cifar10
from keras.models import Sequential, save_model
from keras.layers import MaxPooling2D, Dropout, Flatten, Conv2D, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

class neural_net_cifar10:

    def Save(self, model: Sequential):
        save_model(model=model, filepath='cifar10_model_2.hdf5')

    def run(self):
        np.set_printoptions(suppress=True)

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        #X_train = X_train[:50000]
        #X_test = X_test[:len(X_train)]
        #y_train = y_train[:len(X_train)]
        #y_test = y_test[:len(X_train)]
        #print(to_categorical(y_train))

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train / 255.0, to_categorical(y_train),
                  batch_size=5000,
                  shuffle=True,
                  epochs=100,
                  validation_data=(X_test / 255.0, to_categorical(y_test)),
                  callbacks=[EarlyStopping(min_delta=0.001, patience=3)])
        print(model.predict(X_test))
        scores = model.evaluate(X_test / 255.0, to_categorical(y_test))

        print('Loss: %.3f' % scores[0])
        print('Accuracy: %.3f' % scores[1])

        #self.Save(model)
#

neural_net_cifar10().run()