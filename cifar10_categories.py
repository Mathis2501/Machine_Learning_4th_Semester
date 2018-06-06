from keras.datasets import cifar10
from keras.models import Sequential, save_model
from keras.layers import MaxPooling2D, Dropout, Flatten, Conv2D, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from PIL import Image

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

checks = []
count = 0
for x in X_train:
    if checks.__contains__(y_train[count]):
        continue
    if(len(checks) == 10):
        break
    checks.append(y_train[count])
    image = Image.fromarray(x)
    image.save('converted/cat'+str(y_train[count])+'.png')
    count+=1
