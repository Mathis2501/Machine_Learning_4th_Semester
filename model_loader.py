from keras.models import Sequential, load_model
from scipy import misc
import numpy as np
import Image
import pandas as pd
import glob as glob
import matplotlib.pyplot as plt
categories = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
np.set_printoptions(suppress=True)

class model_loader():
    model: Sequential = None
    def load(self, file):
        self.model = load_model(filepath=file)

    def run(self, test_data):
        return self.model.predict(test_data)

model = model_loader()
model.load('cifar10_model_2.hdf5')

#image_list = []
#for file in glob.glob('/images/Cars/*.png'):
#    im = misc.imread(file)
#    image_list.append(im)


image_list = map(misc.imread, glob.glob('images/Cars/*.png'))

for image in image_list:
    image = image[:,:,:3]
    image = image.reshape((1,) + image.shape)

    result = model.run(image)

    y = result[0, :]
    N = len(y)
    x = range(N)

    plt.barh(x, y, 0.75)
    plt.yticks(x, categories)
    plt.show()

