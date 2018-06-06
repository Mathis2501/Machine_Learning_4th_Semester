from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from data_import import data
import matplotlib.pyplot as plt

class knn:
    def run(self, dataset: pd.DataFrame()):
        xs = []
        ys = []
        for i in range(1,50,1):
            model = KNeighborsClassifier(n_neighbors=i)
            y = dataset['species'].astype('category')
            X = dataset.drop(['species'], axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print("knn: k="+ str(i) + " = " + str(score))
            xs.append(i)
            ys.append(score)

        xs = np.array(xs)
        ys = np.array(ys)
        plt.plot(xs, ys, 'g-')

        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        plt.plot(xs, p(xs), 'r-')

        plt.ylim(ymax=1, ymin=0.5)
        plt.show()

data = data()
knn().run(dataset=data.load())
