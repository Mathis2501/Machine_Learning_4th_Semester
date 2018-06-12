from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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
            cv_scores = cross_val_score(model, X, y, cv=5)
            xs.append(i)
            ys.append(np.mean(cv_scores))
            print("knncv: k="+ str(i) + " = " + str(np.mean(cv_scores)))
        xs = np.array(xs)
        ys = np.array(ys)
        plt.plot(xs, ys, 'g-')

        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        plt.plot(xs, p(xs), 'r-')
        plt.ylim(ymax=1, ymin=0.5)
        plt.ylabel("Accuracy")
        plt.xlabel("K")
        plt.title("KNN CV")
        plt.show()

data = data()
knn().run(dataset=data.load())
