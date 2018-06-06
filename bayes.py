from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_predict
import pandas as pd
import numpy as np
from data_import import data
import matplotlib.pyplot as plt
import seaborn as sns

class knn:
    def run(self, dataset: pd.DataFrame()):
        np.set_printoptions(suppress=True)
        model = GaussianNB()
        y = dataset['species'].astype('category')
        X = dataset.drop(['species'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        df = pd.DataFrame(y_pred)
        df.columns=["setosa", "versicolor", 'virginica']
        df = df.round(5)
        print(df.head())
        print(df)

        dataset = dataset.drop(['species'], axis=1)
        dataset.apply(lambda x: (x - np.mean(x))/(np.max(x)-np.min(x)))
        sns.heatmap(dataset.corr())
        plt.yticks(rotation=45)
        plt.show()

data = data()
knn().run(dataset=data.load())
