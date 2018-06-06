import keras
from keras.layers import Dense
from keras.models import Sequential
import data_import as data
from sklearn.model_selection import train_test_split
import numpy as np

class neural_net:
    def run(self):
        np.set_printoptions(suppress=True)
        df = data.data().load()
        df['setosa'] = np.zeros(150)
        df['versicolor'] = np.zeros(150)
        df['virginica'] = np.zeros(150)
        for i in range(len(df)):
            if df.loc[i, 'species'] == 'setosa':
                df.loc[i, 'setosa'] = 1
            elif df.loc[i, 'species'] == 'versicolor':
                df.loc[i, 'versicolor'] = 1
            else:
                df.loc[i, 'virginica'] = 1
        print(df.to_string())
        y = df[['setosa', 'versicolor', 'virginica']]
        X = df.drop(['species'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        model = Sequential()
        n_cols = X.shape[1]
        model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, steps_per_epoch=30)
        #print(model)
        print(model.predict(X_test))

neural_net().run()