import pandas as pd


class data:
    def load(self):
        df = pd.read_json('iris.json')
        print(df.info())
        return df

