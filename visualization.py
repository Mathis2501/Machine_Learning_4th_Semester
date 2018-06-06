import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import data_import as data
import pandas as pd

class visualizer:

    def visualize(self):

        # legends
        #region

        red_patch = mpatches.Patch(color='red', label='setosa')
        green_patch = mpatches.Patch(color='green', label='versicolor')
        blue_patch = mpatches.Patch(color='blue', label='virginica')

        #endregion

        # sepalLength + sepalWidth
        #region

        # setosa
        self.appendgraph('setosa', 'sepalLength', 'sepalWidth', 'rd')

        # versicolor
        self.appendgraph('versicolor', 'sepalLength', 'sepalWidth', 'gd')

        # virginica
        self.appendgraph('virginica', 'sepalLength', 'sepalWidth', 'bd')

        plt.legend(handles=[red_patch, green_patch, blue_patch])
        plt.show()
        #endregion

        # sepalLength + petalLength
        #region

        # setosa
        self.appendgraph('setosa', 'sepalLength', 'petalLength', 'rd')

        # versicolor
        self.appendgraph('versicolor', 'sepalLength', 'petalLength', 'gd')

        # virginica
        self.appendgraph('virginica', 'sepalLength', 'petalLength', 'bd')

        plt.legend(handles=[red_patch, green_patch, blue_patch])
        plt.show()
        #endregion

        # sepalLength + petalWidth
        # region

        # setosa
        self.appendgraph('setosa', 'sepalLength', 'petalWidth', 'rd')

        # versicolor
        self.appendgraph('versicolor', 'sepalLength', 'petalWidth', 'gd')

        # virginica
        self.appendgraph('virginica', 'sepalLength', 'petalWidth', 'bd')

        plt.legend(handles=[red_patch, green_patch, blue_patch])
        plt.show()

        # endregion

        # sepalWidth + petalLength
        # region

        # setosa
        self.appendgraph('setosa', 'sepalWidth', 'petalLength', 'rd')

        # versicolor
        self.appendgraph('versicolor', 'sepalWidth', 'petalLength', 'gd')

        # virginica
        self.appendgraph('virginica', 'sepalWidth', 'petalLength', 'bd')

        plt.legend(handles=[red_patch, green_patch, blue_patch])
        plt.show()

        # endregion

        # sepalWidth + petalWidth
        # region

        # setosa
        self.appendgraph('setosa', 'sepalWidth', 'petalWidth', 'rd')

        # versicolor
        self.appendgraph('versicolor', 'sepalWidth', 'petalWidth', 'gd')

        # virginica
        self.appendgraph('virginica', 'sepalWidth', 'petalWidth', 'bd')

        plt.legend(handles=[red_patch, green_patch, blue_patch])
        plt.show()

        # endregion

        # petalWidth + petalLength
        # region

        # setosa
        self.appendgraph('setosa', 'petalWidth', 'petalLength', 'rd')

        # versicolor
        self.appendgraph('versicolor', 'petalWidth', 'petalLength', 'gd')

        # virginica
        self.appendgraph('virginica', 'petalWidth', 'petalLength', 'bd')

        plt.legend(handles=[red_patch, green_patch, blue_patch])
        plt.show()

        # endregion

    def appendgraph(self, species, X, Y, color):
        xs = self.getrows(species, X)
        ys = self.getrows(species, Y)
        plt.plot(xs, ys, color)
        plt.xlabel(X)
        plt.ylabel(Y)

    def getrows(self, X, Y):
        return df.loc[df['species'] == X][Y]

df = data.data().load()
visualizer().visualize()