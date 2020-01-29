import matplotlib.pyplot as plt
import numpy as np

class Grapher:
    def __init__(self, title="Untitled Graph", xlabel="X", ylabel="Y", xlim=None, ylim=None):
        """Set up Grapher
        Args:
            title (str): Title of graph
            xlabel (str): X-axis title
            ylabel (str): Y-axis title, .5
            xlim (2-tuple): x-min, x-max
            ylim (2-tuple): y-min, y-max
        """
        self.plt = plt
        self.ylabel = ylabel)
        self.xlabel = xlabel
        self.title = title

        # set graph limits
        self.xlim = xlim
        self.ylim = ylim
        # if not xlim is None:
        #     self.plt.xlim(xlim)
        # if not ylim is None:
        #     self.plt.ylim(ylim)
        self.x_space = np.linspace(-1, 1, 100)
        self.linear_funcs = []
        self.data_sets = []

    def show(self, save_path=None):
        """Will show a graph, possibly save it.
        Cannot manually save a graph after showing!
        Args:
            save_path (str): path to file
        """
        self._save(save_path)
        self.plt.show()

    def _save(self, save_path=None):
        """ Manually saves a graph
        Cannot be called after showing!
        Args:
            save_path (str): path to file
        """
        if save_path:
            self.plt.savefig(save_path)


    def add_function(self, func):
        """ Simple demonstration of graphing a function
        Args:
            slope (int)
            y_intercept (int)
        Returns:
            Graph of line
        """
        self.linear_funcs.append(func)

        # if save_path:
        #     plt.savefig(save_path) # don't call plt.savefig after plt.show!
        # plt.show()

    def add_points(self, x, y, labels=None, points=True, style='fivethirtyeight', legend=True):

        """ Graph results
        Args:
            x (array-like): a list of x-coordinates
            y (array-like): a list of y-coordinates
            labels (array-like): a list of integers corresponding to classes
            points (bool): True will plot points, False a line
            style: Plot style (e.g. ggplot, fivethirtyeight, classic etc.)
        Returns:
            (graph)
        """

        # prep data
        x = np.asarray(x)
        y = np.asarray(y)
        labels = np.asarray(labels)
        point_style = 'o' if points else ''


        self.data_sets.append({"x": x, "y": y, "labels": labels, "points_style": points_style, "style": style})

    def plot(self):
        
        fig, ax = self.plt.subplots(figsize=(8,6))
        self.plt.ylabel(self.ylabel)
        self.plt.xlabel(self.xlabel)
        self.plt.title(self.title)

        # # set graph limits # moved to __init__
        if not self.xlim is None:
            self.plt.xlim(self.xlim)
        if not self.ylim is None:
            self.plt.ylim(self.ylim)

        for dataset in self.data_sets:
            self.plt.style.use(style)

            plot_list = []

            # TODO: there is a plot call in this loop, check it out
            for l in np.unique(labels.astype(int)):
                idx = np.where(labels==l)
                plot_list.append(ax.plot(x[idx],y[idx], point_style, label = str(l))[0])

                    # Put legend below
        if legend:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=5, handles=plot_list,
                        facecolor = 'white', edgecolor = 'black')
        





        # set style, create plot
        # self.plt.style.use(style)
        # fig, ax = self.plt.subplots(figsize=(8,6))

        # create labels
        # self.plt.ylabel(ylabel) # moved to __init__
        # self.plt.xlabel(xlabel)
        # self.plt.title(title)

        # # set graph limits # moved to __init__
        # if not xlim is None:
        #     self.plt.xlim(xlim)
        # if not ylim is None:
        #     self.plt.ylim(ylim)




        # plt.show()

        # if save_path:
        #     fig.savefig(save_path)


if __name__ == "__main__":
    grapher = Grapher()
    y = lambda x: 5 * x**2 + 1 # equation of a parabola

    grapher.graph([-0.99, 0, 0.99], [0.5, 2, 3], [0, 1, 1])
    grapher.add_function(y)


    grapher.show()
