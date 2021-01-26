from unittest import TestCase
import matplotlib.pyplot as plt

class testChannelReduction(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_boxPlotChannelReduction(self):
        channel = [4, 3, 2, 1]
        RSquare = [[1,0.98,0.92,0.98,0.98],[0.92,0.85,0.94,0.92],[ .85,0.80,0.81,0.82,0.84,0.74], [.75,.82,0.81,0.71]]

        fig = plt.figure()
        fig.suptitle('Channel reduction comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(RSquare)
        ax.set_xticklabels(channel)
        ax.set_xlabel("Number of Channels")
        ax.set_ylabel("R-Square")
        plt.savefig("R-SquareChannelCompare.eps",format="eps")
        plt.show()


