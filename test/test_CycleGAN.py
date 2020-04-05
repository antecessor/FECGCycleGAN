from unittest import TestCase
import numpy as np

from Utils.TrainUtils import TrainUtils
from deeplearning.CycleGAN import CycleGAN


class TestCycleGAN(TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.trainUtils = TrainUtils()

    def test_trainSignal(self):
        ecgWindows, fecgWindows = self.trainUtils.prepareData(delay=2)
        X_train, X_test, Y_train, Y_test = self.trainUtils.trainTestSplit(ecgWindows, fecgWindows, 0.75)

        X_train = np.reshape(X_train, [-1, X_train.shape[1], X_train.shape[2]])
        # X_test = np.reshape(X_test, [-1, X_test.shape[1], X_test.shape[2], 1])
        Y_train = np.reshape(Y_train, [-1, Y_train.shape[1], Y_train.shape[2]])
        # y_test = np.reshape(Y_test, [-1, Y_test.shape[1], Y_test.shape[2], 1])

        cycleGAN = CycleGAN(Y_train.shape[1], Y_train.shape[2])
        cycleGAN.train(x_train=X_train, y_train=Y_train, epochs=1)
