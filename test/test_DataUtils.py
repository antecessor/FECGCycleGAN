from unittest import TestCase
import scipy
from Utils.DataUtils import DataUtils
from ecgdetectors import Detectors

class TestDataUtils(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataUtils = DataUtils()

    def test_read_data(self):
        ecg, fecg = self.dataUtils.readData(0)
        self.assertIsNotNone(ecg)

    def test_addRepetitionDelay(self):
        ecg, fecg = self.dataUtils.readData(0)
        detectors = Detectors(200)
        # fecgDelayed = self.dataUtils.createDelayRepetition(fecg, 4, 5)
        # self.assertIsNotNone(fecgDelayed)
        r_peaks = detectors.hamilton_detector(ecg)
        pass
