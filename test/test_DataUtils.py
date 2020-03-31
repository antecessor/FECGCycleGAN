from unittest import TestCase

from Utils.DataUtils import DataUtils


class TestDataUtils(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataUtils = DataUtils()

    def test_read_data(self):
        ecg, fecg = self.dataUtils.readData(0)
        self.assertIsNotNone(ecg)


