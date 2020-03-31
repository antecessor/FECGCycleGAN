import pyedflib
import numpy as np


class DataUtils:

    def __init__(self) -> None:
        super().__init__()
        self.fileNames = ["r01.edf", "r04.edf", "r07.edf", "r08.edf", "r10edf"]

    def readData(self, sigNum, path="E:\Workspaces\ECG2FetalCycleGAN\\abdominal-and-direct-fetal-ecg-database-1.0.0\\"):
        file_name = path + self.fileNames[sigNum]
        f = pyedflib.EdfReader(file_name)
        n = f.signals_in_file
        # signal_labels = f.getSignalLabels()
        abdECG = np.zeros((n, f.getNSamples()[0]))
        fetalECG = np.zeros((1, f.getNSamples()[0]))
        for i in np.arange(1, n):
            abdECG[i, :] = f.readSignal(i)
        fetalECG[0, :] = f.readsignal(0)
        return abdECG, fetalECG

    def windowingSig(sig, labels, windowSize=15):
        signalLen = labels.shape[1]
        if len(labels.shape) == 1:
            labelsWindow = [labels[int(i):int(i + windowSize)].transpose() for i in range(0, signalLen - 1, windowSize)]
        else:
            labelsWindow = [labels[:, int(i):int(i + windowSize)].transpose() for i in range(0, signalLen - windowSize, windowSize)]
        signalsWindow = [sig[:, int(i):int(i + windowSize)].transpose() for i in range(0, signalLen - windowSize, windowSize)]

        return signalsWindow, labelsWindow
