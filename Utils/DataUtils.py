import pyedflib
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale


class DataUtils:

    def __init__(self) -> None:
        super().__init__()
        self.fileNames = ["r01.edf", "r04.edf", "r07.edf", "r08.edf", "r10.edf"]

    def readData(self, sigNum, path="E:\Workspaces\ECG2FetalCycleGAN\\abdominal-and-direct-fetal-ecg-database-1.0.0\\"):
        file_name = path + self.fileNames[sigNum]
        f = pyedflib.EdfReader(file_name)
        n = f.signals_in_file
        # signal_labels = f.getSignalLabels()
        abdECG = np.zeros((n - 1, f.getNSamples()[0]))
        fetalECG = np.zeros((1, f.getNSamples()[0]))
        fetalECG[0, :] = f.readSignal(0)
        fetalECG[0, :] = scale(self.butter_bandpass_filter(fetalECG, 10, 50, 1000), axis=1)
        for i in np.arange(1, n):
            abdECG[i - 1, :] = f.readSignal(i)
        abdECG = scale(self.butter_bandpass_filter(abdECG, 10, 50, 1000), axis=1)

        abdECG = signal.resample(abdECG, int(abdECG.shape[1] / 5), axis=1)
        fetalECG = signal.resample(fetalECG, int(fetalECG.shape[1] / 5), axis=1)
        return abdECG, fetalECG

    def windowingSig(self, sig1, sig2, windowSize=15):
        signalLen = sig2.shape[1]
        signalsWindow1 = [sig1[:, int(i):int(i + windowSize)].transpose() for i in range(0, signalLen - windowSize, windowSize)]
        signalsWindow2 = [sig2[:, int(i):int(i + windowSize)].transpose() for i in range(0, signalLen - windowSize, windowSize)]

        return signalsWindow1, signalsWindow2

    def calculateICA(self, sdSig, labels, component=7):
        ica = FastICA(n_components=component, max_iter=1000)
        icaRes = []
        labelNew = []
        for index, sig in enumerate(sdSig):
            try:
                if labels[index].shape[0] == component and labels[index].shape[1] == sdSig[0].shape[1]:
                    icaRes.append(np.array(ica.fit_transform(sig.transpose())).transpose())
                    labelNew.append(labels[index])
            except:
                pass
        return np.array(icaRes), np.array(labelNew)

    def createDelayRepetition(self, signal, numberDelay=4, delay=10):
        signal = np.repeat(signal, numberDelay, axis=0)
        for row in range(1, signal.shape[0]):
            signal[row, :] = np.roll(signal[row, :], shift=delay * row)
        return signal

    def __butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3, axis=1):
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=axis)
        return y
