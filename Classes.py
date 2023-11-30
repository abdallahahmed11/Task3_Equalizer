import numpy as np 
from scipy.signal.windows import get_window
class signalProcessing:
    def __init__(self) -> None:
        pass
    def setData(self,data):
        self.data = data
    def getData(self):
        return self.data
    @staticmethod
    def fourierTransform(data,len,sampleRate):
        fft_data = np.fft.rfft(data)/len
        return fft_data
    @staticmethod
    def get_frequencies(dataLength,samplingRate):
        return np.fft.rfftfreq(dataLength,1/samplingRate)
    @staticmethod
    def inverseFourier(newComplexData,len):
        return np.fft.irfft(newComplexData*len)
    @staticmethod
    def calcMagnitudePhase(complexData):
        magnitude = np.abs(complexData)*2 
        phase = np.angle(complexData)
        return magnitude,phase
    @staticmethod
    def calcEqualizedSignalComplex(equalizedMagnitude,phase):
        return (equalizedMagnitude * np.exp(1j*phase))
    @staticmethod
    def applyWindow(amplitude,window):
        newAmlitude = amplitude * window
        return newAmlitude
    
    

class musicalInstruments:
    def __init__(self,originalData,originalTime,sampleRate,chosenWindow):
        self.originalData = originalData
        self.originalTime = originalTime
        self.sampleRate = sampleRate
        self.datalength = len(self.originalData)
        self.transformedDataComplex = signalProcessing.fourierTransform(self.originalData,self.datalength,self.sampleRate)
        self.magnitude,self.phase = signalProcessing.calcMagnitudePhase(self.transformedDataComplex)
        self.frequency = signalProcessing.get_freq(self.datalength,self.sampleRate)
        self.window = chosenWindow
    

class Window:
    def __init__(self, window_type, length, **kwargs):
        self.window_type = window_type
        self.length = length
        self.parameters = kwargs
        self.window = self._create_window()

    def _create_window(self):
        try:
            window_function = get_window(self.window_type, self.length, **self.parameters)
        except ValueError as e:
            raise ValueError(f"Error creating window: {e}")
        return window_function