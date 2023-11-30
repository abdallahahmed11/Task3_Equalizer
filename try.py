import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.uic import loadUiType
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from scipy import signal
import pyqtgraph as pg
import os
import scipy.signal

# import UI file
FORM_CLASS, _ = loadUiType("Equalizerr_2.ui")  # replace "Equalizerr_2.ui" with the path to your UI file

class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.Handle_Buttons()


    def Handle_Buttons(self):
        # self.pushButton.clicked.connect(self.load_and_plot)
        self.pushButton_2.clicked.connect(self.load_signal_2)

    # def load_signal(self):
    #     # create a simple sine wave as an example signal
    #     fs = 10e3
    #     N = 1e5
    #     amp = 2 * np.sqrt(2)
    #     freq = 1234.0
    #     noise_power = 0.01 * fs / 2
    #     time = np.arange(N) / fs
    #     self.SignalArray = amp*np.sin(2*np.pi*freq*time)
    #
    # def load_and_plot(self):
    #     self.load_signal_2()
    #     self.CreateWindowFigure()
    #     self.apply_windowing(100)

    def load_signal_2(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Data Files (*.dat *.csv)")
        if filepath:
            _, extension = os.path.splitext(filepath)
            if not os.path.exists(filepath):
                QMessageBox.critical(self, "File Not Found", f"Could not find file at {filepath}.")
                return
            data = None
            if extension == '.dat':
                # Read the .dat file as 16-bit integers
                data = np.fromfile(filepath, dtype=np.int16)
            elif extension == '.csv':
                data = np.loadtxt(filepath, delimiter=',', skiprows=1)

            self.signal_2 = data  # Assign the loaded data to main_app.signal
            window=signal.windows.hann(len(self.signal_2))
            window_2=window*0


            self.graphicsView.addItem(pg.PlotDataItem(window_2))



    # def apply_windowing(self, window_size):
    #     window = scipy.signal.windows.boxcar(window_size)
    #     self.axes_window.plot(window)
    #     self.Window.draw()
    #
    # def CreateWindowFigure(self):
    #     self.figure = plt.figure()
    #     self.figure.patch.set_facecolor('black')
    #     self.axes_window = self.figure.add_subplot()
    #     self.Window = Canvas(self.figure)
    #     self.verticalLayout.addWidget(self.Window)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()