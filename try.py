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
from PyQt5.QtCore import QTimer


# import UI file
FORM_CLASS, _ = loadUiType("Equalizerr_2.ui")  # replace "Equalizerr_2.ui" with the path to your UI file

class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.Handle_Buttons()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)

    def Handle_Buttons(self):
        self.pushButton_2.clicked.connect(self.load_signal_2)

    def load_signal_2(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Data Files (*.dat *.csv)")
        if filepath:
            _, extension = os.path.splitext(filepath)
            if not os.path.exists(filepath):
                QMessageBox.critical(self, "File Not Found", f"Could not find file at {filepath}.")
                return
            data = None
            if extension == '.dat':
                data = np.fromfile(filepath, dtype=np.int16)
            elif extension == '.csv':
                data = np.loadtxt(filepath, delimiter=',', skiprows=1)

            print(f"Loaded data: {data}")  # Print the loaded data to check if it's loaded correctly
            self.signal_2 = data  # Assign the loaded data to main_app.signal
            self.current_sample = 0  # Reset the current sample
            self.dynamic_plotting(self.signal_2, self.graphicsView)  # Plot initially
            self.timer.start(1000)  # Start the timer

    def dynamic_plotting(self, data, graph):
        print(
            f"Current sample: {self.current_sample}")  # Print the current sample to check if it's being updated correctly
        graph.clear()
        curve = graph.plot()
        curve.setData(data[:self.current_sample])
        graph.setXRange(max(0, self.current_sample - 100), self.current_sample)
        graph.setLimits(xMin=0, xMax=self.current_sample + 100, yMin=0, yMax=1.1)
        self.current_sample += 1
        if self.current_sample >= len(data):
            self.timer.stop()
        graph.showGrid(x=True, y=True)

    def update_plot(self):
        self.dynamic_plotting(self.signal_2, self.graphicsView)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()