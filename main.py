from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt5.uic import loadUiType
import os
from os import path
import sys
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph as pg
# import the function from utils.py
from FunctionsOOP import SignalProcessor, SpectrogramPlotter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib.pyplot as plt
import scipy.signal

# import UI file
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "Equalizer-All.ui"))


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)

        # Create an instance of SignalProcessor
        self.signal_processor = SignalProcessor(self)

        # Add your custom logic and functionality here
        self.Handle_Buttons()
        self.intializer()



    def intializer(self):
        self.tabWidget.setCurrentIndex(0)
        self.window_type='Rectangle'
        self.verticalSlider_1.setRange(0, 10)
        self.verticalSlider_2.setRange(0, 10)
        self.verticalSlider_3.setRange(0, 10)
        self.verticalSlider_4.setRange(0, 10)
        self.verticalSlider_5.setRange(0, 10)
        self.verticalSlider_6.setRange(0, 10)
        self.verticalSlider_7.setRange(0, 10)
        self.verticalSlider_8.setRange(0, 10)
        self.verticalSlider_9.setRange(0, 10)
        self.verticalSlider_10.setRange(0, 10)
        self.verticalSlider_1.setValue(1)
        self.verticalSlider_2.setValue(1)
        self.verticalSlider_3.setValue(1)
        self.verticalSlider_4.setValue(1)
        self.verticalSlider_5.setValue(1)
        self.verticalSlider_6.setValue(1)
        self.verticalSlider_7.setValue(1)
        self.verticalSlider_8.setValue(1)
        self.verticalSlider_9.setValue(1)

        # Create an instance of SpectrogramPlotter




    def Handle_Buttons(self):
        self.tabWidget.currentChanged.connect(self.tab_changed_handler)


    def apply_equalizer_handler(self, freqGraph , outputTimeGraph):
        freq_ranges, magnitude, phases, freqs, time  = self.signal_processor.get_freq_components(self.signal_processor.signal )
        self.spectrogram_plotter.plot_spectro(magnitude, 1)
        self.signal_processor.apply_equalizer_uniform(freq_ranges, magnitude, phases, freqs, time ,freqGraph, outputTimeGraph)
        self.spectrogram_plotter_2.plot_spectro(magnitude, 1)

    def apply_sound_equalizer_handler(self, freqGraph, outputTimeGraph):
        freq_ranges, magnitude, phases, freqs, time = self.signal_processor.get_freq_components_sound(
            self.signal_processor.signal)

        self.signal_processor.apply_equalizer_sound(freq_ranges, magnitude, phases, freqs, time, freqGraph,
                                                    outputTimeGraph)

    def tab_changed_handler(self, index):
        if index == 0:
            print("First tab clicked")
            # self.pushButton.clicked.connect(self.signal_processor.load_signal)
            self.pushButton.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.graphicsView))
            self.spectrogram_plotter = SpectrogramPlotter(self.verticalLayout_16)
            self.spectrogram_plotter_2 = SpectrogramPlotter(self.verticalLayout_17)
            self.pushButton.clicked.connect(self.signal_processor.default_graph_drawing)
            # self.pushButton_2.clicked.connect(self.apply_equalizer_handler)
            self.pushButton_2.clicked.connect(lambda:self.apply_equalizer_handler(freqGraph=self.graphicsView_3 , outputTimeGraph=self.graphicsView_2))
            self.signal_processor.clear_modes([2, 3, 4])
            self.comboBox_mode1.currentIndexChanged.connect(
                lambda: self.signal_processor.on_window_type_changed2(index=0, comboBox=self.comboBox_mode1))

            self.graphicsView.setXLink(self.graphicsView_2)
            self.graphicsView.setYLink(self.graphicsView_2)
            self.pushButton_4.clicked.connect(lambda :self.signal_processor.zoomIn(graph=self.graphicsView))
            self.pushButton_5.clicked.connect(lambda :self.signal_processor.zoomOut(graph=self.graphicsView))
            self.pushButton_6.clicked.connect(lambda :self.signal_processor.fitScreen(graph=self.graphicsView))




        elif index == 1:
            print("Second tab clicked")
            # self.pushButton_57.clicked.connect(self.signal_processor.load_signal)
            self.pushButton_57.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.graphicsView_56))
            self.spectrogram_plotter = SpectrogramPlotter(self.verticalLayout_22)
            self.spectrogram_plotter_2 = SpectrogramPlotter(self.verticalLayout_23)
            self.pushButton_57.clicked.connect(self.signal_processor.default_graph_drawing)

            self.pushButton_7.clicked.connect(lambda:self.apply_sound_equalizer_handler(freqGraph=self.graphicsView_60, outputTimeGraph= self.graphicsView_58))
            self.signal_processor.clear_modes([1, 3, 4])
            self.comboBox_mode2.currentIndexChanged.connect(
                lambda: self.signal_processor.on_window_type_changed2(index=0, comboBox=self.comboBox_mode2))
            self.graphicsView_56.setXLink(self.graphicsView_58)
            self.graphicsView_56.setYLink(self.graphicsView_58)
            self.pushButton_54.clicked.connect(lambda: self.signal_processor.zoomIn(graph=self.graphicsView_56))
            self.pushButton_55.clicked.connect(lambda: self.signal_processor.zoomOut(graph=self.graphicsView_56))
            self.pushButton_56.clicked.connect(lambda: self.signal_processor.fitScreen(graph=self.graphicsView_56))



        elif index == 2:
            print("Third tab clicked")
            # self.pushButton_22.clicked.connect(self.signal_processor.load_signal)
            self.pushButton_22.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.graphicsView_21))
            self.spectrogram_plotter = SpectrogramPlotter(self.verticalLayout_20)
            self.spectrogram_plotter_2 = SpectrogramPlotter(self.verticalLayout_21)
            self.pushButton_22.clicked.connect(self.signal_processor.default_graph_drawing)

            self.pushButton_8.clicked.connect(lambda: self.apply_equalizer_handler(freqGraph=self.graphicsView_25 ,outputTimeGraph=self.graphicsView_23))
            self.signal_processor.clear_modes([1, 2, 4])
            self.comboBox_mode3.currentIndexChanged.connect(
                lambda: self.signal_processor.on_window_type_changed2(index=0, comboBox=self.comboBox_mode3))
            self.graphicsView_21.setXLink(self.graphicsView_23)
            self.graphicsView_21.setYLink(self.graphicsView_23)
            self.pushButton_19.clicked.connect(lambda: self.signal_processor.zoomIn(graph=self.graphicsView_21))
            self.pushButton_20.clicked.connect(lambda: self.signal_processor.zoomOut(graph=self.graphicsView_21))
            self.pushButton_21.clicked.connect(lambda: self.signal_processor.fitScreen(graph=self.graphicsView_21))


        elif index == 3:
            print("Fourth tab clicked")
            # self.pushButton_27.clicked.connect(self.signal_processor.load_signal)
            self.pushButton_27.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.graphicsView_26))
            self.spectrogram_plotter = SpectrogramPlotter(self.verticalLayout_18)
            self.spectrogram_plotter_2 = SpectrogramPlotter(self.verticalLayout_19)
            self.pushButton_27.clicked.connect(self.signal_processor.default_graph_drawing)

            self.pushButton_9.clicked.connect(lambda: self.apply_equalizer_handler(freqGraph=self.graphicsView_30 ,outputTimeGraph= self.graphicsView_28))
            self.signal_processor.clear_modes([1, 2, 3])
            self.comboBox_mode4.currentIndexChanged.connect(
                lambda: self.signal_processor.on_window_type_changed2(index=0, comboBox=self.comboBox_mode4))
            self.graphicsView_26.setXLink(self.graphicsView_28)
            self.graphicsView_26.setYLink(self.graphicsView_28)
            self.pushButton_24.clicked.connect(lambda: self.signal_processor.zoomIn(graph=self.graphicsView_26))
            self.pushButton_25.clicked.connect(lambda: self.signal_processor.zoomOut(graph=self.graphicsView_26))
            self.pushButton_26.clicked.connect(lambda: self.signal_processor.fitScreen(graph=self.graphicsView_26))




def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()