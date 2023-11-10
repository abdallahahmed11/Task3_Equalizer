from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUiType
import os
from os import path
import sys
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph as pg
# import the function from utils.py
from Functions import load_signal
from Functions import get_freq_components
from Functions import apply_equalizer



# import UI file
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "Equalizerr_2.ui"))



# initiate UI file
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        # Add your custom logic and functionality here
        self.Handle_Buttons()

    def Handle_Buttons(self):
        self.browseBtn.clicked.connect(lambda: load_signal(self))
        self.applyBtn.clicked.connect(lambda: apply_equalizer(self, *get_freq_components(self, self.signal)))


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()