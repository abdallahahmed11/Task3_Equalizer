import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph as pg

def load_signal(main_app):
    filepath, _ = QFileDialog.getOpenFileName(main_app, "Open File", "", "Data Files (*.dat *.csv)")
    if filepath:
        _, extension = os.path.splitext(filepath)
        if not os.path.exists(filepath):
            QMessageBox.critical(main_app, "File Not Found", f"Could not find file at {filepath}.")
            return
        data = None
        if extension == '.dat':
            # Read the .dat file as 16-bit integers
            data = np.fromfile(filepath, dtype=np.int16)
        elif extension == '.csv':
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        main_app.graphicsView.addItem(pg.PlotDataItem(data))