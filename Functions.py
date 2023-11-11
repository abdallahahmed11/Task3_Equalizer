import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import pyqtgraph as pg

import main as main_app
# import matplotlib.pyplot as plt
# from main import *


class Functions:
    def __init__(self):
        self.current_time = None

    def pause_audio(self, player):
        if QMediaPlayer.StoppedState == 0:
            QMediaPlayer.StoppedState = 1
            player.play()
        elif QMediaPlayer.StoppedState == 1:
            QMediaPlayer.StoppedState = 0
            player.pause()

    def handle_media_position_change(self, position):
        self.current_time = position

