from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUiType
from PyQt5.QtCore import QTimer
from os import path
import sys
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import pyqtgraph as pg
import numpy as np
import scipy.io.wavfile as wavfile

import Functions

# import UI file
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "Equalizer-All.ui"))


# initiate UI file
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.player = None
        self.setupUi(self)
        self.handle_buttons()
        self.batch_size = 1000  # Adjust the batch size as needed
        self.audio_data = None
        self.audio_buffer = np.zeros(self.batch_size)
        self.functions = Functions.Functions()
        self.timer = QTimer()
        # self.timer.timeout.connect(lambda: self.update_plot(self.audio_data))
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)
        # self.current_time = 0

    def handle_buttons(self):
        self.pushButton.clicked.connect(self.load_audio)
        self.pushButton_2.clicked.connect(lambda: self.functions.pause_audio(self.player))

    def load_audio(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Audio Files (*.mp3 *.wav *.ogg)")

        file_path, _ = file_dialog.getOpenFileName()

        if file_path:
            self.player = QMediaPlayer()
            media_content = QMediaContent(QUrl.fromLocalFile(file_path))
            self.player.setMedia(media_content)
            self.player.positionChanged.connect(self.functions.handle_media_position_change)

            sr, audio_data = wavfile.read(file_path)
            print(f'sample rate: {sr}')
            print(f'audio shape: {audio_data.shape}')
            print(f'number of samples: {audio_data.shape[0]}')
            print(f'number of channels: {audio_data.shape[1]}')
            audio_data = audio_data[:, 0]
            # length = audio_data.shape[0] / sr
            # time = np.linspace(0., length, audio_data.shape[0])
            # plt.plot(time, audio_data[:, 0], label="Left channel")
            # plt.show()
            self.audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            if QMediaPlayer.StoppedState == 0:
                QMediaPlayer.StoppedState = 1
                self.player.play()

    def plot_audio(self, audio_data):
        if audio_data is not None:
            self.graphicsView.clear()
            x = np.arange(0, len(audio_data))
            self.graphicsView.setYRange(-1, 1)
            self.graphicsView.setXRange(self.buffer_start, self.buffer_end)
            self.graphicsView.addItem(pg.PlotDataItem(audio_data))

    # def update_plot(self, audio_data):
    #     current_time = self.functions.current_time
    #     if audio_data is None:
    #         return
    #     if current_time is not None and current_time < len(audio_data):
    #         self.plot_audio(audio_data[:int(current_time)])
    #         # self.plot_audio(self.audio_data[:int(self.functions.current_time * len(self.audio_data))])
    #
    #     else:
    #         current_time = 0  # Loop when audio is finished

    def update_plot(self):
        if self.audio_data is not None and self.functions.current_time is not None:
            current_time = self.functions.current_time
            if QMediaPlayer.PlayingState == 1:
                # Update the audio buffer with new data
                # print(int(current_time * len(self.audio_data)))
                # self.buffer_start = int(current_time * len(self.audio_data))
                self.buffer_start = int(current_time)
                self.buffer_end = self.buffer_start + self.batch_size
                if self.buffer_end > len(self.audio_data):
                    # self.buffer_start = len(self.audio_data) - self.batch_size
                    # self.buffer_end = len(self.audio_data)
                    self.buffer_start = self.buffer_end
                    # self.buffer_end = self.buffer_start + self.batch_size
                    self.buffer_end = current_time

                self.audio_buffer = self.audio_data[self.buffer_start:self.buffer_end]

                # Plot the updated audio buffer
                self.plot_audio(self.audio_buffer)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()