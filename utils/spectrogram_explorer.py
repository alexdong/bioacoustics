import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
try:
    # Try Qt6 backend first (newer matplotlib versions)
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    try:
        # Fall back to Qt5 backend
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        # Last resort - use the default backend
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        print("Warning: Qt backends not available. Using default backend.")
import librosa
import librosa.display
import soundfile as sf
from PySide6.QtWidgets import (QApplication, QFileDialog, QMainWindow, QVBoxLayout, 
                              QHBoxLayout, QSlider, QPushButton, QLabel, QWidget, 
                              QComboBox, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt, Slot
import os
import sys

# Initial parameters
INIT_FRAME_LENGTH = 2048
INIT_HOP_LENGTH = 512
INIT_DB_RANGE = 80
INIT_CMAP = 'viridis'
INIT_SCALE = 'linear'

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)

class SpectrogramExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.audio_path = None
        self.y = None
        self.sr = None
        
        # Available colormaps
        self.cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        self.current_cmap = INIT_CMAP
        self.current_scale = INIT_SCALE
        
        self.setWindowTitle("Spectrogram Explorer")
        self.setup_ui()
        
        # Load audio file on startup
        self.load_audio_file()

    def setup_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Matplotlib canvas for the spectrogram
        self.canvas = MatplotlibCanvas(self, width=12, height=8)
        main_layout.addWidget(self.canvas)
        
        # Controls layout
        controls_group = QGroupBox("Spectrogram Controls")
        controls_layout = QGridLayout()
        
        # FFT Size slider
        self.frame_length_label = QLabel(f"FFT Size: {INIT_FRAME_LENGTH}")
        self.frame_length_slider = QSlider(Qt.Horizontal)
        self.frame_length_slider.setMinimum(256)
        self.frame_length_slider.setMaximum(4096)
        self.frame_length_slider.setValue(INIT_FRAME_LENGTH)
        self.frame_length_slider.setSingleStep(256)
        self.frame_length_slider.valueChanged.connect(self.on_frame_length_changed)
        
        # Hop Length slider
        self.hop_length_label = QLabel(f"Hop Length: {INIT_HOP_LENGTH}")
        self.hop_length_slider = QSlider(Qt.Horizontal)
        self.hop_length_slider.setMinimum(64)
        self.hop_length_slider.setMaximum(2048)
        self.hop_length_slider.setValue(INIT_HOP_LENGTH)
        self.hop_length_slider.setSingleStep(64)
        self.hop_length_slider.valueChanged.connect(self.on_hop_length_changed)
        
        # dB Range slider
        self.db_range_label = QLabel(f"dB Range: {INIT_DB_RANGE}")
        self.db_range_slider = QSlider(Qt.Horizontal)
        self.db_range_slider.setMinimum(20)
        self.db_range_slider.setMaximum(120)
        self.db_range_slider.setValue(INIT_DB_RANGE)
        self.db_range_slider.valueChanged.connect(self.on_db_range_changed)
        
        # Colormap dropdown
        self.cmap_label = QLabel("Colormap:")
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(self.cmaps)
        self.cmap_combo.setCurrentText(INIT_CMAP)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        
        # Scale dropdown
        self.scale_label = QLabel("Scale:")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['linear', 'mel'])
        self.scale_combo.setCurrentText(INIT_SCALE)
        self.scale_combo.currentTextChanged.connect(self.on_scale_changed)
        
        # Load button
        self.load_button = QPushButton("Load Audio File")
        self.load_button.clicked.connect(self.load_audio_file)
        
        # Add widgets to the grid layout
        controls_layout.addWidget(self.frame_length_label, 0, 0)
        controls_layout.addWidget(self.frame_length_slider, 0, 1)
        controls_layout.addWidget(self.hop_length_label, 1, 0)
        controls_layout.addWidget(self.hop_length_slider, 1, 1)
        controls_layout.addWidget(self.db_range_label, 2, 0)
        controls_layout.addWidget(self.db_range_slider, 2, 1)
        controls_layout.addWidget(self.cmap_label, 3, 0)
        controls_layout.addWidget(self.cmap_combo, 3, 1)
        controls_layout.addWidget(self.scale_label, 4, 0)
        controls_layout.addWidget(self.scale_combo, 4, 1)
        controls_layout.addWidget(self.load_button, 5, 0, 1, 2)
        
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)
        
        self.setCentralWidget(main_widget)
        self.resize(1200, 800)

    @Slot()
    def on_frame_length_changed(self):
        value = self.frame_length_slider.value()
        # Round to nearest multiple of 256
        value = (value // 256) * 256
        if value < 256:
            value = 256
        self.frame_length_slider.setValue(value)
        self.frame_length_label.setText(f"FFT Size: {value}")
        self.update_spectrogram()

    @Slot()
    def on_hop_length_changed(self):
        value = self.hop_length_slider.value()
        # Round to nearest multiple of 64
        value = (value // 64) * 64
        if value < 64:
            value = 64
        self.hop_length_slider.setValue(value)
        self.hop_length_label.setText(f"Hop Length: {value}")
        self.update_spectrogram()

    @Slot()
    def on_db_range_changed(self):
        value = self.db_range_slider.value()
        self.db_range_label.setText(f"dB Range: {value}")
        self.update_spectrogram()

    @Slot(str)
    def on_cmap_changed(self, cmap):
        self.current_cmap = cmap
        self.update_spectrogram()

    @Slot(str)
    def on_scale_changed(self, scale):
        self.current_scale = scale
        self.update_spectrogram()

    @Slot()
    def load_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*.*)"
        )
        
        if file_path:
            self.audio_path = file_path
            self.y, self.sr = librosa.load(file_path, sr=None)
            print(f"Loaded: {os.path.basename(file_path)} | SR: {self.sr} Hz | Duration: {len(self.y)/self.sr:.2f}s")
            self.setWindowTitle(f"Spectrogram Explorer - {os.path.basename(file_path)}")
            self.update_spectrogram()

    def compute_spectrogram(self, frame_length, hop_length, scale):
        if not self.y is None:
            if scale == 'mel':
                S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, 
                                                n_fft=frame_length, 
                                                hop_length=hop_length)
            else:
                S = np.abs(librosa.stft(self.y, n_fft=frame_length, hop_length=hop_length))
            
            return librosa.amplitude_to_db(S, ref=np.max)
        return None

    def update_spectrogram(self):
        if self.y is None:
            return
            
        frame_length = self.frame_length_slider.value()
        hop_length = self.hop_length_slider.value()
        db_range = self.db_range_slider.value()
        
        # Clear the axis
        self.canvas.ax.clear()
        
        # Compute spectrogram
        D = self.compute_spectrogram(frame_length, hop_length, self.current_scale)
        
        if D is not None:
            # Display parameters
            img = librosa.display.specshow(D, sr=self.sr, 
                                        hop_length=hop_length,
                                        x_axis='time', y_axis=self.current_scale,
                                        cmap=self.current_cmap,
                                        vmin=-db_range, vmax=0,
                                        ax=self.canvas.ax)
            
            self.canvas.ax.set_title(f'Spectrogram: {os.path.basename(self.audio_path)}\n'
                            f"FFT: {frame_length} | Hop: {hop_length} | Scale: {self.current_scale.upper()}")
            
            # Add colorbar
            self.canvas.fig.colorbar(img, ax=self.canvas.ax, format="%+2.0f dB")
            
            # Refresh canvas
            self.canvas.fig.tight_layout()
            self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    explorer = SpectrogramExplorer()
    explorer.show()
    sys.exit(app.exec())
