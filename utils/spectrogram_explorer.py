import numpy as np
import matplotlib
import sys
import os
import librosa
import librosa.display

# Set the backend explicitly before importing FigureCanvas
matplotlib.use('QtAgg')

# Import after setting the backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QVBoxLayout, 
    QSlider, QPushButton, QLabel, QWidget, 
    QComboBox, QGroupBox, QGridLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox, QHBoxLayout
)
from PySide6.QtCore import Qt, Slot

# Initial parameters
INIT_FRAME_LENGTH = 2048
INIT_HOP_LENGTH = 512
INIT_DB_RANGE = 80
INIT_CMAP = 'viridis'
INIT_SCALE = 'linear'
INIT_N_MELS = 128
INIT_DPI = 100
INIT_INTERPOLATION = 'nearest'

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=8, dpi=INIT_DPI):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)

class SpectrogramExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.audio_path = None
        self.y = None
        self.sr = None
        self.colorbar = None  # Track the colorbar to prevent duplicates
        
        # Available colormaps
        self.cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                      'jet', 'hot', 'cool', 'gray', 'bone']
        self.interpolations = ['nearest', 'bilinear', 'bicubic', 'spline16',
                              'spline36', 'hanning', 'hamming', 'hermite', 
                              'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel',
                              'mitchell', 'sinc', 'lanczos', 'none']
        self.current_cmap = INIT_CMAP
        self.current_scale = INIT_SCALE
        self.current_interpolation = INIT_INTERPOLATION
        self.dpi = INIT_DPI
        
        self.setWindowTitle("Spectrogram Explorer")
        self.setup_ui()
        
        # Load audio file on startup
        self.load_audio_file()

    def setup_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Matplotlib canvas for the spectrogram
        self.canvas = MatplotlibCanvas(self, width=12, height=8, dpi=self.dpi)
        main_layout.addWidget(self.canvas)
        
        # Controls layout
        controls_group = QGroupBox("Spectrogram Controls")
        controls_layout = QGridLayout()
        
        # FFT Size slider
        self.frame_length_label = QLabel(f"FFT Size: {INIT_FRAME_LENGTH}")
        self.frame_length_slider = QSlider(Qt.Horizontal)
        self.frame_length_slider.setMinimum(256)
        self.frame_length_slider.setMaximum(8192)
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
        
        # Mel bins control (for mel spectrograms)
        self.n_mels_label = QLabel(f"Frequency Bins: {INIT_N_MELS}")
        self.n_mels_spinbox = QSpinBox()
        self.n_mels_spinbox.setMinimum(32)
        self.n_mels_spinbox.setMaximum(512)
        self.n_mels_spinbox.setValue(INIT_N_MELS)
        self.n_mels_spinbox.valueChanged.connect(self.on_n_mels_changed)
        
        # DPI control for resolution
        self.dpi_label = QLabel(f"DPI (Resolution): {INIT_DPI}")
        self.dpi_spinbox = QSpinBox()
        self.dpi_spinbox.setMinimum(72)
        self.dpi_spinbox.setMaximum(300)
        self.dpi_spinbox.setValue(INIT_DPI)
        self.dpi_spinbox.valueChanged.connect(self.on_dpi_changed)
        
        # Frequency range controls
        self.freq_range_label = QLabel("Frequency Range (Hz):")
        self.freq_min_spinbox = QSpinBox()
        self.freq_min_spinbox.setMinimum(0)
        self.freq_min_spinbox.setMaximum(20000)
        self.freq_min_spinbox.setValue(0)
        self.freq_min_spinbox.valueChanged.connect(self.update_spectrogram)
        
        self.freq_max_spinbox = QSpinBox()
        self.freq_max_spinbox.setMinimum(100)
        self.freq_max_spinbox.setMaximum(22050)
        self.freq_max_spinbox.setValue(22050)
        self.freq_max_spinbox.valueChanged.connect(self.update_spectrogram)
        
        # Colormap dropdown
        self.cmap_label = QLabel("Colormap:")
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(self.cmaps)
        self.cmap_combo.setCurrentText(INIT_CMAP)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        
        # Interpolation dropdown
        self.interpolation_label = QLabel("Interpolation:")
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(self.interpolations)
        self.interpolation_combo.setCurrentText(INIT_INTERPOLATION)
        self.interpolation_combo.currentTextChanged.connect(self.on_interpolation_changed)
        
        # Scale dropdown
        self.scale_label = QLabel("Scale:")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['linear', 'mel', 'log'])
        self.scale_combo.setCurrentText(INIT_SCALE)
        self.scale_combo.currentTextChanged.connect(self.on_scale_changed)
        
        # Load button
        self.load_button = QPushButton("Load Audio File")
        self.load_button.clicked.connect(self.load_audio_file)
        
        # Save image button
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_spectrogram)
        
        # Add widgets to the grid layout
        row = 0
        controls_layout.addWidget(self.frame_length_label, row, 0)
        controls_layout.addWidget(self.frame_length_slider, row, 1)
        
        row += 1
        controls_layout.addWidget(self.hop_length_label, row, 0)
        controls_layout.addWidget(self.hop_length_slider, row, 1)
        
        row += 1
        controls_layout.addWidget(self.db_range_label, row, 0)
        controls_layout.addWidget(self.db_range_slider, row, 1)
        
        row += 1
        controls_layout.addWidget(self.n_mels_label, row, 0)
        controls_layout.addWidget(self.n_mels_spinbox, row, 1)
        
        row += 1
        controls_layout.addWidget(self.dpi_label, row, 0)
        controls_layout.addWidget(self.dpi_spinbox, row, 1)
        
        row += 1
        freq_range_layout = QHBoxLayout()
        freq_range_layout.addWidget(self.freq_min_spinbox)
        freq_range_layout.addWidget(QLabel("-"))
        freq_range_layout.addWidget(self.freq_max_spinbox)
        controls_layout.addWidget(self.freq_range_label, row, 0)
        controls_layout.addLayout(freq_range_layout, row, 1)
        
        row += 1
        controls_layout.addWidget(self.cmap_label, row, 0)
        controls_layout.addWidget(self.cmap_combo, row, 1)
        
        row += 1
        controls_layout.addWidget(self.interpolation_label, row, 0)
        controls_layout.addWidget(self.interpolation_combo, row, 1)
        
        row += 1
        controls_layout.addWidget(self.scale_label, row, 0)
        controls_layout.addWidget(self.scale_combo, row, 1)
        
        row += 1
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        controls_layout.addLayout(button_layout, row, 0, 1, 2)
        
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)
        
        self.setCentralWidget(main_widget)
        self.resize(1200, 900)

    @Slot()
    def on_frame_length_changed(self):
        value = self.frame_length_slider.value()
        # Round to nearest power of 2
        power = round(np.log2(value))
        value = 2 ** power
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
        
    @Slot()
    def on_n_mels_changed(self):
        value = self.n_mels_spinbox.value()
        self.n_mels_label.setText(f"Frequency Bins: {value}")
        self.update_spectrogram()
        
    @Slot()
    def on_dpi_changed(self):
        value = self.dpi_spinbox.value()
        self.dpi_label.setText(f"DPI (Resolution): {value}")
        self.dpi = value
        # Recreate the canvas with new DPI
        self.canvas.fig.set_dpi(value)
        self.canvas.draw()
        self.update_spectrogram()

    @Slot(str)
    def on_cmap_changed(self, cmap):
        self.current_cmap = cmap
        self.update_spectrogram()
        
    @Slot(str)
    def on_interpolation_changed(self, interpolation):
        self.current_interpolation = interpolation
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
            
            # Update frequency range max based on sample rate
            nyquist = self.sr // 2
            self.freq_max_spinbox.setMaximum(nyquist)
            self.freq_max_spinbox.setValue(nyquist)
            
            self.update_spectrogram()
            
    @Slot()
    def save_spectrogram(self):
        if self.y is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spectrogram Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
        )
        
        if file_path:
            # Save with high DPI for better quality
            save_dpi = max(self.dpi, 300)  # Use at least 300 DPI for saving
            self.canvas.fig.savefig(file_path, dpi=save_dpi, bbox_inches='tight')
            print(f"Saved spectrogram to {file_path}")

    def compute_spectrogram(self, frame_length, hop_length, scale, n_mels):
        if self.y is not None:
            if scale == 'mel':
                S = librosa.feature.melspectrogram(
                    y=self.y, 
                    sr=self.sr, 
                    n_fft=frame_length, 
                    hop_length=hop_length,
                    n_mels=n_mels
                )
            elif scale == 'log':
                S = np.abs(librosa.stft(self.y, n_fft=frame_length, hop_length=hop_length))
                # Convert to log scale
                S = librosa.amplitude_to_db(S, ref=np.max)
                return S  # Already in dB
            else:  # linear
                S = np.abs(librosa.stft(self.y, n_fft=frame_length, hop_length=hop_length))
            
            return librosa.amplitude_to_db(S, ref=np.max)
        return None

    def update_spectrogram(self):
        if self.y is None:
            return
            
        frame_length = self.frame_length_slider.value()
        hop_length = self.hop_length_slider.value()
        db_range = self.db_range_slider.value()
        n_mels = self.n_mels_spinbox.value()
        freq_min = self.freq_min_spinbox.value()
        freq_max = self.freq_max_spinbox.value()
        
        # Clear the figure completely to prevent colorbar duplication
        self.canvas.fig.clear()
        self.canvas.ax = self.canvas.fig.add_subplot(111)
        
        # Compute spectrogram
        D = self.compute_spectrogram(frame_length, hop_length, self.current_scale, n_mels)
        
        if D is not None:
            # Display parameters
            img = librosa.display.specshow(
                D, 
                sr=self.sr, 
                hop_length=hop_length,
                x_axis='time', 
                y_axis=self.current_scale,
                cmap=self.current_cmap,
                vmin=-db_range, 
                vmax=0,
                ax=self.canvas.ax,
                fmin=freq_min,
                fmax=freq_max
            )
            
            # Set interpolation for the axes images
            for image in self.canvas.ax.get_images():
                image.set_interpolation(self.current_interpolation)
            
            self.canvas.ax.set_title(
                f'Spectrogram: {os.path.basename(self.audio_path)}\n'
                f"FFT: {frame_length} | Hop: {hop_length} | Bins: {n_mels} | Scale: {self.current_scale.upper()}"
            )
            
            # Add colorbar (will be cleared with the figure on next update)
            self.canvas.fig.colorbar(img, ax=self.canvas.ax, format="%+2.0f dB")
            
            # Refresh canvas
            self.canvas.fig.tight_layout()
            self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    explorer = SpectrogramExplorer()
    explorer.show()
    sys.exit(app.exec())
