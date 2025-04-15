import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import librosa
import librosa.display
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
import os

# Initial parameters
INIT_FRAME_LENGTH = 2048
INIT_HOP_LENGTH = 512
INIT_DB_RANGE = 80
INIT_CMAP = 'viridis'
INIT_SCALE = 'linear'

class SpectrogramExplorer:
    def __init__(self):
        self.audio_path = None
        self.y = None
        self.sr = None
        self.fig = None
        self.ax = None
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.withdraw()  # Hide main tk window
        self.load_audio_file()
        
        if self.audio_path:
            self.setup_interface()
            plt.show()

    def load_audio_file(self):
        file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")]
        )
        if file_path:
            self.audio_path = file_path
            self.y, self.sr = librosa.load(file_path, sr=None)
            print(f"Loaded: {os.path.basename(file_path)} | SR: {self.sr} Hz | Duration: {len(self.y)/self.sr:.2f}s")

    def setup_interface(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.4)
        
        # Create sliders
        slider_ax = [
            plt.axes([0.25, 0.25, 0.55, 0.03]),  # FFT size
            plt.axes([0.25, 0.20, 0.55, 0.03]),  # Hop length
            plt.axes([0.25, 0.15, 0.55, 0.03]),  # dB range
            plt.axes([0.25, 0.10, 0.55, 0.03]),  # Window type
            plt.axes([0.25, 0.05, 0.55, 0.03]),  # Scale
        ]

        self.sliders = {
            'frame_length': Slider(slider_ax[0], 'FFT Size', 256, 4096, valinit=INIT_FRAME_LENGTH, valstep=256),
            'hop_length': Slider(slider_ax[1], 'Hop Length', 64, 2048, valinit=INIT_HOP_LENGTH, valstep=64),
            'db_range': Slider(slider_ax[2], 'dB Range', 20, 120, valinit=INIT_DB_RANGE),
            'cmap': Slider(slider_ax[3], 'Colormap', 0, 3, valinit=0, valstep=1),
            'scale': Slider(slider_ax[4], 'Scale', 0, 1, valinit=0, valstep=1)
        }

        # Add button for color maps
        cmap_ax = plt.axes([0.85, 0.7, 0.1, 0.15])
        self.cmap_btn = Button(cmap_ax, 'Cycle\nCmap', color='lightgoldenrodyellow')
        self.cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        
        # Set initial state
        self.current_cmap = INIT_CMAP
        self.current_scale = INIT_SCALE
        self.update_spectrogram()
        
        # Wire up events
        for slider in self.sliders.values():
            slider.on_changed(self.update_spectrogram)
            
        self.cmap_btn.on_clicked(self.cycle_colormap)

    def compute_spectrogram(self, frame_length, hop_length, scale):
        if scale == 'mel':
            S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, 
                                              n_fft=frame_length, 
                                              hop_length=hop_length)
        else:
            S = np.abs(librosa.stft(self.y, n_fft=frame_length, hop_length=hop_length))
        
        return librosa.amplitude_to_db(S, ref=np.max)

    def update_spectrogram(self, val=None):
        frame_length = int(self.sliders['frame_length'].val)
        hop_length = int(self.sliders['hop_length'].val)
        db_range = self.sliders['db_range'].val
        scale = 'mel' if self.sliders['scale'].val > 0.5 else 'linear'
        
        self.ax.clear()
        
        # Compute spectrogram
        D = self.compute_spectrogram(frame_length, hop_length, scale)
        
        # Display parameters
        img = librosa.display.specshow(D, sr=self.sr, 
                                     hop_length=hop_length,
                                     x_axis='time', y_axis=scale,
                                     cmap=self.current_cmap,
                                     vmin=-db_range, vmax=0,
                                     ax=self.ax)
        
        self.ax.set(title=f'Spectrogram: {os.path.basename(self.audio_path)}\n'
                        f"FFT: {frame_length} | Hop: {hop_length} | Scale: {scale.upper()}")
        plt.draw()

    def cycle_colormap(self, event):
        current_idx = self.cmaps.index(self.current_cmap)
        self.current_cmap = self.cmaps[(current_idx + 1) % len(self.cmaps)]
        self.update_spectrogram()

if __name__ == "__main__":
    SpectrogramExplorer()
