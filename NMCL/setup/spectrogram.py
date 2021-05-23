import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

for i in range(1, 1299):
    y, sr = librosa.load(f'audioWav/audio{str(i).zfill(4)}.wav')
    n_fft = 1024
    ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft+1))
    spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    plt.savefig(f"spectrogram/spectrogram{str(i).zfill(4)}.png")
