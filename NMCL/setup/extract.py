import subprocess
import sys
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image, ImageFilter

numberOfImages = int(sys.argv[1])
global_frame_count = 1
rows = []


def imageprepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))

    if width > height:
        nheight = int(round((20.0 / width * height), 0))
        if (nheight == 0):
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(
            ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1

        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(
            ImageFilter.SHARPEN)

        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))
    tv = list(newImage.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva


for i in range(1, numberOfImages + 1):

    vidName = f"../data/video/video{str(i).zfill(4)}.mp4"
    mp3Name = f"../data/audio/audio{str(i).zfill(4)}.mp3"
    wavName = f"../data/audioWav/audio{str(i).zfill(4)}.wav"
    specName = f"../data/spectrogram/spectrogram{str(i).zfill(4)}.png"

    # Extract Audio from video
    extract_audio = list(
        f"ffmpeg -i {vidName} -f mp3 -ab 22050 -y -vn {mp3Name}".strip().split(" "))
    print(extract_audio)
    extract_audio_files = subprocess.run(extract_audio)
    print(f"The exit code was: {extract_audio_files.returncode}")

    # Resample the audio to the right format
    resample_audio = list(
        f"ffmpeg -i {mp3Name} -acodec pcm_s16le -ac 1 -ar 22050 {wavName}".strip().split(" "))
    resample_audio_files = subprocess.run(resample_audio)
    print(f"The exit code was: {resample_audio_files.returncode}")

    # Generate Spectogram
    y, sr = librosa.load(wavName)
    n_fft = 1024
    ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft+1))
    spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    plt.savefig(specName)

    # Extract I - Frames
    p_frame_thresh = 50000  # You may need to adjust this threshold

    cap = cv2.VideoCapture(vidName)
    if (cap.isOpened() == False):
        print("Error opening video file")
    print("zone start")

    # Read the first frame.
    ret, prev_frame = cap.read()

    count_P_frames = 0
    count_frames = 0

    while ret:
        ret, curr_frame = cap.read()
        count_frames += 1
        if ret:
            gray_curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_curr_frame, gray_prev_frame)
            non_zero_count = np.count_nonzero(diff)

            if non_zero_count > p_frame_thresh:
                count_P_frames += 1
                # print("Got P-Frame")
            else:
                imageName = "images/image{}.jpg".format(
                    str(global_frame_count).zfill(6))
                global_frame_count += 1
                # cv2.imshow(imageName, curr_frame)
                cv2.imwrite(imageName, curr_frame)
            prev_frame = curr_frame

    print("Number of PFrames: {} out of {} frames".format(
        count_P_frames, count_frames))

    # Resize the spectrogram to 28x28 pixels and make csv
    im1 = Image.open(
        r'..\data\spectrogram\spectrogram{}.jpg'.format(str(i).zfill(4)))
    im1.save(r'..\data\spectrogram\spectrogram{}.png'.format(str(i).zfill(4)))

    x = imageprepare('spectrogram\\spectrogram{}.png'.format(
        str(i).zfill(4)))  # file path here
    rows.append(x)

    print("zone over")

with open('spectrograms', 'w+') as f:
    write = csv.writer(f)
    write.writerows(rows)
