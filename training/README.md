Training the Dataset

Visual Modality

1. Use OpenCV to find keyframes for a video
2. Use the extracted images to train ResNet using Caffe
3. Use Average pooling on all keyframes

Acoustic Modality

1. Separate Audio from video using FFmpeg
   cli command available
2. Resample the audio to 22050Hz, 16-bit mono channel
   default audio resampling rate
3. Using librosa to make spectogram on 46ms window and 50% overlapping
   cli command
4. Spectrogram is an image. Train it using Denoising AutoEncoder
   Original Source Code in python 2.7, is available

Textual Modality

1. Use Paragraph Vector

Note:

1. Implementation of the mathematical logic for Neural Multimodal Cooperative Learning has been done.
2. Libraries have been used to train the dataset
3. Program to train the dataset has been excluded
4. Pretrained models are available on Baidu
