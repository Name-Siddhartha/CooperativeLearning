import csv
from PIL import Image, ImageFilter
import sys


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


numberOfImages = int(sys.argv[1])
rows = []
for i in range(1, numberOfImages + 1):
    print(i)
    im1 = Image.open(r'spectrogram\spectrogram{}.jpg'.format(str(i).zfill(4)))
    im1.save(r'spectrogram\spectrogram{}.png'.format(str(i).zfill(4)))

    x = imageprepare('spectrogram\\spectrogram{}.png'.format(
        str(i).zfill(4)))  # file path here
    rows.append(x)

with open('spectrograms', 'w+') as f:
    write = csv.writer(f)
    write.writerows(rows)
