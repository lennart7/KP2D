import os
import torchvision.transforms as transforms

from PIL import Image

DATA_DIR = "/data/datasets/keypoint_export"

for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)
    image = Image.open(path)
    image_transform = transforms.Resize((640, 480), interpolation=Image.ANTIALIAS)
    try:
        image_transform(image)
    except OSError:
        print(path)
        os.remove(path)
