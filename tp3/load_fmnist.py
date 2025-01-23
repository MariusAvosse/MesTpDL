import numpy as np
import gzip
import os

DATA_DIR = "./datasets/fashion-mnist"
TRAIN_IMAGES_FILE = os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")
TRAIN_LABELS_FILE = os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")
TEST_IMAGES_FILE = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz")
TEST_LABELS_FILE = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")

TEXT_LABELS = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

def load_images(test=False):
    file_path = TRAIN_IMAGES_FILE
    if test:
        file_path = TEST_IMAGES_FILE
    with gzip.open(file_path, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32, count=4)
        num_images, rows, cols = num_images.byteswap(), rows.byteswap(), cols.byteswap()
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num_images, rows, cols) / 255.0

def load_labels(test=False):
    file_path = TRAIN_LABELS_FILE
    if test:
        file_path = TEST_LABELS_FILE
    with gzip.open(file_path, 'rb') as f:
        magic, num_labels = np.frombuffer(f.read(8), dtype=np.uint32, count=2)
        num_labels = num_labels.byteswap()
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels