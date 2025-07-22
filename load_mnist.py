import gzip
import struct
import numpy as np


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 读取前16个字节的header信息
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        
        # 读取图像数据
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)
        images = images.reshape((num_images, rows, cols))
        return images


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
    
# 使用示例
# images = load_mnist_images(r'C:\Users\lab509\dataset\mnist\train-images-idx3-ubyte.gz')
# print("Shape:", images.shape)  # 应该是 (60000, 28, 28)