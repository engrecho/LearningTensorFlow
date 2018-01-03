import os
import gzip
import numpy

from scipy.misc import imsave
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile


class mnist2image():
    """
    ImageType: only in (train,validation,test)
    Num : train [0:60000-validation_size-1]; validation [0:validation_size-1] test [0:9999]
        validation_size = train_size[:validation_size]
        validation_size = train_size[:validation_size]
        train_size = train_size[validation_size:]
        train_size = train_size[validation_size:]
    """
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    def __init__(self, ImageType = 'train', Num = 0, validation_size=5000):
        self.Num = Num


        if ImageType == 'validation':
            filename = 'MNIST_data/' + self.TRAIN_IMAGES
            self.shift = Num
        elif ImageType == 'train':
            filename = 'MNIST_data/' + self.TRAIN_IMAGES
            self.shift = Num + validation_size
        elif ImageType == 'test':
            filename = 'MNIST_data/' + self.TEST_IMAGES
            self.shift = Num
        else:
            filename = 'MNIST_data/' + self.TRAIN_IMAGES  #如果输入错误，就按照Train来处理
            self.shift = Num

        with gfile.Open(filename, 'rb') as f:
            self.Data = self.GetImages(f)

    def save_image(self):
        if not os.path.isdir("mnist_images"):
            os.makedirs("mnist_images")
        filename = "mnist_images/" + str(self.Num) + ".jpg"
        try:
            imsave(filename, self.Data[self.shift][:, :, 0])
            print(filename, 'Save Successfully!')
        except Exception:
            print(filename, 'Save Failed! ')


    def show_image(self):
        plt.imshow(self.Data[self.shift][:, :, 0])
        plt.show()

    def read32(self,bytestream):
        """
        Read image zip bytestream
        """
        dt = numpy.dtype(numpy.uint32).newbyteorder('>')
        return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


    def GetImages(self,f):

        """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

        Args:
          f: A file object that can be passed into a gzip reader.

        Returns:
          data: A 4D uint8 numpy array [Image_index, y, x, depth].

        Raises:
          ValueError: If the bytestream does not start with 2051.

        """
        with gzip.GzipFile(fileobj=f) as bytestream:

            magic = self.read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, f.name))
            num_images = self.read32(bytestream)
            rows = self.read32(bytestream)
            cols = self.read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = numpy.frombuffer(buf, dtype=numpy.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            # print('Extracting', f.name)
            return data


if __name__ == '__main__':
    a=mnist2image('train',1).show_image()
    b=mnist2image('test',1).save_image()
