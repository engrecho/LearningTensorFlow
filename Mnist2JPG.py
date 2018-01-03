import Pillow
from scipy.misc import imsave
import tensorflow as tf
import os
import gzip
import numpy



class Mnist2JPG():
  """docstring for Mni"""
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  def __init__(self, imagetype, Num, validation_size=5000):
    if not os.path.isdir("mnist_images"):
      os.makedirs("mnist_images")
    if imagetype = 'train':
      filename = os.path.join('mnist_images', TRAIN_IMAGES)
      shift = Num 
    if imagetype = 'validation:'
      filename = os.path.join('mnist_images', TRAIN_IMAGES)
      shift = Num + validation_size
    if imagetype = 'test'
      filename = os.path.join('mnist_images', TEST_IMAGES)
      shift = Num


    Data = GetImages(filename)

    imsave("train-images/" + str(Num) + ".jpg", Data[shift][:,:,0])

  def _read32(bytestream):
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
      magic = _read32(bytestream)
      if magic != 2051:
        raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                         (magic, f.name))
      num_images = _read32(bytestream)
      rows = _read32(bytestream)
      cols = _read32(bytestream)
      buf = bytestream.read(rows * cols * num_images)
      data = numpy.frombuffer(buf, dtype=numpy.uint8)
      data = data.reshape(num_images, rows, cols, 1)
      print('Extracting', f.name)
      return data

  def ShowImage()
    