# Copyright(n.  版权; 著作权) 2016 The TensorFlow Authors. 
# All Rights Reserved(adj.  留作专用的; 储备的; 预订的; 沉默寡言的).
#
# Licensed(adj.  得到许可的) under the Apache License, Version 2.0 (the "License");
# you may not use this file except(v.  除, 反对, 除外; 反对; 表示异议) in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required(adj.  必须的; 必修的) by applicable(adj.  可应用的; 可适用的) law or agreed to in writing, 
# software distributed(v.  分发; 散布; 分配) under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES(n.  担保, 根据, 保证) OR CONDITIONS(n.  情况; 环境; 状态; 形势) OF ANY KIND, 
# either express(n.  快车; 专使; 快递) or implied(v.  暗示; 意味).
# See the License for the specific(n.  特性; 详情; 特效药; 详细说明书) language 
# governing(v.  统治; 决定, 影响, 指导; 管理; 控制; 进行统治, 管理, 执政; 居支配地位) 
# permissions(n.  许可, 允许, 同意; 施予; 给使用者权力进入某资料源 (计算机用语)) and
# limitations(n.  限制, 限制因素; 极限, 限度; 局限; 追诉时效) under the License.
# ==============================================================================
r"""Downloads and converts cifar10 data to TFRecords of TF-Example protos.

This module downloads the cifar10 data, uncompresses it, reads the files
that make up the cifar10 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised(v.  包含, 构成) of a set of TF-Example
protocol(n.  议定书, 协议, 处理某问题的方法; 草案, 谈判的起草; 礼节, 行为准则; 协议, 处理资料传送的标准 (计算机用语))
buffers(n.  缓冲存储器; 分隔, 划分; 记忆里被指定为暂时存储的位置 (计算机用语); 减震器; 起缓冲作用的人), 
each of which contain a single image and label.

The script should take several minutes to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
from six.moves import cPickle
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

# The URL where the CIFAR data can be downloaded.
# 国内原网址下载速度太慢, 更换其它下载地址
# _DATA_URL = 'http://192.168.0.108/D%3A/cifar-10-python.tar.gz'
_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# The number of training files.
_NUM_TRAIN_FILES = 5

# The height and width of each image.
_IMAGE_SIZE = 32

# The names of the classes.
_CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]


def _add_to_tfrecord(filename, tfrecord_writer, offset=0):
  """Loads data from the cifar10 pickle files and writes files to a TFRecord.

  Args:
    filename: The filename of the cifar10 pickle file.
    tfrecord_writer: The TFRecord writer to use for writing.
    offset: An offset into the absolute number of images previously written.
            absolute(adj.  纯粹的; 绝对的; 完全的; 专制的) previously(adv.  事先; 仓促地; 以前; 不成熟地)
  Returns:
    The new offset.
  """
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info < (3,):
      data = cPickle.load(f)
    else:
      data = cPickle.load(f, encoding='bytes')

# 总的图片数
  images = data[b'data']
  num_images = images.shape[0]
# 将图片转为三通道的32x32的彩图
  images = images.reshape((num_images, 3, 32, 32))
  labels = data[b'labels']

  with tf.Graph().as_default():
    image_placeholder = tf.placeholder(dtype=tf.uint8)
    encoded_image = tf.image.encode_png(image_placeholder)

    with tf.Session('') as sess:
      # 读取每一张图片做处理
      for j in range(num_images):
        sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
            filename, offset + j + 1, offset + num_images))
        sys.stdout.flush()
        # 把原有的(1,3,32,32)转成(32,32,3)
        image = np.squeeze(images[j]).transpose((1, 2, 0))
        label = labels[j]
        # 把(32,32,3)转成str格式
        png_string = sess.run(encoded_image,
                              feed_dict={image_placeholder: image})

        example = dataset_utils.image_to_tfexample(
            png_string, b'png', _IMAGE_SIZE, _IMAGE_SIZE, label)
        # def image_to_tfexample(image_data, image_format, height, width, class_id):
        #   return tf.train.Example(features=tf.train.Features(feature={
        #       'image/encoded': bytes_feature(image_data),
        #       'image/format': bytes_feature(image_format),
        #       'image/class/label': int64_feature(class_id),
        #       'image/height': int64_feature(height),
        #       'image/width': int64_feature(width),
        #   }))
        tfrecord_writer.write(example.SerializeToString())

  return offset + num_images


def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The dataset directory where the dataset is 
                  stored(v.  储存, 供给, 贮藏).
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/cifar10_%s.tfrecord' % (dataset_dir, split_name)


def _download_and_uncompress_dataset(dataset_dir):
  """Downloads cifar10 and uncompresses it locally.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]  #filename
  filepath = os.path.join(dataset_dir, filename)  #locally filename

  if not os.path.exists(filepath): 
    #判断要下载的文件是否存在
    def _progress(count, block_size, total_size):
      # progress(n.  进步, 前进, 发展)
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(_DATA_URL, filepath, _progress)
    # retrieve(v.  重新得到, 收回; 检索, 撷取; 衔回; 使恢复; 找回猎物)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    # 解压文件
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.
              temporary(adj.  暂时的, 临时性, 临时的)

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'cifar-10-batches-py')
  tf.gfile.DeleteRecursively(tmp_dir)
  # Recursively(adv.  重复地)


def run(dataset_dir):
  """Runs the download and conversion(n.  转变, 换位, 改宗) operation.
  operation(n.  操作; 经营; 运转; 营运; 手术)

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train')
  testing_filename = _get_output_filename(dataset_dir, 'test')
  # dataset_dir/cifar10_{train/test}.tfrecord 

  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  # 下载并解压 tar 文件

  # First, process(v.  加工; 用计算机处理;) the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    offset = 0
    for i in range(_NUM_TRAIN_FILES):
      filename = os.path.join(dataset_dir,
                              'cifar-10-batches-py',
                              'data_batch_%d' % (i + 1))  # 1-indexed.
      offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    filename = os.path.join(dataset_dir,
                            'cifar-10-batches-py',
                            'test_batch')
    _add_to_tfrecord(filename, tfrecord_writer)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
  
  # 清理缓存文件
  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Cifar10 dataset!')
