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
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

LABELS_FILENAME = 'labels.txt'


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]  #将传入数据轩为 元组或列表 
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.  string -> byte

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]  #将传入数据轩为 元组或列表
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]  #filename
  filepath = os.path.join(dataset_dir, filename)  #locally filename

  def _progress(count, block_size, total_size):
    # progress(n.  进步, 前进, 发展)
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  # retrieve(v.  重新得到, 收回; 检索, 撷取; 衔回; 使恢复; 找回猎物)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  # 解压文件
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.
     specify(v.  具体指定; 明确说明; 详细指明; 把...列入说明书)

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  # 读入二进制文件 并解码成str
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  # 以 \n 分隔, 并去掉 None
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    # 往字典里添加 label (integer) : class name
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names
