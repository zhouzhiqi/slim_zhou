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
"""Provides data for the Cifar10 dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_cifar10.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'cifar10_%s.tfrecord'

SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}

_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'label': 'A single integer between 0 and 9',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading cifar10.
                                instruction(n.  指示; 教育; 用法说明)

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
                            pattern(n.  花样, 图案; 格局; 形态, 样式; 样品, 样本)
      It is assumed that the pattern contains a '%s' string so that the split
      assumed(adj.  假装的; 假定的, 设想的; 假冒的; 被承担的)
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  #                      signature(n.  签名, 签字, 签署)
  if not reader:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(shape=[32, 32, 3]),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
