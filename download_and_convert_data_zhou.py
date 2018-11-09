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
r"""Downloads and converts a particular(adj.  特别的, 挑剔的, 独有的) dataset.

根据输入的数据集的名称以及路径, 执行对应的函数去下载和转换数据

Usage(n.  用法, 习惯, 使用):
```shell

$ python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import download_and_convert_cifar10
from datasets import download_and_convert_flowers
from datasets import download_and_convert_mnist

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "cifar10", "flowers", "mnist".')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,                                        #temporary(adj.  暂时的, 临时性, 临时的)
    'The directory where the output TFRecords and temporary files are saved.')


def main(_):
  if not FLAGS.dataset_name:  #supply v.  补给, 提供, 供给; 替代他人职务, 替代
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if FLAGS.dataset_name == 'cifar10':
    download_and_convert_cifar10.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'flowers':
    download_and_convert_flowers.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'mnist':
    download_and_convert_mnist.run(FLAGS.dataset_dir)
  else:
    raise ValueError(             #recognized v.  认出, 识别; 正式承认; 认识; 认可, 认定
        'dataset_name [%s] was not recognized.' % FLAGS.dataset_name)

if __name__ == '__main__':
  tf.app.run()
