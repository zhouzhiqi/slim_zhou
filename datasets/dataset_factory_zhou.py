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
"""A factory-pattern class which returns classification image/label pairs."""
# pattern(n.  花样, 图案; 格局; 形态, 样式; 样品, 样本)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import cifar10
from datasets import flowers
from datasets import imagenet
from datasets import mnist
from datasets import quiz

datasets_map = {
    'cifar10': cifar10,
    'flowers': flowers,
    'quiz': quiz,
    'imagenet': imagenet,
    'mnist': mnist,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    match(v.  使较量, 使比赛; 和...相配, 和...相称; 检查数据项目的相似性 (计算机用语))
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      subclass(n.  亚纲; 子集合)
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return datasets_map[name].get_split(
      split_name,
      dataset_dir,
      file_pattern,
      reader)
