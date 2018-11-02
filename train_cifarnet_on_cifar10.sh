#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifarnet_on_cifar10.sh
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=~/tmp/cifarnet-model

# Where the dataset is saved to.
DATASET_DIR=~/tmp/cifar10

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=cifar10 \
  --dataset_dir=${DATASET_DIR}

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet \
  --preprocessing_name=cifarnet \
  --max_number_of_steps=5000 \
  --batch_size=128 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004

# Run evaluation.
# INFO:tensorflow:Evaluation [198/200]
# INFO:tensorflow:Evaluation [199/200]
# INFO:tensorflow:Evaluation [200/200]
# 2018-10-31 17:34:15.447038: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.7588]
# 2018-10-31 17:34:15.447241: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[0.9811]
# INFO:tensorflow:Finished evaluation at 2018-10-31-09:34:15
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=cifarnet

# inference a picture with a ckpt
# INFO:tensorflow:Restoring parameters from /home/daniel/tmp/cifarnet-model/model.ckpt-5000
# [array([[-3.1016066 , -4.70461   ,  0.13729343,  5.724987  , -1.3885372 ,
#          10.317946  ,  0.6634918 , -0.02268255, -3.6647892 , -3.9853208 ]],
#          dtype=float32)]
# 5
python inference_with_ckpt.py \
  --model_name=cifarnet \
  --checkpoint_path=${TRAIN_DIR} \
  --dataset_name=cifar10 \
  --dataset_dir=${DATASET_DIR} \
  --pic_path=${DATASET_DIR}/test.jpg \
  --model_name=cifarnet


# export the graph
# INFO:tensorflow:Scale of 0 disables regularizer.
python export_inference_graph_v2.py \
  --model_name=cifarnet \
  --batch_size=1 \
  --dataset_name=cifar10 \
  --output_file=${DATASET_DIR}/cifarnet_graph_def.pb \
  --dataset_dir=${DATASET_DIR}

# freeze the graph
# Converted 10 variables to const ops.
python freeze_graph.py \
  --input_graph=${DATASET_DIR}/cifarnet_graph_def.pb \
  --input_binary=True \
  --input_checkpoint=${TRAIN_DIR}/model.ckpt-5000 \
  --output_graph=${DATASET_DIR}/freezed_cifarnet_5000.pb \
  --output_node_names=output

# inference a picture with the freezed graph
# [[1.47070443e-06 2.96039616e-07 3.75112759e-05 1.00199655e-02
#   8.15646035e-06 9.89835680e-01 6.34873068e-05 3.19657665e-05
#   8.37410028e-07 6.07761251e-07]]
# 5
python inference_with_freezed_graph_pb.py \
  --freezed_graph=${DATASET_DIR}/freezed_cifarnet_5000.pb \
  --pic_path=${DATASET_DIR}/test.jpg
