"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): 
    # 返回正态初始化
    return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    # Composite function, H(x) ( BN, Relu, Con, Dropout )
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    # dense_block, 瓶颈层, 减少输入特征维度
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5
    
    def reduce_dim(input_feature):  #压缩, 增加模型紧凑性
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            pass
            ##########################
            # 32 x 32 x 3
            #print(images.get_shape().as_list(),'----------------------------------------------')
            end_point = 'Conv_0'
            net = slim.conv2d(images, 2*growth, [7,7], stride=2, padding='SAME', scope=end_point)
            end_point = 'Pool_0'
            print(net.get_shape().as_list(),'----------------------------------------------')
            # 16 x 16 x 48
            net = slim.max_pool2d(net, [3,3], stride=2, padding='SAME', scope=end_point)
            end_points[end_point] = net
            # 8 x 8 x 48
            print(net.get_shape().as_list(),'----------------------------------------------')
            for i in range(4):
                end_point = 'dense_{}'.format(i+1)
                net = block(net, 6, reduce_dim(net), scope=end_point) #拼接
                print(net.get_shape().as_list(),'----------------------------------------------')
                net = bn_act_conv_drp(net, (i+1)*growth, [1,1], scope=end_point)  #非线性变换
                print(net.get_shape().as_list(),'----------------------------------------------')
                end_points[end_point] = net
                

            # 8 x 8 x L*growth
            end_point = 'logits'
            net_shape = net.get_shape().as_list()
            # global_avg_pool2d
            net = slim.avg_pool2d(net, net_shape[1:3], scope=end_point)
            # [batch_size, 8, 8, i*growth]
            print(net.get_shape().as_list(),'----------------------------------------------')
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope=end_point)
            print(net.get_shape().as_list(),'----------------------------------------------')
            # [batch_size, 1, 1, num_classes]
            logits = tf.squeeze(net, [1, 2], name=end_point)
            # [batch_size, `num_classes`]
            end_points[end_point] = logits
            
            ##########################

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    # 返回(BN / Dropout)
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope([slim.batch_norm],
            scale=True, 
            is_training=is_training, 
            updates_collections=None):
        with slim.arg_scope( [slim.dropout],
                is_training=is_training, 
                keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, 
        biases_initializer=None, 
        padding='same',
        stride=1) as sc:
        return sc


densenet.default_image_size = 224
