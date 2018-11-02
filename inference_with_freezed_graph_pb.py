from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'cifar10', 'The name of the architecture to save.')

tf.app.flags.DEFINE_string(
    'pic_path', '', 'The path of the picture to inference.')


tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string('dataset_name', 'cifar10',
                           'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'freezed_graph', 'no_useing', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '', 'Where to save the checkpoint file to.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')

FLAGS = tf.app.flags.FLAGS



def main(_):
  if not FLAGS.freezed_graph:
    raise ValueError('You must supply the path to save to with --freezed_graph')
  tf.logging.set_verbosity(tf.logging.INFO)

  with open(FLAGS.freezed_graph, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  image_value = open(FLAGS.pic_path, 'rb').read()
  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('output:0')
    predictions = sess.run(softmax_tensor, feed_dict={'input:0':image_value})
    print(predictions)
    print(np.argmax(predictions))

"""
  with tf.Graph().as_default() as graph:
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                          FLAGS.dataset_dir)
    preprocessing_name = FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    image_size = FLAGS.image_size or network_fn.default_image_size
#    x = tf.placeholder(name='input', dtype=tf.float32,
#                                 shape=[FLAGS.batch_size, image_size,
#                                        image_size, 3])

    placeholder = tf.placeholder(name='input', dtype=tf.string)
    image = tf.image.decode_jpeg(placeholder, channels=3)
    image = image_preprocessing_fn(image, image_size, image_size)
    x = tf.expand_dims(image, axis=0)

    logits, end_points = network_fn(x)
    preditions = tf.nn.softmax(logits, name='output')
    saver = tf.train.Saver()
    sess = tf.Session()
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    saver.restore(sess, checkpoint_path)
    image_value = open(FLAGS.pic_path, 'rb').read()
    logit_value = sess.run([logits], feed_dict={placeholder:image_value})
    print(logit_value)
    print(np.argmax(logit_value))
"""

if __name__ == '__main__':
  tf.app.run()
