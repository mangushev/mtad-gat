
import tensorflow as tf
import numpy as np
import os
import argparse
import sys
import random
import logging

from sklearn.preprocessing import MinMaxScaler

FLAGS = None

np.set_printoptions(edgeitems=12, linewidth=10000, precision=4, suppress=True)

logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def example(input, label, anomaly):
    logger.debug ("input shape {}: label shape {}".format(input.shape, label.shape))

    record = {
        'input': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(input, [-1]))),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(label, [-1]))),
        'anomaly': tf.train.Feature(int64_list=tf.train.Int64List(value=[anomaly]))
    }

    return tf.train.Example(features=tf.train.Features(feature=record))

def create_records(files_path, label_path, window_size, tfrecords_file):

    for machine_file in os.listdir(files_path):

        with tf.io.TFRecordWriter(tfrecords_file.format(os.path.splitext(machine_file)[0])) as writer:

            data_array = np.loadtxt(os.path.join(files_path,machine_file), delimiter=',', dtype=np.float32)

            data_array = MinMaxScaler().fit_transform(data_array)

            logger.debug ("data file {}: data shape {}".format(machine_file, data_array.shape))
            if label_path:
                label_array = np.loadtxt(os.path.join(label_path,machine_file), delimiter=',', dtype=np.int)
                logger.debug ("label file {}: label shape {}".format(machine_file, label_array.shape))

            record_count = 0
            position = 0

            while position + window_size + 1 < data_array.shape[0]:

                if label_path:
                    tf_example = example(data_array[position : position + window_size], 
                        data_array[position + window_size + 1], label_array[position + window_size + 1])
                else:
                    tf_example = example(data_array[position : position + window_size], 
                        data_array[position + window_size + 1], 0)

                writer.write(tf_example.SerializeToString())

                record_count = record_count + 1
                position = position + 1

            logger.info ("file {}: record_count {}".format(machine_file, record_count))

    return record_count

def main():
    create_records(FLAGS.files_path, FLAGS.label_path, FLAGS.window_size, FLAGS.tfrecords_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int, default=100,
            help='Time series window size.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--files_path', type=str, default='/work/datasets/ServerMachineDataset/train',
            help='Metrics files, train or test.')
    parser.add_argument('--label_path', type=str, default=None,
            help='Labels location.')
    parser.add_argument('--tfrecords_file', type=str, default='gs://anomaly_detection/mtad_tf/data/train/{}.tfrecords',
            help='tfrecords output file. It will be used as a prefix if split.')

    FLAGS, unparsed = parser.parse_known_args()

    logger.setLevel(FLAGS.logging)

    logger.debug ("Running with parameters: {}".format(FLAGS))

    main()
