
import numpy as np
np.set_printoptions(edgeitems=25, linewidth=10000, precision=4, suppress=True)

import collections
import re
import argparse
import sys
import os
import tensorflow as tf

from model import MtadGat, get_shape_list
from evaluate import calculate_metrics

FLAGS = None

def make_input_fn(filename, is_training, drop_reminder):
  """Returns an `input_fn` for train and eval."""

  def input_fn(params):
    def parser(serialized_example):
      example = tf.io.parse_single_example(
          serialized_example,
          features={
              "input": tf.io.FixedLenFeature([FLAGS.window_size, FLAGS.num_features], tf.float32),
              "label": tf.io.FixedLenFeature([FLAGS.num_features], tf.float32),
              "anomaly": tf.io.FixedLenFeature((), tf.int64)
          })
      
      return example

    dataset = tf.data.TFRecordDataset(
      filename, buffer_size=FLAGS.dataset_reader_buffer_size)
    
    if is_training:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
        parser, batch_size=params["batch_size"],
        num_parallel_batches=8,
        drop_remainder=drop_reminder))
    return dataset

  return input_fn

def model_fn_builder(init_checkpoint, learning_rate, num_train_steps, use_tpu):

  def model_fn(features, labels, mode, params):

    input = features["input"]
    label = features["label"]
    anomaly = features["anomaly"]

    is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False

    #conv1d_act_fn=tf.nn.relu,
    #conv1d_act_fn=tf.math.softplus,
    #conv1d_act_fn=tf.nn.leaky_relu,

    model = MtadGat(input,
      label=label,
      conv1d_act_fn=tf.nn.relu,
      d0=params["conv1d_filter_width"],
      d1=params["GRU_hidden_size"],
      d2=params["fc_hidden_size"],
      d3=params["VAE_latent_space_dimension"],
      gamma=params["gamma"],
      tc_act_fn=tf.nn.relu,
      gru_act_fn=tf.math.tanh,
      initializer_range=params["initializer_range"],
      dropout_prob=params["dropout_prob"],
      is_training=is_training)

    #(A) --> (1)
    total_loss = tf.reduce_mean(model.per_example_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
      #
      # ServerMachineDataset is ~28378, 100 epochs, batch is 128 large, this is because of reparameterisation requitrement, sampling
      #steps: (28378/128)*100~22100
      #1000 is when change is occuring in case of staircase, start 1e-3 as in papare and goal is 1e-6 at the end, hopefully it will be well trained by this time
      #latest : 1e-4 --> 1e-6 40000 ~ 200 ecpochs
      #1e-4 * np.power(0.895, 40000 / 1000) --> 1.1828274988827724e-06, at 46k loss is -40,000

      #calculated_learning_rate = tf.math.maximum(learning_rate * tf.math.pow(0.895, tf.cast(tf.compat.v1.train.get_global_step() / 1000, tf.float32)), 5e-6)

      calculated_learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, tf.compat.v1.train.get_global_step(), 1000, 0.895, staircase=False)

      effective_learning_rate = calculated_learning_rate

      tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

      tf.logging.info("Trainable Variables")
      for i, v in enumerate(tvars):
        tf.logging.info("{}: {}".format(i, v))

      grads = tf.gradients(total_loss, tvars, name='gradients')

      #for index in range(len(grads)):
      #  if grads[index] is not None:
      #    gradstr = "\n g_nan/g_inf/v_nan/v_inf/guid/grad [%i] | tvar [%s] =" % (index, tvars[index].name) 
      #    grads[index] = tf.Print(grads[index], [tf.reduce_any(tf.is_nan(grads[index])), tf.reduce_any(tf.is_inf(grads[index])), tf.reduce_any(tf.is_nan(tvars[index])), tf.reduce_any(tf.is_inf(tvars[index])), guid, grads[index], tvars[index]], gradstr, summarize=100)

      if (FLAGS.clip_gradients > 0):
        gradients, _ = tf.clip_by_global_norm(grads, FLAGS.clip_gradients)
      else:
        gradients = grads

      optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=effective_learning_rate)
      if FLAGS.use_tpu:
        optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer) #, reduction=alignmnt_loss.Reduction.MEAN)

      train_op = optimizer.apply_gradients(zip(gradients, tvars), global_step=tf.compat.v1.train.get_global_step())

      training_hooks = None
      if not FLAGS.use_tpu:
        #logging_hook = tf.train.LoggingTensorHook({"loss": total_loss, "reconstruction_log_prob": tf.reduce_mean(model.reconstruction_log_probability), "-Dkl": tf.reduce_mean(model.minusDkl), "lr": effective_learning_rate, "step": tf.train.get_global_step()}, every_n_iter=1)
        logging_hook = tf.train.LoggingTensorHook({"loss": total_loss, "reconstruction_log_prob": total_loss, "-Dkl": total_loss, "lr": effective_learning_rate, "step": tf.train.get_global_step()}, every_n_iter=1)
        training_hooks = [logging_hook]

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode, predictions=None, loss=total_loss, train_op=train_op, eval_metrics=None,
        export_outputs=None, scaffold_fn=None, host_call=None, training_hooks=training_hooks,
        evaluation_hooks=None, prediction_hooks=None)

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, labels, logits, is_real_example):
        predictions = tf.cast(tf.math.greater(per_example_loss, params["threshold"]), tf.int32)
        accuracy = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions)
        precision = tf.compat.v1.metrics.precision(labels=labels, predictions=predictions)
        #precision = tf.Print(precision1, [precision1], "Precision")
        recall = tf.compat.v1.metrics.precision(labels=labels, predictions=predictions)
        #f1 = (2 * precision * recall) / (precision + recall + 1e-12)
        #    "eval_f1": f1,
        loss = tf.metrics.mean(values=per_example_loss, weights=None)
        return {
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_loss": loss
        }

      eval_metrics = (metric_fn,
                      [model.per_example_loss, anomaly, 0, 1])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=None)

    else:
      if params["prediction_task"] == "RMS_loss":
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={'RMS_loss': model.per_example_loss
                      })
      elif params["prediction_task"] == "EVALUATE":
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={'predicted': tf.cast(tf.math.greater(model.per_example_loss, params["threshold"]), tf.int32),
                       'label': anomaly
                    })
      else:
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={'anomaly': tf.cast(tf.math.greater(model.per_example_loss, params["threshold"]), tf.int32)
                    })
    return output_spec 

  return model_fn   

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  #tf.logging.set_verbosity(tf.logging.ERROR)

  tpu_cluster_resolver = None

  if FLAGS.use_tpu:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=None,
      job_name='worker',
      coordinator_name=None,
      coordinator_address=None,
      credentials='default', 
      service=None,
      discovery_url=None
    )

  tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
    iterations_per_loop=FLAGS.iterations_per_loop, 
    num_cores_per_replica=FLAGS.num_tpu_cores,
    per_host_input_for_training=True 
  )

  run_config = tf.compat.v1.estimator.tpu.RunConfig(
    tpu_config=tpu_config,
    evaluation_master=None,
    session_config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True), #, arithmetic_optimization=False),
    master=None,
    cluster=tpu_cluster_resolver,
    **{
      'save_checkpoints_steps': FLAGS.save_checkpoints_steps,
      'tf_random_seed': FLAGS.random_seed,
      'model_dir': FLAGS.output_dir, 
      'keep_checkpoint_max': FLAGS.keep_checkpoint_max,
      'log_step_count_steps': FLAGS.log_step_count_steps
    }
  )

  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
    model_fn=model_fn_builder(FLAGS.init_checkpoint, FLAGS.learning_rate, FLAGS.num_train_steps, FLAGS.use_tpu),
    use_tpu=FLAGS.use_tpu,
    train_batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.batch_size,
    predict_batch_size=FLAGS.batch_size,
    config=run_config,
    params={
        "conv1d_filter_width": FLAGS.conv1d_filter_width,
        "GRU_hidden_size": FLAGS.GRU_hidden_size,
        "fc_hidden_size": FLAGS.fc_hidden_size,
        "VAE_latent_space_dimension": FLAGS.VAE_latent_space_dimension,
        "gamma": FLAGS.gamma,
        "initializer_range": FLAGS.initializer_range,
        "num_features": FLAGS.num_features,
        "dropout_prob": FLAGS.dropout_prob,
        "use_tpu": FLAGS.use_tpu,
        "prediction_task": FLAGS.prediction_task,
        "threshold": FLAGS.threshold
    })

  if FLAGS.action == 'TRAIN':
    estimator.train(input_fn=make_input_fn(FLAGS.train_file, is_training=True, drop_reminder=True), max_steps=FLAGS.num_train_steps)
  
  if FLAGS.action == 'EVALUATE':
    eval_drop_remainder = True if FLAGS.use_tpu else False
    results = estimator.evaluate(input_fn=make_input_fn(FLAGS.test_file, is_training=False, drop_reminder=eval_drop_remainder), steps=None)

    for key in sorted(results.keys()):
      tf.logging.info("  %s = %s", key, str(results[key]))

  if FLAGS.action == 'PREDICT':
    predict_drop_remainder = True if FLAGS.use_tpu else False
    results = estimator.predict(input_fn=make_input_fn(FLAGS.test_file, is_training=False, drop_reminder=predict_drop_remainder))

    if FLAGS.prediction_task == 'RMS_loss':
      output_predict_file = os.path.join("./", "RMS_loss.csv")
      with tf.gfile.GFile(output_predict_file, "w") as writer:
        for prediction in results:
          writer.write(str(prediction["RMS_loss"]) + "\n")
    elif FLAGS.prediction_task == 'EVALUATE':
      labels = []
      anomalies = []
      for prediction in results:
        labels.append(prediction["label"])
        anomalies.append(prediction["predicted"])

      metrics = calculate_metrics(anomalies, labels, True)

      tf.logging.info("  %s = %s", "threshold", FLAGS.threshold)
      for key in sorted(metrics.keys()):
        tf.logging.info("  %s = %s", key, str(metrics[key]))
    else:
      output_predict_file = os.path.join("./", "Anomaly.csv")
      with tf.gfile.GFile(output_predict_file, "w") as writer:
        for prediction in results:
          writer.write(str(prediction["anomaly"]) + "\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='gs://anomaly_detection/mdat_gat/output',
            help='Model directrory in google storage.')
    parser.add_argument('--init_checkpoint', type=str, default=None,
            help='This will be checkpoint from previous training phase.')
    parser.add_argument('--train_file', type=str, default='gs://anomaly_detection/mtad_gat/data/train/machine-1-1.tfrecords',
            help='Train file location in google storage.')
    parser.add_argument('--test_file', type=str, default='gs://anomaly_detection/mtad_gat/data/test/machine-1-1.tfrecords',
            help='Test file location in google storage.')
    parser.add_argument('--dropout_prob', type=float, default=0.0,
            help='This used for all dropouts.')
    parser.add_argument('--num_train_steps', type=int, default=50000,
            help='Number of steps to run trainer.')
    parser.add_argument('--iterations_per_loop', type=int, default=1000,
            help='Number of iterations per TPU training loop.')
    parser.add_argument('--save_checkpoints_steps', type=int, default=1000,
            help='Number of tensorflow checkpoint to keep.')
    parser.add_argument('--log_step_count_steps', type=int, default=1000,
            help='Number of step to write logs.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=50,
            help='Number of tensorflow checkpoint to keep.')
    parser.add_argument('--batch_size', type=int, default=128,
            help='Batch size 128. VAE require at least 100.')
    parser.add_argument('--dataset_reader_buffer_size', type=int, default=100,
            help='input pipeline is I/O bottlenecked, consider setting this parameter to a value 1-100 MBs.')
    parser.add_argument('--shuffle_buffer_size', type=int, default=29000,
            help='Items are read from this buffer.')
    parser.add_argument('--use_tpu', default=False, action='store_true',
            help='Train on TPU.')
    parser.add_argument('--tpu', type=str, default='node-1-15-2',
            help='TPU instance name.')
    parser.add_argument('--num_tpu_cores', type=int, default=8,
            help='Number of cores on TPU.')
    parser.add_argument('--tpu_zone', type=str, default='us-central1-c',
            help='TPU instance zone location.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
            help='Optimizer learning rate.')
    parser.add_argument('--clip_gradients', type=float, default=-1.,
            help='Clip gradients to deal with explosive gradients.')
    parser.add_argument('--random_seed', type=int, default=123,
            help='Random seed to initialize values in a grath. It will produce the same results only if data and grath did not change in any way.')
    parser.add_argument('--initializer_range', type=float, default=0.02,
            help='.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--action', default='PREDICT', choices=['TRAIN','EVALUATE','PREDICT'],
            help='An action to execure.')
    parser.add_argument('--prediction_task', default=None, choices=['RMS_loss', 'EVALUATE'],
            help='Values to predict.')
    parser.add_argument('--restore', default=False, action='store_true',
            help='Restore last checkpoint.')
    parser.add_argument('--conv1d_filter_width', type=int, default=7,
            help='kernel size of 1D convolution for first conv1d layer')
    parser.add_argument('--GRU_hidden_size', type=int, default=300,
            help='GRU layer hidden size.')
    parser.add_argument('--fc_hidden_size', type=int, default=300,
            help='3fc layer hidden size.')
    parser.add_argument('--VAE_latent_space_dimension', type=int, default=18,
            help='Latent space dimension of the VAE model.')
    parser.add_argument('--gamma', type=float, default=0.8,
            help='Hyperparameter to combine multiple inference scores.')
    parser.add_argument('--window_size', type=int, default=100,
            help='Time series window size.')
    parser.add_argument('--num_features', type=int, default=38,
            help='Computer instance metrics.')
    parser.add_argument('--threshold', type=float, default=None,
            help='Anomaly cut-off threshold. It is different per model. POT model calculates this.')

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.threshold is None and (FLAGS.action== "EVALUATE" or FLAGS.action == "PREDICT" and FLAGS.prediction_task != "RMS_loss"):
      tf.logging.error("EVAL and PREDICT need threshold value")
      sys.exit()

    main()
