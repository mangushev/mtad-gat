
#TODO:
#sampling N(1,1)
#last feature probability change to combined feature probability

import tensorflow as tf

import sys
import math
import six

import numpy as np

def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)

#inputs: batch_shape + [in_width, in_channels]
#filter: [filter_width, in_channels, out_channels]
def conv1d_layer(inputs, filter_width, in_channels, out_channels, padding, activation, initializer, trainable=True, name="conv"):
  with tf.compat.v1.variable_scope(name):
    filter = tf.compat.v1.get_variable(initializer=initializer, shape=[filter_width, in_channels, out_channels], trainable=trainable, name='filter')
    conv = tf.nn.conv1d(inputs, filter, [1], padding=padding, name="conv")
    bias = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[out_channels], trainable=trainable, name='bias')
    conv_bias = tf.nn.bias_add(conv, bias, name='conv_bias')
    if activation:
      conv_bias_relu = activation(conv_bias, name='conv_bias_relu')
      return conv_bias_relu
    return conv_bias

def dense_layer(input_tensor, hidden_size, activation, initializer, name="dense"):
  with tf.compat.v1.variable_scope(name):
    input_shape = get_shape_list(input_tensor)

    if len(input_shape) != 2 and len(input_shape) != 3:
      assert_rank(tensor, expected_rank, tensor.name)

    batch_size = input_shape[0]
    if len(input_shape) == 3:
      seq_length = input_shape[1]
      input_width = input_shape[2]
      x = tf.reshape(input_tensor, [-1, input_width])
    else:
      input_width = input_shape[1]
      x = input_tensor

    w = tf.compat.v1.get_variable(initializer=initializer, shape=[input_width, hidden_size], name="w")
    z = tf.matmul(x, w, transpose_b=False)
    b = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[hidden_size], name="b")
    y = tf.nn.bias_add(z, b)
    if (activation):
      y = activation(y)

    if len(input_shape) == 3:
      return tf.reshape(y, [batch_size, seq_length, hidden_size])

    return y

def layer_norm(input_tensor, trainable=True, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  #return tf.keras.layers.LayerNormalization(name=name,trainable=trainable,axis=-1,epsilon=1e-14,dtype=tf.float32)(input_tensor)
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, trainable=trainable, scope=name)

def print_shape(tensor, rank, tensor_name):
  return tensor
  tensor_shape = get_shape_list(tensor, expected_rank=rank)
  return tf.Print(tensor, [tensor_shape], tensor_name, summarize=8)

#expected input is batch, num_nods, node_size
#   N = number of nodes
#   D = node size
def GAT(input_tensor, num_nodes, node_size, initializer_range, gat_id):
  with tf.variable_scope("gat_%d" %gat_id):

    #[D]
    gat_weights = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
          shape=[2 * node_size, 1], name='gat_weights')

    #[A, N, D] --> [A, N, 1, D]
    i_dim1 = tf.reshape(input_tensor, [-1, num_nodes, 1, node_size], name='i_dim1')
    #[A, N, 1, D] --> [A, N, N, D]
    i_dim = tf.tile(i_dim1, [1, 1, num_nodes, 1], name='i_dim')
    j_dim = tf.transpose(i_dim, [0, 2, 1, 3], name='j_dim')

    #[A, N, N, D] + [A, N, N, D] --> [A, m, m, 2D]
    ij_concat_p = tf.concat([i_dim, j_dim], axis=-1, name='ij_concat') 
    ij_concat = print_shape(ij_concat_p, 4, "ij_concat shape")

    #[A, 1, 1, 2D, 1] --> [A, N, N, 2D, 1]
    ij_concat_1 = tf.reshape(ij_concat, [-1, num_nodes, num_nodes, 2 * node_size, 1], name='ij_concat_1')

    #[2D, 1]' . [A, N, N, 2D, 1] --> [A, N, N, 1, 1]
    mm_p = tf.matmul(gat_weights, ij_concat_1, transpose_a=True, name='mm')
    mm = print_shape(mm_p, 5, "mm shape")
 
    #[A, N, N, 1, 1] --> [A, N, N]
    mm = tf.squeeze(mm, axis=-1, name='mm_squeeze1')
    mm = tf.squeeze(mm, axis=-1, name='mm_squeeze2')

    mm = tf.keras.layers.LayerNormalization(axis=[1,2],epsilon=1e-14,dtype=tf.float32)(mm)

    e_ij = tf.nn.leaky_relu(mm, alpha=0.2, name='e_ij')

    #[A, N, N] --> [A, N, N]
    alpha_ij_p = tf.nn.softmax(e_ij, axis=1, name='alpha_ij')
    alpha_ij = print_shape(alpha_ij_p, 3, "alpha_ij shape")

    #[A, N, D] --> [A, N, 1, D]
    x1 = tf.reshape(input_tensor, [-1, num_nodes, 1, node_size], name='x1')
    #[A, N, 1, D] --> [A, 1, N, D]
    x2 = tf.transpose(x1, [0, 2, 1, 3], name='x2')
    #[A, 1, N, D] --> [A, N, N, D]
    x3 = tf.tile(x2, [1, num_nodes, 1, 1], name='x3')
    #[A, N, N] --> [A, N, N, 1]
    alpha_ij_1 = tf.reshape(alpha_ij, [-1, num_nodes, num_nodes, 1], name='alpha_ij_1')
    #[A, N, N, 1] * [A, N, N, D] --> reduce_sum axis=2 --> [A, N, D]
    h_i_p = tf.math.sigmoid(tf.reduce_sum(alpha_ij_1 * x3, axis=2), name='h_i')
    h_i = print_shape(h_i_p, 3, "h_i shape")

    return h_i

#returns probability of every feature
def pdf(data, mu, var):
  import math

  pi = tf.constant(math.pi, dtype=tf.float32)
  #var usually 1e-1 - 1e-2
  epsilon = tf.constant(1e-14, dtype=tf.float32)

  #p get sometimes 1e-42 (>max float32)
  return tf.math.exp(-tf.math.pow(data - mu, 2, name='p2') / (2.0 * (var + epsilon))) / tf.math.sqrt(2.0 * pi * (var + epsilon), name='p1')

  p1 = tf.reduce_sum(tf.math.log(p0 + tf.constant(1e-35, dtype=tf.float32)), axis=-1, keepdims=False)
  p2 = tf.clip_by_value(p1, tf.constant(math.log(1e-35), dtype=tf.float32), tf.constant(math.log(1e+35), dtype=tf.float32))
  p3 = tf.math.exp(p2)
  return tf.clip_by_value(p3, tf.constant(1e-35, dtype=tf.float32), tf.constant(1e+35, dtype=tf.float32))

class MtadGat(object):
  #   A - batch size
  #   k m = number of variables or features (metrics for computer instance)
  #   n w = window size
  #   d0 k0 = conv1d filter width 
  #   d1 k1 = hidden dimension of the GRU layer
  #   d2 k2 = hidden dimension of fully connected layers
  #   d3 = latent space dimension of the VAE model
  #   gamma = hyperparameter to combine multiple inference scores
  def __init__(self,
               input_tensor,
               label,
               conv1d_act_fn=tf.nn.relu,
               d0=7,
               d1=300, #k1
               d2=300, #k2
               d3=300,
               gamma=0.8,
               tc_act_fn=tf.nn.relu,
               gru_act_fn=tf.math.tanh,
               initializer_range=0.02,
               dropout_prob=0.1,
               is_training=True):
    #[A, w, k/m]
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    window_size = input_shape[1]
    num_features = input_shape[2]

    if is_training == False:
       dropout_prob = 0.0

    input_t = print_shape(input_tensor, 3, "input_tensor shape")
    #input_t = tf.Print(_input_t, [_input_t], "_input_t", summarize=1000)

    #conv1d_act_fn=tf.nn.relu
    #conv1d_act_fn=tf.math.softplus
 
    #conv1d_act_fn=tf.nn.leaky_relu

    #1). 1D convolution to alleviate the possible noise effects
    #[A, w, k/m] --> [A, w, k/m]
    with tf.variable_scope("alleviate_noise"):
      conv1d_output1 = conv1d_layer(input_t, d0, num_features, num_features, "SAME", 
        None, create_initializer(initializer_range), name="conv_1")
      conv1d_output1 = layer_norm(conv1d_output1)
      #conv1d_output1 = conv1d_act_fn(conv1d_output1)

      xx0 = tf.math.exp(conv1d_output1) + 1
      xx1  = tf.clip_by_value(xx0, tf.constant(math.log(1e-35), dtype=tf.float32), tf.constant(math.log(1e+35), dtype=tf.float32))
      conv1d_output1 = tf.math.log(xx1)

      #conv1d_output_p = dropout(conv1d_output1, dropout_prob)
      _conv1d_output = print_shape(conv1d_output1, 3, "conv1d_output shape")
      conv1d_output = tf.Print(_conv1d_output, [_conv1d_output], "_conv1d_output", summarize=1000)

    conv1d_output = input_t

    #2). feed conv1d output into feature and time oriented GAT and concatenate GAT outputs with original conv1d
    #[A, n/w, k/m] --> [A, n/w, k/m]
    with tf.variable_scope("processing1"):

      #feature-oriented GAT 
      #[A, n/w, k/m] --> [A, k/m, n/w]
      feature_gat_input = tf.transpose(conv1d_output, [0, 2, 1], name='feature_gat_input')
      feature_gat_output1 = GAT(feature_gat_input, num_features, window_size, initializer_range, 1)
      #[A, k/m, n/w] --> [A, n/w, k/m]
      _feature_gat_output = tf.transpose(feature_gat_output1, [0, 2, 1], name='feature_gat_output')
      feature_gat_output = tf.Print(_feature_gat_output, [_feature_gat_output], "_feature_gat_output", summarize=1000)

      #[A, n/w, k/m] --> [A, n/w, k/m]
      _temporal_gat_output = GAT(conv1d_output, window_size, num_features, initializer_range, 2)
      temporal_gat_output = tf.Print(_temporal_gat_output, [_temporal_gat_output], "_temporal_gat_output", summarize=1000)

      #[A, n/w, k/m] concat [A, n/w, k/m] concat [A, n/w, k/m] --> [A, n/w, 3k/m]
      concat_output1 = tf.concat([feature_gat_output, temporal_gat_output, conv1d_output], 2, name='concat_all')
      #concat_output1 = tf.concat([feature_gat_output, feature_gat_output, conv1d_output], 2, name='concat_all')
      #concat_output1 = tf.concat([conv1d_output, conv1d_output, conv1d_output], 2, name='concat_all')
      concat_output = print_shape(concat_output1, 3, "concat_output shape")
      #concat_output = tf.Print(_concat_output, [_concat_output], "_concat_output", summarize=1000)

    #3). GRU for long temporal
    #[A, n/w, 3k/m] --> [A, n/w, d1/k1]
    with tf.name_scope('long_temporal') as scope:
      #[A, n/w, 3k/m] -> [n/w, A, 3k/m]
      step_inputs = tf.transpose(concat_output, [1, 0, 2], name='step_inputs')

      with tf.compat.v1.variable_scope('gru_cells'):

        gru_cell = tf.keras.layers.GRUCell(d1, activation=gru_act_fn, kernel_initializer=tf.compat.v1.initializers.he_normal(), recurrent_initializer=tf.orthogonal_initializer, bias_initializer=tf.zeros_initializer, dropout=dropout_prob) #, name='gru_cell')

        step = tf.constant(0, name='step')
        output_ta = tf.TensorArray(size=window_size, dtype=tf.float32)
        initial_state = tf.zeros((batch_size, d1), dtype=tf.float32, name='initial_state')

        def cond(step, output_ta, state):
          return tf.less(step, window_size)

        def body(step, output_ta, state):
          input = tf.slice(step_inputs, [step, 0, 0], [1, -1, -1], name='slice')
          input_one = tf.squeeze(input, axis=0, name='squeeze')
          output,state = gru_cell(input_one, state, training=is_training)

          return (step + 1, output_ta.write(step, output), state)

        _, output_ta_final, state = tf.while_loop(cond, body, [step, output_ta, [initial_state]], name='gru_loop')

      #time, batch, features: add outputs as per article
      time_gru_output = output_ta_final.stack(name='stack_ta')
   
      #[n/w, A, d1] -> [A, n/w, d1]
      gru_output = tf.transpose(time_gru_output, [1, 0, 2])
      #gru_output = tf.Print(_gru_output, [_gru_output], "_gru_output", summarize=1000)

    #4). three fully connected layers
    #[A, d1] --> [A, d2]
    with tf.variable_scope("layer_3fc"):
      layer_output = gru_output[:, -1, :]
      for i in range(3):
        with tf.variable_scope("fc_%d" %i):
          layer_output = dense_layer(layer_output, d2, activation=None, initializer=create_initializer(initializer_range))
          #layer_output = layer_norm(layer_output)
          layer_output = tf.nn.relu(layer_output)
          #layer_output = dropout(layer_output, dropout_prob)
  
    #[A, d2] --> [A, k/m]
    next_feature = dense_layer(layer_output, num_features, activation=None, initializer=create_initializer(initializer_range))

    #[A, k/m] --> [A]
    self._forecasting_loss = tf.math.sqrt(tf.reduce_sum(tf.math.squared_difference(label, next_feature), axis=-1))
    #self._forecasting_loss = tf.Print(_forecasting_loss, [_forecasting_loss], "_forecasting_loss", summarize=1000)
 
    #5). Reconstruction - VAE
    #[A, n/w, d1] --> [A, n/w, k/m]
    with tf.variable_scope("reconstruction"):
      #encoder_input1 = tf.transpose(gru_output, [0, 2, 1])
      #encoder_input = tf.reshape(encoder_input1, [batch_size, d1, window_size], name='encoder_input')

      #going back with data dimention to original num_features to be able to get recovery probability per feature
      #[A, n/w, d1] --> [A, n/w, k/m]
      encoder_input = dense_layer(gru_output, num_features, activation=None, initializer=create_initializer(initializer_range))
      #encoder_input = layer_norm(encoder_input)
      encoder_input = tf.nn.relu(encoder_input)
      #encoder_input = tf.nn.leaky_relu(encoder_input, alpha=0.2)
      #_encoder_input = tf.math.softplus(encoder_input)
      #encoder_input = dropout(encoder_input, dropout_prob)

      #encoder_input = tf.Print(_encoder_input, [_encoder_input], "_encoder_input", summarize=1000)

      z_size = 18

      #calulate phi parameters
      #[A, n/w, k/m] --> [A, n/w, z]
      encoder_h = dense_layer(encoder_input, z_size, activation=tf.nn.tanh, initializer=create_initializer(initializer_range), name="encoder_h")
      #encoder_h = tf.Print(_encoder_h, [_encoder_h], "_encoder_h", summarize=1000)

      #[A, n/w, 3] --> [A, n/w, z]
      encoder_mu = dense_layer(encoder_h, z_size, activation=None, initializer=create_initializer(initializer_range), name="encoder_mu")
      #encoder_mu = tf.Print(_encoder_mu, [_encoder_mu], "_encoder_mu", summarize=1000)
      #[A, n/w, 3] --> [A, n/w, z]
      encoder_log_variance = dense_layer(encoder_h, z_size, activation=None, initializer=create_initializer(initializer_range), name="encoder_variance")
      #encoder_log_variance = tf.Print(_encoder_log_variance, [_encoder_log_variance], "_encoder_variance", summarize=1000)

      encoder_variance = tf.math.exp(encoder_log_variance)
      #encoder_variance = tf.Print(_encoder_variance, [_encoder_variance], "_encoder_variance", summarize=1000)

      encoder_scale = tf.math.sqrt(encoder_variance)
      #encoder_scale = tf.Print(_encoder_scale, [_encoder_scale], "_encoder_scale", summarize=1000)

      #[A, n/w, z]
      epsilon_sampler = tf.random.normal([batch_size, window_size, z_size], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, name='epsilon')

      #g(.) =location+scale*epsilon
      #use phi parameters
      z = encoder_mu + encoder_scale*epsilon_sampler

      #calulate teta parameters
      #[A, n/w, z] --> [A, n/w, k/m]
      decoder_h = dense_layer(z, num_features, activation=tf.nn.tanh, initializer=create_initializer(initializer_range), name="decoder_h")
      #decoder_h = tf.Print(_decoder_h, [_decoder_h], "_decoder_h", summarize=1000)
      #[A, n/w, k/m] --> [A, n/w, k/m]
      decoder_mu = dense_layer(decoder_h, num_features, activation=None, initializer=create_initializer(initializer_range), name="decoder_mu")
      #decoder_mu = tf.Print(_decoder_mu, [_decoder_mu], "_decoder_mu", summarize=1000)
      #[A, n/w, k/m] --> [A, n/w, k/m]
      decoder_log_variance = dense_layer(decoder_h, num_features, activation=None, initializer=create_initializer(initializer_range), name="decoder_variance")
      #decoder_log_variance = tf.Print(_decoder_log_variance, [_decoder_log_variance], "_decoder_variance", summarize=1000)

      decoder_variance = tf.math.exp(decoder_log_variance)
      #decoder_variance = tf.Print(_decoder_variance, [_decoder_variance], "_decoder_variance", summarize=1000)

      #[A, n/w, k/m], [A, n/w, k/m], [A, n/w, k/m] --> [A, n/w, k/m]
      # this is an estimate P teta given z for each individual feature
      feature_probability = pdf(encoder_input, decoder_mu, decoder_variance)
      #feature_probability = tf.Print(_feature_probability, [_feature_probability], "_feature_probability", summarize=1000)

      #[A, n/w, k/m] --> [A, n/w]
      # this is an estimate P teta given z total product
      reconstruction_log_probability = tf.reduce_sum(tf.math.log(feature_probability + tf.constant(1e-35, dtype=tf.float32)), axis=-1, keepdims=False)
      #reconstruction_log_probability = tf.Print(_reconstruction_log_probability, [_reconstruction_log_probability], "_reconstruction_log_probability", summarize=1000)
      self._reconstruction_log_probability = tf.reduce_sum(reconstruction_log_probability, axis=-1, keepdims=False)

      #[A, n/w, z] --> [A, n/w]
      #this is -Dkl formula
      minusDkl = tf.reduce_sum(((1 + tf.math.log(tf.math.square(encoder_scale))) - tf.math.square(encoder_mu) - tf.math.square(encoder_scale)) / 2, axis=-1, keepdims=False)
      #minusDkl = tf.Print(_minusDkl, [_minusDkl], "_minusDkl", summarize=1000)
      self._minusDkl = tf.reduce_sum(minusDkl, axis=-1, keepdims=False)

      #reconstraction loss is negated lower bound [ELBO]
      #[A, n/w], [A, n/w] --> [A]
      _reconstruction_loss1 = -(self._reconstruction_log_probability + self._minusDkl)

    #6). Combined per example loss
    #[A], [A] --> [A]
    self._reconstruction_loss = print_shape(_reconstruction_loss1, 1, "_reconstruction_loss shape")
    #self._reconstruction_loss = tf.Print(_reconstruction_loss2, [_reconstruction_loss2], "_reconstruction_loss", summarize=1000)
    #self._per_example_loss = self._forecasting_loss + _reconstruction_loss
    self._per_example_loss = self._forecasting_loss
    #self._per_example_loss = self._reconstruction_loss

    #7). inference score
    #[A, n/w, k/m] --> [A, k/m]
    last_feature_probability = feature_probability[:, -1, :]

    #[A, k/m], [A, k/m] --> [A, k/m]
    self._inference_score = tf.reduce_sum((tf.math.squared_difference(label, next_feature) + gamma*(1-last_feature_probability)) / (1 + gamma), axis=-1, keepdims=False)

  @property
  def forecasting_loss(self):
    return self._forecasting_loss

  @property
  def reconstruction_loss(self):
    return self._reconstruction_loss

  @property
  def reconstruction_log_probability(self):
    return self._reconstruction_log_probability

  @property
  def minusDkl(self):
    return self._minusDkl

  @property
  def per_example_loss(self):
    return self._per_example_loss

  @property
  def inference_score(self):
    return self._inference_score
