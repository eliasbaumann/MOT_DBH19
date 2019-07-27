import tensorflow as tf
import numpy as np

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

def sparse_cost_sensitive_loss (logits, labels, cost_matrix):
    batch_cost_matrix = tf.nn.embedding_lookup(cost_matrix, labels)
    eps = 1e-6
    probability = tf.clip_by_value(tf.nn.softmax(logits), eps, 1-eps)
    cost_values = tf.log(1-probability)*batch_cost_matrix
    loss = tf.reduce_mean(-tf.reduce_sum(cost_values, axis=1))
    return loss


def onehot(data,label_dict={'boat':1,'nature':0}):
    a = np.array([label_dict[k] for k in data])
    b = np.zeros((len(a),a.max()+1))
    b[np.arange(len(data)),a] = 1
    return b.astype(np.int32)


def weighted_ce(targets, logits, beta, name=None):
  """Computes a weighted cross entropy like in 
  http://www.vision.ee.ethz.ch/~cvlsegmentation/driu/data/paper/DRIU_MICCAI2016.pdf
  cross entropy is computed as follows:
    z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
  = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
  = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
  = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
  = (1 - z) * x + log(1 + exp(-x))
  = x - x * z + log(1 + exp(-x))
  """
  with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    targets = ops.convert_to_tensor(targets, name="targets")
    targets = tf.cast(targets,tf.float32)
    try:
      targets.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError(
          "logits and targets must have the same shape (%s vs %s)" %
          (logits.get_shape(), targets.get_shape()))
    targets = tf.math.add(targets,tf.keras.backend.epsilon())
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = array_ops.where(cond, logits, zeros)
    neg_abs_logits = array_ops.where(cond, -logits, logits)
    return tf.reduce_mean(tf.math.abs((math_ops.add(
        beta * (relu_logits - logits * targets), # false negatives
        (1-beta)*(math_ops.log1p(math_ops.exp(neg_abs_logits))), # false positives
        name=name))))
