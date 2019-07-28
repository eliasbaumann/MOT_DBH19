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


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)
