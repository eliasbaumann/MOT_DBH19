import tensorflow as tf
import numpy as np

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
