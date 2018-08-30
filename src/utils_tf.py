import tensorflow as tf


def variable_summaries(var):
    '''
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    From https://www.tensorflow.org/get_started/summaries_and_tensorboard
    '''
    var_no_nan = tf.where(tf.is_nan(var), tf.zeros_like(var), var, name="remove_NaNs")
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var_no_nan)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var_no_nan - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var_no_nan))
        tf.summary.scalar('min', tf.reduce_min(var_no_nan))
        tf.summary.histogram('histogram', var_no_nan)


def resize_tensor_variable(sess, tensor_variable, shape):
    sess.run(tf.assign(tensor_variable, tf.zeros(shape), validate_shape=False))
