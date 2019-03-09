import tensorflow as tf
import numpy as np

keep_prob = 1.0
initializer = tf.random_uniform_initializer(-1, 1)

def biLstm(x, weights, biases,num_layers,num_units):
    outputs = x
    #track through the layers
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer),reuse=tf.AUTO_REUSE):

            #forward cells
            cell_fw = tf.contrib.rnn.LSTMCell(num_units,initializer=initializer)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)
            
            #backward cells
            cell_bw = tf.contrib.rnn.LSTMCell(num_units,initializer=initializer)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)
                
            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw, outputs,dtype=tf.float32)
            state = last_state

    rnn_outputs_fw = tf.reshape(output_fw, [-1, num_units])
    rnn_outputs_bw = tf.reshape(output_bw, [-1, num_units])
    out_fw = tf.matmul(rnn_outputs_fw, weights['out']) + biases['out']
    out_bw = tf.matmul(rnn_outputs_bw, weights['out']) + biases['out']
    return np.add(out_fw,out_bw)
