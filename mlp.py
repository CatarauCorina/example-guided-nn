import tensorflow as tf


def multilayer_perceptron(x, weights, biases, keep_prob):
    loss = tf.losses.mean_squared_error()
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


def main():


if __name__ == '__main__':
    main()