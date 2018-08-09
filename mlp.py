import tensorflow as tf

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder("float")

def multilayer_perceptron(x, weights, biases, keep_prob):
    loss = tf.losses.mean_squared_error()
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.matmul(layer_1, weights['h2']) + biases['out']
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    layer_3 = tf.matmul(layer_2, weights['h3']) + biases['out']
    layer_3 = tf.nn.sigmoid(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


def main():


if __name__ == '__main__':
    main()