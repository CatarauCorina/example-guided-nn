import tensorflow as tf
import numpy as np

sess = tf.Session()


def complex_learn():
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    sess.run(tf.global_variables_initializer())
    linear_model = W * x + b

    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    print(sess.run(loss, {x: x_train, y: y_train}))

    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))


def simple_learn():
    features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4,
                                                  num_epochs=1000)
    estimator.fit(input_fn=input_fn, steps=1000)
    print(estimator.evaluate(input_fn=input_fn))


def model(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b
    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))

    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                         tf.assign_add(global_step, 1))
    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality.

    return tf.contrib.learn.ModelFnOps(
            mode=mode, predictions=y,
            loss=loss,
            train_op=train)


def custom_model():
    estimator = tf.contrib.learn.Estimator(model_fn=model)
    # define our data set
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

    # train
    estimator.fit(input_fn=input_fn, steps=1000)
    # evaluate our model
    print(estimator.evaluate(input_fn=input_fn, steps=10))


def main():
    simple_learn()
    custom_model()


if __name__ == '__main__':
    main()