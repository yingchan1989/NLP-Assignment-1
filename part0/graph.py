import numpy as np
import tensorflow as tf

class AddTwo(object):
    def __init__(self):
        # If you are constructing more than one graph within a Python kernel
        # you can either tf.reset_default_graph() each time, or you can
        # instantiate a tf.Graph() object and construct the graph within it.

        # Hint: Recall from live sessions that TensorFlow
        # splits its models into two chunks of code:
        # - construct and keep around a graph of ops
        # - execute ops in the graph
        #
        # Construct your graph in __init__ and run the ops in Add.
        #
        # We make the separation explicit in this first subpart to
        # drive the point home.  Usually you will just do them all
        # in one place, including throughout the rest of this assignment.
        #
        # Hint:  You'll want to look at tf.placeholder and sess.run.

        # START YOUR CODE
        self.x_ = tf.placeholder(tf.int64)
        self.y_ = tf.placeholder(tf.int64)
        self.summation = tf.add(self.x_, self.y_)
        pass

        # END YOUR CODE

    def Add(self, x, y):

        # START YOUR CODE

        sess = tf.Session()
        output = sess.run(self.summation, feed_dict={self.x_: x, self.y_: y})
        return output
        # END YOUR CODE


def affine_layer(hidden_dim, x, seed=0):
    # x: a [batch_size x # features] shaped tensor.
    # hidden_dim: a scalar representing the # of nodes.
    # seed: use this seed for xavier initialization.

    # START YOUR CODE

    input_dim = x.get_shape()

    w_ = tf.get_variable("w", shape=[input_dim[1], hidden_dim],
                        initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b_ = tf.get_variable("b", shape=[hidden_dim],
                        initializer=tf.constant_initializer(0.0))

    return tf.matmul(x, w_) + b_

    # END YOUR CODE

def fully_connected_layers(hidden_dims, x):
    # hidden_dims: A list of the width of the hidden layer.
    # x: the initial input with arbitrary dimension.
    # To get the tests to pass, you must use relu(.) as your element-wise nonlinearity.
    #
    # Hint: see tf.variable_scope - you'll want to use this to make each layer 
    # unique.

    # START YOUR CODE
    hidden_dim_length = len(hidden_dims)
    for i in range(0, hidden_dim_length):
        hidden_dim = hidden_dims[i]
        with tf.variable_scope(str(i)):
            x = tf.nn.relu(affine_layer(hidden_dim, x))
    return x

    # END YOUR CODE

def train_nn(X, y, X_test, hidden_dims, batch_size, num_epochs, learning_rate):
    # Train a neural network consisting of fully_connected_layers
    # to predict y.  Use sigmoid_cross_entropy_with_logits loss between the
    # prediction and the label.
    # Return the predictions for X_test.
    # X: train features
    # Y: train labels
    # X_test: test features
    # hidden_dims: same as in fully_connected_layers
    # learning_rate: the learning rate for your GradientDescentOptimizer.

    # Construct the placeholders.
    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.float32, shape=[None, X.shape[-1]])
    y_ph = tf.placeholder(tf.float32, shape=[None])
    global_step = tf.Variable(0, trainable=False)


    # Construct the neural network, store the batch loss in a variable called `loss`.
    # At the end of this block, you'll want to have these ops:
    # - y_hat: probability of the positive class
    # - loss: the average cross entropy loss across the batch
    #   (hint: see tf.sigmoid_cross_entropy_with_logits)
    #   (hint 2: see tf.reduce_mean)
    # - train_op: the training operation resulting from minimizing the loss
    #             with a GradientDescentOptimizer
    # START YOUR CODE

    if len(hidden_dims) == 0:
        input_dim = X.shape[1]
    else:
        input_dim = hidden_dims[-1]

    w_ = tf.Variable(tf.zeros([input_dim, 1], dtype=tf.float32), name="w")
    b_ = tf.Variable(0.0, dtype=tf.float32, name="b")

    neural_net = fully_connected_layers(hidden_dims, x_ph)

    logits_ = tf.squeeze(tf.matmul(neural_net, w_)) + b_
    y_hat = tf.sigmoid(logits_)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_hat, y_ph))
    optimizer_ = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer_.minimize(loss)

    # END YOUR CODE


    # Output some initial statistics.
    # You should see about a 0.6 initial loss (-ln 2).
    sess = tf.Session(config=tf.ConfigProto(device_filters="/cpu:0"))
    sess.run(tf.initialize_all_variables())
    global_step_value = sess.run(global_step)
    print 'Initial loss:', sess.run(loss, feed_dict={x_ph: X, y_ph: y})

    for var in tf.trainable_variables():
        print 'Variable: ', var.name, var.get_shape()
        print 'dJ/dVar: ', sess.run(
                tf.gradients(loss, var), feed_dict={x_ph: X, y_ph: y})

    for epoch_num in xrange(num_epochs):
        for batch in xrange(0, X.shape[0], batch_size):
            X_batch = X[batch : batch + batch_size]
            y_batch = y[batch : batch + batch_size]

            # Populate loss_value with the loss this iteration.
            # START YOUR CODE
            c, p, _ = sess.run([loss, y_hat, train_op],
                                  feed_dict={x_ph: X, y_ph: y})

            loss_value = c
            global_step_value = global_step_value + 1
            # END YOUR CODE
        if epoch_num % 300 == 0:
            print 'Step: ', global_step_value, 'Loss:', loss_value
            for var in tf.trainable_variables():
                print var.name, sess.run(var)
            print ''

    # Return your predictions.
    # START YOUR CODE
    p = sess.run(y_hat, feed_dict={x_ph: X_test})
    y_pred = p

    for i in range(len(p)):
        if p[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    return y_pred
    # END YOUR CODE
