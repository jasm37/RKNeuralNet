import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


class RungeKuttaNN:
    """__Runge Kutta integrator neural network__
    Given the ODE: y'=f(y,x), with y(0)=f_t_init
    This function learns the 4th Order Runge Kutta integrator
    given the data at many time steps
    *Data must be fed with the method start_train
    """
    def __init__(self, timestep, t_init, t_end):
        """Constructor

        :param n_oppar = dim(x), number of operating parameters
        :param n_var = dim(y), number of variables
        :param timestep: time step length
        :param t_init: initial time
        :param t_end: end time
        """

        # Time step parameters
        self.timestep = timestep
        self.t_init = t_init
        self.t_end = t_end

        # Number of variables and operating parameters
        self.n_oppar = 0
        self.n_var = 0
        self.n_inputs = 0 # To be self.n_oppar + self.n_var

        ## Fixed parameters: can be changed after initialising
        # Number of neurons in each hidden layer
        self.n_hidden1 = 20
        self.n_hidden2 = 20

        # Batch parameters for training
        self.n_batches = 10
        self.bsize = 0
        self.train_epochs = 900

        # Display step
        self.dstep = 50

        # Learning rate
        self.lr = 0.0001

        # Percentage of data to be used as test
        self.test_ratio = 0.1

        # Error bound to stop iterations
        self.bound = 0.000005

        # Percentage of data to be used to test
        self.test_prop = 0.2

        # Stored arrays of neural net: weights and biases(numeric values)
        self.w1 = []
        self.w2 = []
        self.w3 = []

        self.b1 = []
        self.b2 = []
        self.b3 = []

        # Tensorflow: stored tensor arrays for neural networks
        self.tf_w1 = []
        self.tf_w2 = []
        self.tf_w3 = []

        self.tf_b1 = []
        self.tf_b2 = []
        self.tf_b3 = []

        # Main data array
        self.input_x = []   # Input operating parameters array
        self.input_y = []   # Input variables array
        self.output = []    # Output array
        self.data = []  # Joint input data

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        tf.reset_default_graph()

    def _tf_act_fun(self, y):
        """Tensorflow activation function
        So far only uses sigmoid, tanh and ReLU

        :param y: tensor to be fed into activation function
        :return: f(y) where f is the activation function
        """
        #return tf.nn.tanh(y)
        return tf.nn.relu(y)
        #return tf.nn.sigmoid(y)

    def _tf_predict_dy(self, y, x):
        """Tensorflow predict F(y,x) as in y' = dy/dt = F(y,x)

        :param y: tensor containing variable vector
        :param x: tensor containing operating parameters
        :return: F(y,x) according to the neural network approximation
        """
        stack = tf.concat(values=[y, x], axis=1)
        layer_1 = tf.matmul(stack, self.tf_w1) + self.tf_b1
        layer_1 = self._tf_act_fun(layer_1)
        layer_2 = tf.matmul(layer_1, self.tf_w2) + self.tf_b2
        layer_2 = self._tf_act_fun(layer_2)
        out = tf.matmul(layer_2, self.tf_w3) + self.tf_b3
        return out

    def _tf_forward_pass(self, y, x):
        """Tensorflow forward pass along the RKNN

        :param y: tensor containing variable vector at time step t
        :param x: tensor containing operating parameters
        :return:
        """
        # y is the variable vector,
        # x are the operating parameters(as a vector)
        # dt is the time step
        dt = self.timestep
        k1 = dt * self._tf_predict_dy(y, x)
        k2 = dt * self._tf_predict_dy(y + k1 / 2, x)
        k3 = dt * self._tf_predict_dy(y + k2 / 2, x)
        k4 = dt * self._tf_predict_dy(y + k3, x)
        out = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        return out

    def _act_fun(self, y):
        """Activation function for prediction
        So far only using sigmoid, tanh and ReLU
        :param y: variable vector to be fed into activation function
        :return: f(y) where f is the activation function
        """
        #return np.tanh(y)
        #return 1.0/(1 + np.exp(-y))
        return np.fmax(0.0, y)

    def _predict_dy(self, y, x):
        """Predict F(y,x) as in y' = dy/dt = F(y,x)

        :param y: variable vector
        :param x: operating parameters
        :return: F(y,x) according to the trained neural network
        """
        z = np.hstack((y, x))
        layer_1 = z @ self.w1 + self.b1
        layer_1 = self._act_fun(layer_1)
        layer_2 = layer_1 @ self.w2 + self.b2
        layer_2 = self._act_fun(layer_2)
        out = layer_2 @ self.w3 + self.b3
        return out

    def predict(self, y, x):
        """Forward pass along the RKNN or prediction y(t) -> y(t+1)

        :param y: variable vector at time step t ( or y(t) )
        :param x: operating parameters at time step t
        :return: variable vector at time step t+1 ( or y(t+1) )
        """
        dt = self.timestep
        k1 = dt * self._predict_dy(y, x)
        k2 = dt * self._predict_dy(y + k1 / 2, x)
        k3 = dt * self._predict_dy(y + k2 / 2, x)
        k4 = dt * self._predict_dy(y + k3, x)
        out = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return out

    def predict_from_samples(self, y, x, num_it):
        """Multiple predict's: long time prediction

        :param y: variable vector at the FIRST time step
        :param x: operating parameter at first time step
        :param num_it: number of iteration desired to compute y_1, y_2 .. y_{num_it}
        :return: vector containing future time steps of length num_it
        """
        current_y = y
        current_x = x[0]
        pred_list = y
        for it in range(num_it - 1):
            pred_var = self.predict(current_y, current_x)
            pred_list = np.vstack((pred_list, pred_var))
            current_y = pred_var
            current_x = x[it+1]
        return pred_list

    def start_train(self, data_y, data_x, output_y):
        """Start training of neural network
        Neural network approximates RK time stepper as:  F( y(t), x ) = y(t+1)

        :param data_y: array of shape (#snapshots, #variables) containing input variables y(t)
        :param data_x: array of shape (#snapshots, #parameters) containing input operating parameters x(t)
        :param output_y: array of shape (#samples, #variables) containing output y(t+1)
        """
        # Basic assertions
        assert (data_y.shape == output_y.shape), \
            "Input variable shape (%f,%f) does not match output dimension (%f,%f)" \
            %(data_y.shape[0], data_y.shape[1], output_y.shape[0], output_y.shape[1])

        assert (data_y.shape[0] == data_x.shape[0]), \
            "Number of variable snapshots %f does not match number of parameter snapshots %f" \
            %(data_y.shape[0], data_x.shape[0])

        assert (self.test_ratio <= 1.0), "Test ratio %f should be smaller than 1" % self.test_ratio

        # Store shapes
        if data_y.ndim == 1:
            data_y = data_y.reshape((data_y.shape[0], 1))
        if data_x.ndim == 1:
            data_x = data_x.reshape((data_x.shape[0], 1))

        self.n_var = data_y.shape[1]
        self.n_oppar = data_x.shape[1]
        self.n_inputs = self.n_var + self.n_oppar

        # Store main arrays
        self.output = output_y
        self.input_y = data_y
        self.input_x = data_x

        # Data has shape (#snapshots, #vars)
        n_samples = self.input_y.shape[0]
        self.data = np.hstack((data_y, data_x))

        joint_input = np.hstack((self.input_y, self.input_x))

        train_input, test_input, \
        train_output, test_output = train_test_split(joint_input, self.output,
                                                     test_size=int(n_samples * self.test_ratio), random_state=42)

        n_train = train_input.shape[0]
        test_y = test_input[:, :self.n_var]
        test_x = test_input[:, self.n_var:]

        # If batch size is given split data according to this number, otherwise use number of batches
        if self.bsize != 0:
            self.n_batches = int(n_train / self.bsize)
        else:
            self.bsize = int(n_train / self.n_batches)

        # Leftover of splitting data in batches
        leftover = n_train - self.n_batches * self.bsize

        # Tensorflow input and output variables
        x = tf.placeholder("float", [None, self.n_oppar], name="OperatingParameters")
        y = tf.placeholder("float", [None, self.n_var], name="ODEInputVariable")
        z = tf.placeholder("float", [None, self.n_var], name="ODEOutputVariable")

        # Tensroflow arrays to train
        self.tf_w1 = tf.get_variable(name="w1", shape=(self.n_inputs, self.n_hidden1),
                                     initializer=tf.contrib.layers.xavier_initializer())
        self.tf_w2 = tf.get_variable(name="w2", shape=(self.n_hidden1, self.n_hidden2),
                                     initializer=tf.contrib.layers.xavier_initializer())
        self.tf_w3 = tf.get_variable(name="w3", shape=(self.n_hidden2, self.n_var),
                                     initializer=tf.contrib.layers.xavier_initializer())

        self.tf_b1 = tf.Variable(tf.zeros([self.n_hidden1]), "b1")
        self.tf_b2 = tf.Variable(tf.zeros([self.n_hidden2]), "b2")
        self.tf_b3 = tf.Variable(tf.zeros([self.n_var]), "b3")

        # Graph output
        pred = self._tf_forward_pass(y, x)

        # Loss function and optimizer
        pow_diff = tf.pow(pred - z, 2)
        loss = tf.reduce_sum(pow_diff)

        # Uncomment following lines to use regularisers
        '''
        # Regularisers:
        reg_losses = tf.nn.l2_loss(self.tf_w1) + tf.nn.l2_loss(self.tf_w2) + \
                     tf.nn.l2_loss(self.tf_w3)
        reg_weight = 0.001
        loss += reg_weight*reg_losses
        '''
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        # Hardcoded parameters:
        epoch = 0
        train_loss = 50

        # Start training session
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            while train_loss > self.bound and epoch < self.train_epochs:
                epoch += 1
                # Shuffle training samples
                rand_index = np.random.permutation(n_train)
                train_y = train_input[rand_index, :self.n_var].reshape((n_train, self.n_var))
                train_x = train_input[rand_index, self.n_var:].reshape((n_train, self.n_oppar))
                train_out = train_output[rand_index].reshape((n_train, self.n_var))
                for i in range(self.n_batches):
                    # Group shuffled samples in batches
                    data_in_y = train_y[i * self.bsize:(i + 1) * self.bsize]
                    data_in_x = train_x[i * self.bsize:(i + 1) * self.bsize]
                    data_out = train_out[i * self.bsize:(i + 1) * self.bsize]
                    values = {y: data_in_y,
                              x: data_in_x,
                              z: data_out}
                    sess.run(optimizer, feed_dict=values)

                # If there was a leftover in the batch grouping then train on these
                if leftover != 0:
                    data_in_y = train_y[-leftover:]
                    data_in_x = train_x[-leftover:]
                    data_out = train_output[-leftover:]
                    values = {y: data_in_y,
                              x: data_in_x,
                              z: data_out}
                    sess.run(optimizer, feed_dict=values)

                # Check Test and Train losses every self.dstep epochs
                if epoch % self.dstep == 0:
                    values = {y: test_y.reshape((test_y.shape[0], self.n_var)),
                              x: test_x.reshape((test_x.shape[0], self.n_oppar)),
                              z: test_output.reshape((test_output.shape[0], self.n_var))}
                    test_loss = sess.run(loss, feed_dict=values)
                    print('Test Loss at step %s: \t%s' % (epoch, test_loss))

                    values = {y: train_y.reshape((train_y.shape[0], self.n_var)),
                              x: train_x.reshape((train_x.shape[0], self.n_oppar)),
                              z: train_out.reshape((train_out.shape[0], self.n_var))}
                    train_loss = sess.run(loss, feed_dict=values)
                    print('Train Loss at step %s: \t%s' % (epoch, train_loss))

                # Store trained arrays
                self.w1, self.w2, self.w3, self.b1, self.b2, self.b3 = sess.run([self.tf_w1, self.tf_w2, self.tf_w3,
                                                                                 self.tf_b1, self.tf_b2, self.tf_b3])


