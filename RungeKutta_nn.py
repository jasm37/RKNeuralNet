from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RungeKutta_nn():
    """
        Given the ODE: y'=f(y,x), with y(0)=f_t_init
        This function(using rknn) learns the 4th Order Runge Kutta integrator
        given the data at many timesteps
        :param n_var = dim(y), number of variables
        :param n_oppar = dim(x), number of operating parameters
        :param timestep = dt
        :param t_init initial time
        :param t_end end time
        Data must be fed with the method start_train
    """
    def __init__(self, n_oppar, n_var, timestep, t_init, t_end):
        self.n_hidden1 = 6
        self.n_hidden2 = 6
        self.timestep = timestep

        self.t_init = t_init
        self.t_end = t_end

        self.f_t_init = 0

        self.n_oppar = n_oppar
        self.n_var = n_var
        self.n_inputs = n_oppar + n_var

        # Batch size
        self.bsize = 10#200  # train_x.shape[0], 1, 2, 5, 10, 20k
        self.train_epochs = 900
        # Display step
        self.dstep = 50
        # Learning rate
        #self.lr = 0.000001#0.01#0.3#0.1#0.3#0.3 #TODO FOR ADADELTA OPT!!
        self.lr = 0.0001#0.0000025#0.0001#0.0000075##0.0000075#0.0001#0.00002#0.0001#0.0002 #TODO: FOR ADAM OPTIMIZER
        # Error bound to stop iterations
        self.bound = 0.000005#8#9#3#2.5#3.2#1.5#2
        self.dropout = 1

        self.output = []
        self.joint_input = []
        self.pred_list = []
        # Percentage of data to be used to test
        self.test_prop = 0.2

        # Stored arrays of neural net: weights and biases(numeric values)
        self.w1 = []
        self.w2 = []
        self.w3 = []

        self.b1 = []
        self.b2 = []
        self.b3 = []

        # Stored tensor arrays for neural networks(tensorflow tensors/variables)
        self.tf_w1 = []
        self.tf_w2 = []
        self.tf_w3 = []

        self.tf_b1 = []
        self.tf_b2 = []
        self.tf_b3 = []

        # Data properties
        self.input_x = []
        self.input_y = []
        self.data_batch = [] #complete dataset
        self.data_coeff = [] #complete POD coefficient representation
        #self.centered_data = 0 #centered dataset: subtracted the mean to all samples
        self.evectors = []
        self.data_list = [] #list of all datasets(for different parameters)
        self.data_mean = []  #mean of data_batch along time samples
        self.point_list = [] #vtk geometry list
        self.coeff_list = []
        self.heatload_list = []

    def tf_act_fun(self, x):
        #return tf.nn.tanh(x)
        return tf.nn.relu(x)
        # return tf.nn.sigmoid(x)

    def tf_predict_dy(self, y, x):
        # y is the variable vector,
        # x are the operating parameters(as a vector)
        stack = tf.concat(values=[y,x], concat_dim=1)
        layer_1 = tf.matmul(stack, self.tf_w1) + self.tf_b1
        layer_1 = self.tf_act_fun(layer_1)
        layer_2 = tf.matmul(layer_1, self.tf_w2) + self.tf_b2
        layer_2 = self.tf_act_fun(layer_2)
        out = tf.matmul(layer_2, self.tf_w3) + self.tf_b3
        return out

    def tf_forward_pass(self, y, x):
        # y is the variable vector,
        # x are the operating parameters(as a vector)
        # dt is the time step
        dt = self.timestep
        k1 = dt * self.tf_predict_dy(y, x)
        k2 = dt * self.tf_predict_dy(y + k1 / 2, x)
        k3 = dt * self.tf_predict_dy(y + k2 / 2, x)
        k4 = dt * self.tf_predict_dy(y + k3, x)
        out = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        return out

    def act_fun(self, x):
        #return np.tanh(x)
        return np.fmax(0.0, x)

    def predict_dy(self, y, x):
        # y is the variable vector,
        # x are the operating parameters(as a vector)
        z = np.hstack((y,x))
        layer_1 = z @ self.w1 + self.b1
        layer_1 = self.act_fun(layer_1)
        layer_2 = layer_1 @ self.w2 + self.b2
        layer_2 = self.act_fun(layer_2)
        out = layer_2 @ self.w3 + self.b3
        return out

    def predict(self, y, x):
        # y is the variable vector,
        # x are the operating parameters(as a vector)
        # dt is the time step
        dt = self.timestep
        k1 = dt * self.predict_dy(y, x)
        k2 = dt * self.predict_dy(y + k1 / 2, x)
        k3 = dt * self.predict_dy(y + k2 / 2, x)
        k4 = dt * self.predict_dy(y + k3, x)
        out = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return out

    def predict_from_samples(self, y, x, num_it):
        current_y = y
        current_x = x[0]
        pred_list = y
        for it in range(num_it - 1):
            pred_var = self.predict(current_y, current_x)
            pred_list = np.vstack((pred_list, pred_var))
            current_y = pred_var
            current_x = x[it+1]
        return pred_list

    def predict_dy_from_samples(self, y, x, num_it):
        current_y = y
        current_x = x
        pred_list = y
        for _ in range(num_it - 1):
            pred_var = self.predict_dy(current_y, current_x)
            pred_list = np.vstack((pred_list, pred_var))
            current_y = pred_var
        return pred_list

    def start_train(self, data_y, data_x, output_y=None):
        if data_y.ndim == 1:
            data_y = data_y.reshape((data_y.shape[0], 1))
        if data_x.ndim == 1:
            data_x = data_x.reshape((data_x.shape[0], 1))

        # Data has shape (#snapshots, #vars)
        #self.f_t_init = data_y[0]
        n_train = data_y.shape[0]

        self.data = np.hstack((data_y, data_x))
        self.input_y = data_y
        self.input_x = data_x
        self.joint_input = np.hstack((self.input_y, self.input_x))

        if output_y is None:
            self.output = data_y[1:]
        else:
            self.output = output_y[:400]

        self.joint_input = np.hstack((self.input_y, self.input_x))
        ratio = 0.1

        train_input, test_input, train_output, test_output = train_test_split(self.joint_input, self.output,
                                                            test_size=int(n_train * ratio), random_state=42)
        test_y = test_input[:, :self.n_var]
        test_x = test_input[:, self.n_var:]

        n_batches = int(train_input.shape[0] / self.bsize)
        leftover = train_input.shape[0] - n_batches * self.bsize

        # Input and output variables
        x = tf.placeholder("float", [None, self.n_oppar], name="OperatingParameters")
        y = tf.placeholder("float", [None, self.n_var], name="ODEInputVariable")
        z = tf.placeholder("float", [None, self.n_var], name="ODEOutputVariable")

        # Arrays to train
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
        pred = self.tf_forward_pass(y, x)

        # Loss function and optimizer
        pow_diff = tf.pow(pred - z, 2)
        loss = tf.reduce_sum(pow_diff)
        '''
        # Regularizers:
        reg_losses = tf.nn.l2_loss(self.tf_w1) + tf.nn.l2_loss(self.tf_w2) + \
                     tf.nn.l2_loss(self.tf_w3)
        reg_weight = 0.001
        loss += reg_weight*reg_losses
        '''
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        # Hardcoded parameters:
        epoch = 0
        train_loss = 50
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            while train_loss > self.bound and epoch < self.train_epochs:
                epoch += 1
                # Shuffle training samples
                rand_index = np.random.permutation(len(train_input))
                # train1 = self.add_gaussian_noise(train_x[rand_index, :self.n_var], 0.0005) ## Testing noisy input
                train_y = train_input[rand_index, :self.n_var]
                train_x = train_input[rand_index, self.n_var:]
                # train_out = self.add_gaussian_noise(train_y[rand_index,:], 0.0005)
                train_out = train_output[rand_index, :]
                for i in range(n_batches):
                    # Group shuffled samples in batches
                    data_in_y = train_y[i * self.bsize:(i + 1) * self.bsize].reshape((self.bsize, self.n_var))
                    data_in_x = train_x[i * self.bsize:(i + 1) * self.bsize].reshape((self.bsize, self.n_oppar))
                    data_out = train_out[i * self.bsize:(i + 1) * self.bsize].reshape((self.bsize, self.n_var))
                    values = {y: data_in_y,
                              x: data_in_x,
                              z: data_out}
                    sess.run(optimizer, feed_dict=values)

                if leftover == 0:
                    pass
                else:
                    data_in_y = train_y[-leftover:, :].reshape((leftover, self.n_var))
                    data_in_x = train_x[-leftover:, :].reshape((leftover, self.n_oppar))
                    data_out = train_output[-leftover:, :].reshape((leftover, self.n_var))
                    values = {y: data_in_y,
                              x: data_in_x,
                              z: data_out}
                    sess.run(optimizer, feed_dict=values)


                if epoch % self.dstep == 0:
                    #values = {y: data_in_1,
                    #          x: data_in_2,
                    #          z: data_out}
                    #res = sess.run(self.tf_forward_pass(y, x), feed_dict=values)
                    #print("Predicted values are:", res[:2, :2])
                    #print("Real values are:", data_out[:2, :2])
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


def add_gaussian_noise(layer, std):
    return np.random.normal(layer, scale=std)

def order_training_data(y_var, params, range_list):
    """
    Given simulation data in one big array, order training data for neural network.
    y' = F( y, x )  (eq1)
    :param y_var: y in eq(1)
    :param params: x in eq(1)
    :param range_list: Number of snapshots per simulation
    :return: inputs and outputs to train the neural network

    """
    y_mean = np.mean(y_var, axis=0)
    y_std = np.std(y_var, axis=0)
    params_mean = np.mean(params, axis=0)
    params_std = np.std(params, axis=0)
    #if params_std == 0:
    #    params_std = params

    y_var -= y_mean
    params -= params_mean
    #params /= params_std


    num_train_snapshots = y_var.shape[0] - len(range_list)

    if y_var.ndim == 1:
        y_var = y_var.reshape((y_var.shape[0],1))
        n_vars = 1
    else:
        n_vars = y_var.shape[1]

    if params.ndim == 1:
        params = params.reshape((params.shape[0], 1))
        n_params = 1
    else:
        n_params = params.shape[1]

    inputvar_list = np.zeros((num_train_snapshots, n_vars))
    param_list = np.zeros((num_train_snapshots, n_params))
    outputvar_list = np.zeros((num_train_snapshots, n_vars))

    start = 0
    #end = 0
    current = 0
    for rang in range_list:
        end = start + rang - 1
        inputvar_list[start:end] = y_var[current:current+rang-1]
        param_list[start:end] = params[current:current+rang-1]
        outputvar_list[start:end] = y_var[current+1:current+rang]
        current += rang
        start = end
    return inputvar_list, param_list, outputvar_list


def main_POD():

    n_oppar = 1
    n_var = 10
    timestep = 0.05
    t_init = 0
    t_end = 20.05


    data_in_y = np.load('data\\test_iny_m.npy')
    data_in_x = np.load('data\\test_inx_m.npy')
    data_out = np.load('data\\test_out_m.npy')
    n_oppar = 1
    n_var = 3
    timestep = 0.1
    t_init = 0
    t_end = 1

    '''
    data = np.load('data\\POD_Coeff_40.npy')
    twod_data = data[:, :n_var] # Simulation run every 401 steps
    twod_data -= np.mean(twod_data, axis=0)
    #twod_data /= np.std(twod_data, axis=0)
    '''
    num_it = int((t_end-t_init)/timestep)
    oppar = 0
    rk = RungeKutta_nn(n_oppar, n_var, timestep, t_init, t_end)
    rk.bsize = int(num_it/10.0)
    rk.train_epochs = 2000
    rk.lr = 0.0001
    rk.bound = 0.0001
    rk.n_hidden1 = 10
    rk.n_hidden2 = 10

    #oppar = np.zeros(twod_data.shape[0])
    #rk.start_train(twod_data, oppar)
    rk.start_train(data_in_y, data_in_x, data_out)
    range10 = np.arange(0,10,1)
    pred = rk.predict_from_samples(y=data_in_y[0], x=data_in_x, num_it=10)
    plt.plot(pred[:,0], pred[:,1])
    plt.plot(data_out[:10, 0], data_out[:10, 1])
    plt.show()

    time_vals = np.arange(t_init, t_end, timestep)
    '''
    pred = rk.predict_from_samples(y=data[-401,:n_var], x=oppar, num_it=num_it)

    # Plots
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(131, projection='3d')
    plt.plot(rk.data[:, 0], rk.data[:, 1], rk.data[:, 2], label="Real Value")
    plt.plot(pred[:, 0], pred[:, 1], pred[:, 2], label="Prediction")
    plt.title("ODE solution")
    plt.legend()

    ax = fig.add_subplot(132)
    plt.plot(time_vals, rk.data[:, 0], label="Real Value")
    plt.plot(time_vals, pred[:, 0], label="Prediction")
    plt.title("ODE solution x_1")
    plt.legend()

    ax = fig.add_subplot(133)
    plt.plot(time_vals, rk.data[:, 1], label="Real Value")
    plt.plot(time_vals, pred[:, 1], label="Prediction")
    plt.title("ODE solution x_2")
    plt.legend()

    plt.show()
    '''


if __name__ == "__main__":
    #main_1d() # To run main_1d, the function ode_sol must be changed to a 1d eq.
    #main_2d()
    main_POD()
