from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def ode_solution(t, init):
    """
    :param t: ode parameter
    :param init: inital value y(0)
    """
    '''
        x1 = 10*np.cos(t*2*np.pi/100).reshape((t.shape[0],1))
        x2 = 3*np.sin(t*2*np.pi/100).reshape((t.shape[0],1))
        joint = np.hstack((x1,x2))
        return joint
    '''
    x1 = (2*(np.cos(t) + t*np.sin(t))).reshape((t.shape[0],1))
    x2 = (2*(np.sin(t) - t*np.cos(t))).reshape((t.shape[0],1))
    joint = np.hstack((x1, x2))
    return joint
    #return (np.sqrt((init + 1) ** 2 - 1 + 2 * t + 1) - 1).reshape((t.shape[0],1))
    #return (((t-50)**3 + 10)/(-50**3 + 10)).reshape((t.shape[0],1))


def generate_data(t_init, t_end, dt, init):
    t = np.arange(t_init, t_end, dt)
    return ode_solution(t, init)


def set_data(timestep, t_init, t_end, f_t_init):
    data = generate_data(dt=timestep, t_init=t_init, t_end=t_end, init=f_t_init)
    mean = np.mean(data)
    data -= mean
    std = np.std(data)
    data /= std
    return data, mean, std


class rknn():
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
        #self.sess_name = 'RungeKutta'

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
        self.data_batch = [] #complete dataset
        self.data_coeff = [] #complete POD coefficient representation
        #self.centered_data = 0 #centered dataset: subtracted the mean to all samples
        self.evectors = []
        self.data_list = [] #list of all datasets(for different parameters)
        self.data_mean = []  #mean of data_batch along time samples
        self.point_list = [] #vtk geometry list
        self.coeff_list = []
        self.heatload_list = []

    def add_gaussian_noise(self, layer, std):
        return np.random.normal(layer, scale=std)

    def tf_act_fun(self, x):
        #return tf.nn.tanh(x)
        return tf.nn.relu(x)
        # return tf.nn.sigmoid(x)

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

    def start_train(self, data):
        self.f_t_init = data[0]
        x_size = data.shape[0] - 1
        input_var = data[:x_size,:].reshape((x_size,data.shape[1]))
        self.output = data[1:]
        self.data = data
        oppar = np.zeros((x_size,1))
        joint_input = np.hstack((input_var, oppar))
        self.input_var = input_var
        self.oppar = oppar
        self.joint_input = np.hstack((input_var, oppar))
        ratio = 0.1

        train_x, test_x, train_y, test_y = train_test_split(joint_input, self.output,
                                                            test_size=int(x_size*ratio), random_state=42)
        test1 = test_x[:, :self.n_var]
        test2 = test_x[:, self.n_var:]

        n_batches = int(train_x.shape[0] / self.bsize)
        leftover = train_x.shape[0] - n_batches * self.bsize

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
        # Regularizers:
        '''
        reg_losses = tf.nn.l2_loss(self.tf_w1) + tf.nn.l2_loss(self.tf_w2) + \
                     tf.nn.l2_loss(self.tf_w3)
        reg_weight = 0.00001
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
                rand_index = np.random.permutation(len(train_x))
                #train1 = self.add_gaussian_noise(train_x[rand_index, :self.n_var], 0.0005) ## Testing noisy input
                train1 = train_x[rand_index, :self.n_var]
                train2 = train_x[rand_index, self.n_var:]
                #train_out = self.add_gaussian_noise(train_y[rand_index,:], 0.0005)
                train_out = train_y[rand_index,:]
                for i in range(n_batches):
                    # Group shuffled samples in batches
                    data_in_1 = train1[i * self.bsize:(i + 1) * self.bsize].reshape((self.bsize, self.n_var))
                    data_in_2 = train2[i * self.bsize:(i + 1) * self.bsize].reshape((self.bsize, self.n_oppar))
                    data_out = train_out[i * self.bsize:(i + 1) * self.bsize].reshape((self.bsize, self.n_var))
                    values = {y: data_in_1,
                              x: data_in_2,
                              z: data_out}
                    sess.run(optimizer, feed_dict=values)

                if leftover == 0:
                    pass
                else:
                    data_in_1 = train1[-leftover:,:].reshape((leftover, self.n_var))
                    data_in_2 = train2[-leftover:,:].reshape((leftover, self.n_oppar))
                    data_out = train_y[-leftover:,:].reshape((leftover, self.n_var))
                    values = {y: data_in_1,
                              x: data_in_2,
                              z: data_out}
                    sess.run(optimizer, feed_dict=values)

                if epoch % self.dstep == 0:
                    values = {y: test1.reshape((test1.shape[0], self.n_var)),
                              x: test2.reshape((test2.shape[0], self.n_oppar)),
                              z: test_y.reshape((test_y.shape[0], self.n_var))}
                    test_loss = sess.run(loss, feed_dict=values)
                    print('Test Loss at step %s: \t%s' % (epoch, test_loss))

                    values = {y: train1.reshape((train1.shape[0], self.n_var)),
                              x: train2.reshape((train2.shape[0], self.n_oppar)),
                              z: train_out.reshape((train_out.shape[0], self.n_var))}
                    train_loss = sess.run(loss, feed_dict=values)
                    print('Train Loss at step %s: \t%s' % (epoch, train_loss))
                # Store trained arrays
                self.w1, self.w2, self.w3, self.b1, self.b2, self.b3 = sess.run([self.tf_w1, self.tf_w2, self.tf_w3,
                                                                                 self.tf_b1, self.tf_b2, self.tf_b3])

    def predict_from_samples(self, y, x, num_it):
        current_y = y
        current_x = x
        pred_list = y
        for _ in range(num_it - 1):
            pred_var = self.predict(current_y, current_x)
            pred_list = np.vstack((pred_list, pred_var))
            current_y = pred_var
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

def main_1d():
    n_oppar = 1
    n_var = 1
    timestep = 0.01
    t_init = 0
    t_end = 100
    f_t_init = 1
    num_it = int((t_end-t_init)/timestep)
    oppar = 0
    rk = rknn(n_oppar, n_var, timestep, t_init, t_end)
    rk.bsize = int(num_it/10.0)
    rk.train_epochs = 1000
    rk.lr = 0.01
    rk.bound = 0.0000005
    rk.n_hidden1 = 20
    rk.n_hidden2 = 20
    data, mean, std = set_data(timestep, t_init, t_end, f_t_init)

    rk.start_train(data)

    time_vals = np.arange(t_init, t_end, timestep)
    pred = rk.predict_from_samples(y=rk.f_t_init, x=oppar, num_it=num_it)

    # Plots

    plt.plot(time_vals, rk.data[:, 0], label="Real Value")
    plt.plot(time_vals, pred[:, 0], label="Prediction")
    plt.title("ODE solution")
    plt.legend()

    plt.show()

def main_2d():
    n_oppar = 1
    n_var = 2
    timestep = 0.01
    t_init = 0
    t_end = 100
    f_t_init = 1
    num_it = int((t_end-t_init)/timestep)
    oppar = 0
    rk = rknn(n_oppar, n_var, timestep, t_init, t_end)
    rk.bsize = int(num_it/10.0)
    rk.train_epochs = 2000
    rk.lr = 0.001
    rk.bound = 0.000005
    rk.n_hidden1 = 20
    rk.n_hidden2 = 20
    data, mean, std = set_data(timestep, t_init, t_end, f_t_init)

    rk.start_train(data)

    time_vals = np.arange(t_init, t_end, timestep)
    pred = rk.predict_from_samples(y=rk.f_t_init, x=oppar, num_it=num_it)

    # Plots
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(131)
    plt.plot(rk.data[:,0], rk.data[:,1], label="Real Value")
    plt.plot(pred[:,0], pred[:,1], label="Prediction")
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

if __name__ == "__main__":
    #main_1d() # To run main_1d, the function ode_sol must be changed to a 1d eq.
    main_2d()

