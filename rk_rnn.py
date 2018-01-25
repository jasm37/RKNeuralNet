from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class rknn():
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
        #noise = np.random.randn(layer.shape[0], layer.shape[1])*std
        #return layer + noise
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
        stack = tf.stack([y, x])
        var_in = tf.squeeze(stack, axis=2)
        var_in = tf.transpose(var_in)
        layer_1 = tf.matmul(var_in, self.tf_w1) + self.tf_b1
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

    def start_train(self):
        data = self.generate_data(dt=self.timestep, t_init=self.t_init, t_end=self.t_end, init=1)
        data -= np.mean(data)
        data /= np.std(data)
        self.f_t_init = data[0]
        x_size = data.shape[0] - 1
        input_var = data[:x_size].reshape((x_size,1))
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
        test1 = test_x[:, 0]
        test2 = test_x[:, 1]

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
                #train1 = self.add_gaussian_noise(train_x[rand_index, 0], 0.0005) ## Testing noisy input
                train1 = train_x[rand_index, 0]
                train2 = train_x[rand_index, 1]
                #train_out = self.add_gaussian_noise(train_y[rand_index], 0.0005)
                train_out = train_y[rand_index]
                for i in range(n_batches):
                    # Group shuffled samples in batches
                    data_in_1 = train1[i * self.bsize:(i + 1) * self.bsize].reshape((self.bsize, 1))
                    data_in_2 = train2[i * self.bsize:(i + 1) * self.bsize].reshape((self.bsize, 1))
                    data_out = train_out[i * self.bsize:(i + 1) * self.bsize].reshape((self.bsize, 1))
                    values = {y: data_in_1,
                              x: data_in_2,
                              z: data_out}
                    sess.run(optimizer, feed_dict=values)

                if leftover == 0:
                    pass
                else:
                    data_in_1 = train1[-leftover:].reshape((leftover,1))
                    data_in_2 = train2[-leftover:].reshape((leftover,1))
                    data_out = train_y[-leftover:].reshape((leftover,1))
                    values = {y: data_in_1,
                              x: data_in_2,
                              z: data_out}
                    sess.run(optimizer, feed_dict=values)

                if epoch % self.dstep == 0:
                    values = {y: test1.reshape((test1.shape[0], 1)),
                              x: test2.reshape((test2.shape[0], 1)),
                              z: test_y.reshape((test_y.shape[0], 1))}
                    test_loss = sess.run(loss, feed_dict=values)
                    values = {y: test1.reshape((test1.shape[0], 1)),
                              x: test2.reshape((test2.shape[0], 1))}
                    prediction = sess.run(self.tf_forward_pass(y, x), feed_dict=values)

                    x_num = np.array([[0]])
                    y_num = np.array([[1]])
                    y_p1 = sess.run(self.tf_forward_pass(y, x), feed_dict={x: x_num, y: y_num})
                    print("Prediction at t=0: ", y_p1)

                    real_vals = test_y.reshape((test_y.shape[0], 1))
                    print(np.sum(np.square(real_vals-prediction)))
                    print('Test Loss at step %s: \t%s' % (epoch, test_loss))
                    values = {y: train1.reshape((train1.shape[0], 1)),
                              x: train2.reshape((train2.shape[0], 1)),
                              z: train_out.reshape((train_out.shape[0], 1))}
                    train_loss = sess.run(loss, feed_dict=values)
                    print('Train Loss at step %s: \t%s' % (epoch, train_loss))
                # Store trained arrays
                self.w1, self.w2, self.w3, self.b1, self.b2, self.b3 = sess.run([self.tf_w1, self.tf_w2, self.tf_w3,
                                                                                 self.tf_b1, self.tf_b2, self.tf_b3])

    def ode_f(self, y):
        return 1.0/(1.0 + y)# = y'

    def ode_solution(self, t, init):
        """
        :param t: ode parameter
        :param init: inital value y(0)
        :return:
        """
        return np.sqrt((init+1)**2-1 + 2*t + 1) -1
        #return ((t-50)**3 + 10)/(-50**3 + 10)

    def generate_data(self, t_init, t_end, dt, init):
        t = np.arange(t_init, t_end, dt)
        return self.ode_solution(t, init)

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

def main():
    n_oppar = 1
    n_var = 1
    timestep = 0.001
    t_init = 0
    t_end = 100
    f_t_init = 1
    opvar = 0
    rk = rknn(n_oppar, n_var, timestep, t_init, t_end)
    rk.bsize = 10000
    rk.train_epochs = 1000
    rk.lr = 0.0001
    rk.bound = 0.0000005
    rk.n_hidden1 = 20
    rk.n_hidden2 = 20
    rk.start_train()
    time_vals = np.arange(t_init, t_end, timestep)
    pred = rk.predict_from_samples(y=rk.f_t_init, x=opvar, num_it=100000)
    plt.title("ODE solution")
    plt.plot(time_vals, rk.data, label="True")
    plt.plot(time_vals, pred, label="Prediction")
    plt.legend()
    plt.show()

    #pred_dy = rk.predict_dy_from_samples(y=f_t_init, x=opvar, num_it=100000)
    #plt.title("ODE RHS")
    #plt.plot(time_vals, 1/(1+time_vals), label="True")
    #plt.plot(time_vals, pred_dy, label="Prediction")
    #plt.legend()
    #plt.show()

if __name__ == "__main__":
    main()

