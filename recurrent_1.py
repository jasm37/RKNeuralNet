from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def act_fun(x):
    #return tf.nn.tanh(x)
    return tf.nn.relu(x)
    #return tf.nn.sigmoid(x)

def sigma(x):
    #return np.tanh(x)
    return np.maximum(0, x)

def train_func(x):
    #return np.sin(np.pi*x/100) +np.cos(np.pi*x/30)
    #return np.cos(np.pi*x/20) + np.sin(np.pi*x/10)
    #return x/(1+x*x)
    #return x*x/100
    return np.cos(np.pi*x/20)

def predict_from_samples(coeff_1, coeff_2, num_it, w1, w2, wp, b1, b2):
    paired = np.hstack((coeff_1, coeff_2))
    second = coeff_2
    pred_list = np.vstack((coeff_1.reshape(1, coeff_1.shape[0]),
                         coeff_2.reshape(1, coeff_2.shape[0])))
    hl_index = 1
    for _ in range(num_it - 2):
        pred_var = predict(paired[:,:1], paired[:,1:], w1, w2, wp, b1, b2)
        t_pred = pred_var
        pred_list = np.vstack((pred_list, t_pred))
        paired = np.hstack((second, t_pred))
        second = t_pred
        hl_index += 1
    return pred_list

def predict(x1, x2, w1, w2, wp, b1, b2):
    layer_1 = sigma(x1 @ w1 + b1)
    layer_1_2 = sigma(layer_1 @ wp + x2 @ w1 + b1)
    pred = layer_1_2 @ w2 + b2
    return pred

def tensor_pred(x1, x2, w1, w2, wp, b1, b2):
    layer_1 = act_fun(tf.add(tf.matmul(x1, w1), b1))
    layer_1_2 = act_fun(tf.matmul(layer_1, wp) + tf.matmul(x2, w1) + b1)
    pred = tf.add(tf.matmul(layer_1_2, w2), b2)
    return pred

# Get input and output data
n_samples = 202
init_seq = np.arange(0,n_samples)
all_var = train_func(init_seq.reshape(n_samples,1))
all_var -= np.mean(all_var)
all_var /= np.std(all_var)
input1 = all_var[0:n_samples-2]
input2 = all_var[1:n_samples-1]
output = all_var[2:n_samples]

# Split data into training and test samples
joint_input = np.hstack((input1,input2))#np.vstack((input1,input2)).T
train_x, test_x, train_y, test_y = train_test_split(joint_input, output, test_size=10, random_state=42)
test1 = test_x[:,0]
test2 = test_x[:,1]

# Neural Network parameters
batch_size = 5
n_batches = 18
train_loss = 50
bound = 0.000005#0.0000005
train_epochs = 50000
n_hidden1 = 100
epoch = 0
dstep = 50
x_size = 1
y_size = 1
lr = 0.00001#0.0001#0.0001

### Initialize tf/graph variables
# Input and output variables
x1 = tf.placeholder("float", [None, x_size], name="x1")
x2 = tf.placeholder("float", [None, x_size], name="x2")
y = tf.placeholder("float", [None, y_size])

# Arrays to train
w1 = tf.get_variable(name="w1", shape=(x_size, n_hidden1), initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.get_variable(name="w2", shape=(n_hidden1, y_size), initializer=tf.contrib.layers.xavier_initializer())
wp = tf.get_variable(name="wp", shape=(n_hidden1, n_hidden1), initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([n_hidden1]), "b1")
b2 = tf.Variable(tf.zeros([y_size]), "b2")

# Graph output
pred = tensor_pred(x1, x2, w1, w2, wp, b1, b2)

# Loss function and optimizer
pow_diff = tf.pow(pred - y, 2)
loss = tf.reduce_sum(pow_diff)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    while train_loss > bound and epoch < train_epochs:
        epoch += 1
        # Shuffle training samples
        rand_index = np.random.permutation(len(train_x))
        train1 = train_x[rand_index,0]
        train2 = train_x[rand_index, 1]
        train_out = train_y[rand_index]
        for i in range(n_batches):
            # Group shuffled samples in batches
            data_in_1 = train1[i * batch_size:(i + 1) * batch_size].reshape((batch_size,1))
            data_in_2 = train2[i * batch_size:(i + 1) * batch_size].reshape((batch_size,1))
            data_out = train_out[i * batch_size:(i + 1) * batch_size, :].reshape((batch_size,1))
            values = {x1: data_in_1,
                    x2: data_in_2,
                    y: data_out}
            sess.run(optimizer, feed_dict=values)
        if epoch % dstep == 0:
            values = {x1: test1.reshape((test1.shape[0],1)),
                    x2: test2.reshape((test2.shape[0],1)),
                    y: test_y}
            test_loss = sess.run(loss, feed_dict=values)
            print('Test Loss at step %s: \t%s' % (epoch, test_loss))
            values = {x1: train1.reshape((train1.shape[0],1)),
                    x2: train2.reshape((train2.shape[0],1)),
                    y: train_out}
            train_loss = sess.run(loss, feed_dict=values)
            print('Train Loss at step %s: \t%s' % (epoch, train_loss))
        # Store trained arrays
        w_1, w_2, w_p, b_1, b_2 = sess.run([w1, w2, wp, b1, b2])

    # Plot full prediction
    a1 = np.array([input1[0]])
    a2 = np.array([input2[0]])
    num_it = 200
    pred_list = predict_from_samples(a1, a2, num_it, w_1, w_2, w_p, b_1, b_2)
    plot_seq = np.arange(0, 100)
    plt.plot(plot_seq, pred_list,label="Prediction")
    plt.plot(plot_seq, input1, label="Real data")
    plt.legend()
    plt.show()
