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
    # paired = np.hstack((coeff_1, coeff_2, heat_load*np.ones(coeff_1.shape[0])))
    paired = np.hstack((coeff_1, coeff_2))
    #paired = paired.reshape(1, paired.shape[0])
    # second = coeff_2.reshape(1, coeff_2.shape[0])
    second = coeff_2
    # pred_list = [coeff_1.reshape(1, coeff_1.shape[0]),
    #                  coeff_2.reshape(1, coeff_2.shape[0])]
    pred_list = np.vstack((coeff_1.reshape(1, coeff_1.shape[0]),
                         coeff_2.reshape(1, coeff_2.shape[0])))
    # load_vector = heat_load * np.ones((1,second.shape[1]))
    hl_index = 1
    x1 = coeff_1
    x2 = coeff_2
    for _ in range(num_it - 2):
        pred_var = predict(paired[:,:1], paired[:,1:], w1, w2, wp, b1, b2)
        #t_pred = np.squeeze(pred_var)
        t_pred = pred_var
        # pred_list.append(t_pred)
        pred_list = np.vstack((pred_list, t_pred))
        # paired = np.hstack((second, t_pred, load_vector))
        paired = np.hstack((second, t_pred))
        #paired = paired.reshape(1, paired.shape[0])
        second = t_pred
        hl_index += 1
    # pred_list = np.squeeze(np.asarray(pred_list))
    # pred_list = np.asarray(pred_list)
    return pred_list

def predict(x1, x2, w1, w2, wp, b1, b2):
    layer_1 = sigma(x1 @ w1 + b1)
    #pred1 = layer_1 @ w2 + b2
    #layer_1_2 = sigma(pred1 @ wp + x2 @ w1 + b1)
    layer_1_2 = sigma(layer_1 @ wp + x2 @ w1 + b1)
    pred = layer_1_2 @ w2 + b2
    return pred
'''
seq = np.arange(0,100)
#input1 = seq.reshape(seq.shape[0],1)
#input1 = np.sin(seq.reshape(seq.shape[0],1)*np.pi/20)
input1 = train_func(seq.reshape(seq.shape[0],1))
#input2 = (seq + 1).reshape(seq.shape[0],1)
#input2 = np.sin((seq + 1).reshape(seq.shape[0],1)*np.pi/20)
input2 = train_func((seq + 1).reshape(seq.shape[0],1))
total_input = np.hstack((input1,input2))#np.vstack((input1,input2)).T
#output = (seq+2).reshape(1,input1.shape[0]).T
#output = input1 + input2#().reshape(1,input1.shape[0]).T
#output = np.sin((seq + 2).reshape(seq.shape[0],1)*np.pi/20)
output = train_func((seq + 2).reshape(seq.shape[0],1))
'''
seq_2 = np.arange(0,102)
all_var = train_func(seq_2.reshape(seq_2.shape[0],1))
all_var -= np.mean(all_var)
all_var /= np.std(all_var)
input1 = all_var[0:100]
input2 = all_var[1:101]
output = all_var[2:102]
total_input = np.hstack((input1,input2))#np.vstack((input1,input2)).T
x_size = 1
y_size = 1
lr = 0.00001#0.0001#0.0001
x1 = tf.placeholder("float", [None, x_size], name="x1")
x2 = tf.placeholder("float", [None, x_size], name="x2")
y = tf.placeholder("float", [None, y_size])
n_hidden1 = 100
w1 = tf.get_variable(name="w1", shape=(x_size, n_hidden1), initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.get_variable(name="w2", shape=(n_hidden1, y_size), initializer=tf.contrib.layers.xavier_initializer())
#wp = tf.get_variable(name="wp", shape=(y_size, n_hidden1), initializer=tf.contrib.layers.xavier_initializer())
wp = tf.get_variable(name="wp", shape=(n_hidden1, n_hidden1), initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([n_hidden1]), "b1")
b2 = tf.Variable(tf.zeros([y_size]), "b2")
layer_1 = act_fun(tf.add(tf.matmul(x1,w1), b1))
#pred1 = tf.add(tf.matmul(layer_1,w2), b2) # also layer_2
#layer_1_2 = act_fun(tf.matmul(pred1,wp) + tf.matmul(x2,w1) + b1)
layer_1_2 = act_fun(tf.matmul(layer_1,wp) + tf.matmul(x2,w1) + b1)
pred = tf.add(tf.matmul(layer_1_2,w2), b2)
pow_diff = tf.pow(pred - y, 2)
loss = tf.reduce_sum(pow_diff)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
train_x, test_x, train_y, test_y = train_test_split(total_input, output, test_size=10, random_state=42)
test1 = test_x[:,0]
test2 = test_x[:,1]
#Following two are correlated
batch_size = 5
n_batches = 18
train_loss = 50
bound = 0.000005#0.0000005
train_epochs = 50000
epoch = 0
dstep = 50
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while train_loss > bound and epoch < train_epochs:
        epoch += 1
        # Loop through all samples
        rand_index = np.random.permutation(len(train_x))
        train1 = train_x[rand_index,0]
        train2 = train_x[rand_index, 1]
        train_out = train_y[rand_index]
        for i in range(n_batches):
            data_in_1 = train1[i * batch_size:(i + 1) * batch_size].reshape((batch_size,1))
            data_in_2 = train2[i * batch_size:(i + 1) * batch_size].reshape((batch_size,1))
            data_out = train_out[i * batch_size:(i + 1) * batch_size, :].reshape((batch_size,1))
            values = {x1: data_in_1,
                    x2: data_in_2,
                    y: data_out}
            sess.run(optimizer, feed_dict=values)
            #test_loss = sess.run(loss, feed_dict={x: test_x, y: test_y})
            #train_loss = sess.run(loss, feed_dict={x: train_x, y: train_y})
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
            #max_err = sess.run(pow_diff, feed_dict=values)
            print('Train Loss at step %s: \t%s' % (epoch, train_loss))
            #max_err = np.max(max_err)
            #print("Max. error is %s" % max_err)
        w_1, w_2, w_p, b_1, b_2 = sess.run([w1, w2, wp, b1, b2])
    # a1 = np.array([[100.7]])
    a1 = np.array([input1[0]])
    # a2 = np.array([[99.7]])
    #a2 = np.array([[np.sin(np.pi / 20)]])
    a2 = np.array([input2[0]])
    num_it = 100
    pred_list = predict_from_samples(a1, a2, num_it, w_1, w_2, w_p, b_1, b_2)
    seq_150 = np.arange(0, 100)#input1 = np.sin(seq_150.reshape(seq_150.shape[0], 1) * np.pi / 20)
    #input1 = train_func(seq_150.reshape(seq_150.shape[0], 1))
    plt.plot(seq_150, pred_list,label="Prediction")
    plt.plot(seq_150, input1, label="Real data")
    plt.legend()
    plt.show()
    #a1 = np.array([[100.7]])
    #a1 = np.array([[np.sin(np.pi)]])
    #a2 = np.array([[99.7]])
    #a2 = np.array([[np.sin(np.pi + np.pi/100)]])
    #values = {x1: a1,
    #          x2: a2}
    #print("Prediction for (%f, %f) is %f"%(a1[0,0], a2[0,0], sess.run(pred, feed_dict=values)[0,0]))
