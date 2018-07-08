from RungeKuttaNN import RungeKuttaNN
from helper import *
from matplotlib import pyplot as plt

def main():
    """Approximates solution of ODE's with RK NN
    """
    # Time step parameters
    timestep = 0.01
    t_init = 0
    t_end = 100
    num_it = int((t_end-t_init)/timestep)

    ## Initialise neural network and set parameters
    rk = RungeKuttaNN(timestep, t_init, t_end)
    # Number of batches
    rk.n_batches = 10
    # Number of training epochs
    rk.train_epochs = 2000
    # Learning rate
    rk.lr = 1e-3
    # Error target
    rk.bound = 5e-8
    # Number of neurons in the first layer
    rk.n_hidden1 = 20
    rk.n_hidden2 = 20
    #TODO: Parameters that can change : learning rate, #training epochs and number of hidden neurons
    #TODO: Add dimensionality reduction techniques for big data
    # Obtain experimental data and set main quantities
    data, mean, std = set_data(timestep, t_init, t_end)
    n_samples = data.shape[0]
    n_timesteps = n_samples-1
    input_var = data[:n_timesteps]
    input_oppar = np.zeros((n_samples,1))   # Trivial parameter in this example
    output_var = data[1:]

    # Start training
    rk.start_train(input_var, input_oppar[:n_timesteps], output_var)

    time_vals = np.arange(t_init, t_end, timestep)
    pred = rk.predict_from_samples(y=data[0], x=input_oppar, num_it=num_it)

    #TODO: Optional- denormalise prediction and original data with mean and std

    # Plots
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(131)
    plt.plot(data[:,0], data[:,1], label="Real Value")
    plt.plot(pred[:,0], pred[:,1], label="Prediction")
    plt.title(r"ODE solution $y' = F(x,y)$")
    plt.xlabel(r"$y_1$")
    plt.ylabel(r"$y_2$")
    plt.legend()

    ax = fig.add_subplot(132)
    plt.plot(time_vals, data[:, 0], label="Real Value")
    plt.plot(time_vals, pred[:, 0], label="Prediction")
    plt.title(r"ODE solution $y_1$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$y_1$")
    plt.legend()

    ax = fig.add_subplot(133)
    plt.plot(time_vals, data[:, 1], label="Real Value")
    plt.plot(time_vals, pred[:, 1], label="Prediction")
    plt.title(r"ODE solution $y_2$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$y_2$")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
