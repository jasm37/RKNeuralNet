import numpy as np

def ode_solution(t):
    """Sets 2D dynamical system
    """
    '''
    y1 = 10*np.cos(t*2*np.pi/100).reshape((t.shape[0],1))
    y2 = 3*np.sin(t*2*np.pi/100).reshape((t.shape[0],1))
    joint = np.hstack((y1,y2))
    '''
    y1 = (2*(np.cos(t) + t*np.sin(t))).reshape((t.shape[0],1))
    y2 = (2*(np.sin(t) - t*np.cos(t))).reshape((t.shape[0],1))
    joint = np.hstack((y1, y2))
    return joint


def set_data(timestep, t_init, t_end):
    t = np.arange(t_init, t_end, timestep)
    # Obtain ODE sol
    data = ode_solution(t)
    # Normalise
    mean = np.mean(data)
    data -= mean
    std = np.std(data)
    data /= std
    return data, mean, std


