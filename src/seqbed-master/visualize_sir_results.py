import numpy as np
import matplotlib.pyplot as plt

num_iters = 4  # Total number of BED iterations to visualize
dir_MI = "./results/preliminary_051322/MI/"
dir_MI_MM = "./results/preliminary_051322/MI_MM/"
file_prefix = "sirmodel_seq"

for i in range(num_iters):    
    fname = file_prefix + "_iter{0:d}.npz".format(i+1)     

    # load MI result    
    data = np.load(dir_MI + fname, allow_pickle=True)
    gpyopt_data = data["gpyopt_data"]
    X, Y, x_grid, m, v = gpyopt_data[0], gpyopt_data[1], gpyopt_data[2], gpyopt_data[3], gpyopt_data[4]
    Y_norm = ( Y - np.mean(Y) ) / np.std(Y)

    # plot MI
    plt.figure()
    plt.plot(x_grid, - m, '-k', label="Mean (LFIRE)")
    plt.plot(x_grid, - m + v, '--k')
    plt.plot(x_grid, - m - v, '--k')
    plt.plot(X, -Y_norm, 'xk', label="Evaluations (LFIRE)")

    # load MI_MM result
    data = np.load(dir_MI_MM + fname, allow_pickle=True)
    gpyopt_data = data["gpyopt_data"]
    X, Y, x_grid, m, v = gpyopt_data[0], gpyopt_data[1], gpyopt_data[2], gpyopt_data[3], gpyopt_data[4]
    Y_norm = ( Y - np.mean(Y) ) / np.std(Y)    

    # plot MI_MM
    plt.plot(x_grid, - m, '-r', label="Variational")
    plt.plot(x_grid, - m + v, '--r')
    plt.plot(x_grid, - m - v, '--r')
    plt.plot(X, -Y_norm, 'xr', label="Evaluations (Variational)")

    # finalize plot
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Mutual Information", fontsize=14)    
    plt.title("SIR : Iteration {0:d}".format(i+1), fontsize=16)
    if i == (num_iters-1):
        plt.legend(fontsize=14)

plt.show()