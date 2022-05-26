import numpy as np

import seqbed.simulator as simulator
import seqbed.sequentialdesign as sequentialdesign

import time

import multiprocessing as mp

np.random.seed(12345)

def main():

    ## PARAMETERS ##
    #----------------#
    num_cores = 2           # Number of CPU cores
    # BED_START_ITER = 0    # (NOT CURRENTLY USED) 0 : start from scratch, >0 start from saved run 
    BED_ITERS = 4           # Number of experimental design iterations to run
    DIMS = 1                # Keep at 1 for myopic BED    
    NS = 50                 # Number of prior samples (higher => more accurate posterior)    
    MAX_ITER = 40           # Max number of utility evaluations in B.O. (per core)
    utiltype = "MI_MM"         # "MI" : Kleinegesse with LFIRE, "MI_MM" : Moment matching variational estimator
    save_prefix = "./saves/" + utiltype + "/sirmodel_seq"    # Save file prefix
    #----------------#

    # number of initial data points for B.O.
    if num_cores > 5:
        INIT_NUM = num_cores
    else:
        INIT_NUM = 5

    # ----- SPECIFY MODEL ----- #

    # Obtain model prior samples
    param_0 = np.random.uniform(0, 0.5, NS).reshape(-1, 1)
    param_1 = np.random.uniform(0, 0.5, NS).reshape(-1, 1)
    params = np.hstack((param_0, param_1))

    # Define the domain for BO
    domain = [
        {
            "name": "var_1",
            "type": "continuous",
            "domain": (0.01, 3.00),
            "dimensionality": int(DIMS),
        }
    ]

    # Define the constraints for BO
    # Time cannot go backwards
    if DIMS == 1:
        constraints = None
    elif DIMS > 1:
        constraints = list()
        for i in range(1, DIMS):
            dic = {
                "name": "constr_{}".format(i),
                "constraint": "x[:,{}]-x[:,{}]".format(i - 1, i),
            }
            constraints.append(dic)
    else:
        raise ValueError()

    # ----- RUN MODEL ----- #

    # Define the simulator model
    truth = np.array([0.15, 0.05])
    sumtype = "all"
    if DIMS == 1:
        model = simulator.SIRModel(truth, N=50, sumtype=sumtype)
    else:
        model = simulator.SIRModelMultiple(truth, N=50)
    bounds_sir = [[0, 0.5], [0, 0.5]]

    # Define the SequentialBED object
    SIR_death = sequentialdesign.SequentialBED(
        params,
        model,
        domain=domain,
        constraints=constraints,
        num_cores=num_cores,
        utiltype=utiltype,
    )                
        
    # Run the actual optimisation and save data
    t1 = time.time()
    SIR_death.optimisation(
        n_iter=BED_ITERS,
        BO_init_num=INIT_NUM,
        BO_max_iter=MAX_ITER,
        filn=save_prefix,
        obs_file=None,
        bounds=bounds_sir
    )
    t2 = time.time()
    print('TOTAL TIME: {0:.2f}s'.format(t2-t1))

    # SIR_death.bo_obj.plot_acquisition()

if __name__ == '__main__':
    mp.freeze_support()
    main()