import numpy as np
import time
import matplotlib.pyplot as plt

import seqbed.simulator as simulator
import seqbed.inference as inference

def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend(fontsize=12)

def run_sir_benchmark(NS, NUM_RUNS):

    # ----- INIT STUFF ----- #

    # Define the simulator model
    truth = np.array([0.15, 0.05])
    sumtype = "all"
    simobj = simulator.SIRModel(truth, N=50, sumtype=sumtype)
    bounds_sir = [[0, 0.5], [0, 0.5]]

    # ----- RUN MODEL ----- #    
    d = 1.0    
    lfire_times = np.ndarray((NUM_RUNS, len(NS)))
    mm_times = np.ndarray((NUM_RUNS, len(NS)))
    lfire_utils = np.ndarray((NUM_RUNS, len(NS)))
    mm_utils = np.ndarray((NUM_RUNS, len(NS)))
    for idx_samp, numsamp in enumerate(NS):
        print('NS: ', numsamp)

        # run model
        for idx_run in range(NUM_RUNS):
            print('\t', idx_run)

            # Obtain model prior samples
            param_0 = np.random.uniform(0, 0.5, numsamp).reshape(-1, 1)
            param_1 = np.random.uniform(0, 0.5, numsamp).reshape(-1, 1)
            prior_samples = np.hstack((param_0, param_1))
            weights = np.ones(len(prior_samples))

            # Do LFIRE
            t1_lfire = time.time()
            infobj = inference.LFIRE(
                d, prior_samples, weights, simobj
            )
            this_utils, _ = infobj.ratios(numsamp=numsamp)
            t2_lfire = time.time()
            lfire_times[idx_run, idx_samp] = t2_lfire - t1_lfire
            lfire_utils[idx_run, idx_samp] = np.median(this_utils)

            # Do moment matching
            t1_mm = time.time()
            infobj = inference.MOMENT_MATCH(
                d, prior_samples, weights, simobj
            )
            this_utils = infobj.eval(numsamp=numsamp)
            t2_mm = time.time()
            mm_times[idx_run, idx_samp] = t2_mm - t1_mm
            mm_utils[idx_run, idx_samp] = this_utils

    return lfire_times, lfire_utils, mm_times, mm_utils


def plot_sir_benchmark(NS, lfire_times, lfire_utils, mm_times, mm_utils):

    # plot utils
    fig, ax = plt.subplots()
    plt_lfire = ax.boxplot( lfire_utils,
        positions=np.array(np.arange(len(NS)))*2.0-0.35,
        widths=0.6
    )
    plt_mm = ax.boxplot( mm_utils,
        positions=np.array(np.arange(len(NS)))*2.0+0.35,
        widths=0.6
    )
    define_box_properties(plt_lfire, 'black', 'LFIRE') # '#D7191C'
    define_box_properties(plt_mm, 'red', 'Implicit Variational') # '#2C7BB6'
    ax.set_xticks(np.arange(0, len(NS) * 2, 2))
    ax.set_xticklabels([ str(this_ns) for this_ns in NS ],
        {'fontsize': 14}
    )
    ax.set_title('SIR MI Estimates', fontsize=16)
    ax.set_xlabel('Num. Samples', fontsize=14)
    ax.set_ylabel('MI', fontsize=14)
    
    # set the limit for x axis
    # plt.xlim(-2, len(NS)*2)
    
    # set the limit for y axis
    # plt.ylim(0, 50)

    # plot times
    means_lfire = np.mean(lfire_times, axis=0)
    means_mm = np.mean(mm_times, axis=0)
    stds_lfire = np.std(lfire_times, axis=0)
    stds_mm = np.std(mm_times, axis=0)
    fig, ax = plt.subplots()
    ax.errorbar(NS, means_lfire, yerr=(2*stds_lfire), fmt='-k', label="LFIRE")
    ax.errorbar(NS, means_mm, yerr=(2*stds_mm), fmt='-r', label="Variational (Moment-Matching)")
    # plt.xticks(np.arange(0, len(NS) * 2, 2), NS)
    ax.set_xlabel('Num. Samples', fontsize=14)
    ax.set_ylabel('CPU Time (log-seconds)', fontsize=14)
    ax.set_title('SIR Runtime', fontsize=16)
    plt.yscale('log')
    # plt.legend(fontsize=12)

    plt.show()


if __name__ == '__main__':

    np.random.seed(12345)


    ## PARAMETERS ##
    #----------------#
    NS = [50, 100, 200, 500, 1000]      # Number of prior samples (higher => more accurate posterior)    
    # NS = [10, 20]      # Number of prior samples (higher => more accurate posterior)    
    NUM_RUNS = 10           # Number of runs for each collection of samples
    save_prefix = "./saves/"
    bRUN = False
    bPLOT = True
    fname = save_prefix + "benchmark.npz"
    #----------------#

    # run benchmark
    if bRUN:
        lfire_times, lfire_utils, mm_times, mm_utils = run_sir_benchmark(NS, NUM_RUNS)            
        np.savez(
            fname,
            NS=NS,
            NUM_RUNS=NUM_RUNS,
            lfire_times=lfire_times,
            lfire_utils=lfire_utils,
            mm_times=mm_times,
            mm_utils=mm_utils
        )
        print("Saved: ", fname)

    # plot benchmark
    if bPLOT:
        data = np.load(fname, allow_pickle=True)
        this_NS, lfire_times, lfire_utils, mm_times, mm_utils = data['NS'], data['lfire_times'], data['lfire_utils'], data['mm_times'], data['mm_utils']
        plot_sir_benchmark(this_NS, lfire_times, lfire_utils, mm_times, mm_utils)
