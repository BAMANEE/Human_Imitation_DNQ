from time import sleep
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import argparse
import os

def plot_graphs(dirlist, continuous=False):

    while continuous:
        for dir in dirlist:
            scores = np.transpose([np.loadtxt(dir + "/" + file) for file in os.listdir(dir) if os.path.isfile(dir + "/" + file)])
            scores_df = pd.DataFrame(scores)
            scores_df = scores_df.rolling(100).mean()
            mean = scores_df.mean(axis=1)
            std = scores_df.std(axis=1)
            plt.plot(mean, label=dir)
            plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)
        
        plt.legend()
        plt.draw()
        plt.pause(10)
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirlist', type=str, nargs='+', help='List of directories to plot')
    parser.add_argument('--continuous', type=bool, default=False, help='Continuous plot')
    args = parser.parse_args()

    plot_graphs(args.dirlist, args.continuous)