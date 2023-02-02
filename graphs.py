"""Plot graphs of the scores of the agents while training."""
import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_graphs(folder_list, label_list=None, title=None, n_episodes=None, continuous=True):
    """Plot graphs of the scores of the agents while training."""

    while continuous:
        for folder in folder_list:
            scores = np.transpose([np.loadtxt(folder + "/" + file) for file in os.listdir(folder) if os.path.isfile(folder + "/" + file) and os.path.basename(file) !=  "parameters.txt"])
            if n_episodes:
                scores = scores[:n_episodes]
            scores_df = pd.DataFrame(scores)
            scores_df = scores_df.rolling(100).mean()
            mean = scores_df.mean(axis=1)
            std = scores_df.std(axis=1)
            if label_list is not None:
                plt.plot(mean, label=label_list[folder_list.index(folder)])
            else:
                plt.plot(mean, label=folder)
            plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel("Score")
            if title is not None:
                plt.title(title)
            else:
                plt.title("Score while training (rolling average of 100 episodes)")
        
        plt.legend()
        plt.draw()
        plt.pause(10)
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_list', type=str, nargs='+', help='List of directories to plot')
    parser.add_argument('--label_list', type=str, nargs='+', help='List of labels')
    parser.add_argument('--title', type=str, help='Title of the plot ')
    parser.add_argument('--n_episodes', type=int, default=None ,help='Number of episodes to plot')
    args = parser.parse_args()

    plot_graphs(args.folder_list, label_list=args.label_list, title=args.title, n_episodes=args.n_episodes)