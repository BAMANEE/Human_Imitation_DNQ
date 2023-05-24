"""Plot graphs of the scores of the agents while training."""
import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_graphs(folder_list, label_list=None, title=None, rolling_average=100, n_episodes=None, save_path=None):
    """Plot graphs of the scores of the agents while training."""

    while True:
        for folder in folder_list:
            scores = np.transpose([np.loadtxt(folder + "/" + file) for file in os.listdir(folder) if os.path.isfile(folder + "/" + file) and os.path.basename(file) !=  "parameters.txt"])
            if n_episodes:
                scores = scores[:n_episodes]
            scores_df = pd.DataFrame(scores)
            scores_df = scores_df.rolling(rolling_average).mean()
            mean = scores_df.mean(axis=1)
            std = scores_df.std(axis=1)
            if label_list is not None:
                plt.plot(mean, label=label_list[folder_list.index(folder)])
            else:
                plt.plot(mean, label=folder)
            plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel(f"Score (rolling average of {rolling_average} episodes)")
            if title is not None:
                plt.title(title)
            else:
                plt.title("Score while training (rolling average of 100 episodes)")
            plt.grid(True)
        
        plt.legend(loc="lower right")
        plt.draw()
        if save_path is not None:
            plt.savefig(save_path)
            break
        plt.pause(10)
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_list', type=str, nargs='+', help='List of directories to plot')
    parser.add_argument('--label_list', type=str, nargs='+', help='List of labels')
    parser.add_argument('--title', type=str, help='Title of the plot ')
    parser.add_argument('--rolling_average', type=int, default=100, help='Number of episodes to average over')
    parser.add_argument('--n_episodes', type=int, default=None ,help='Number of episodes to plot')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the plot')
    args = parser.parse_args()

    plot_graphs(args.folder_list, label_list=args.label_list, title=args.title, rolling_average=args.rolling_average, n_episodes=args.n_episodes, save_path=args.save_path)