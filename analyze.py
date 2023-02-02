import numpy as np
import os

def score_from_n_to_n(dir, start, end):
    scores = np.transpose([np.loadtxt(dir + "/" + file) for file in os.listdir(dir) if os.path.isfile(dir + "/" + file) and os.path.basename(file) !=  "parameters.txt"])
    #Take scores from start to end 
    scores = scores[start:end]
    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std

def first_rolling_average_above_n(dir, n):
    results = []
    files = [file for file in os.listdir(dir) if os.path.isfile(dir + "/" + file) and os.path.basename(file) !=  "parameters.txt"]
    for file in files:
        scores = np.loadtxt(dir + "/" + file)
        scores = np.convolve(scores, np.ones((100,))/100, mode='valid')
        for i in range(len(scores)):
            if scores[i] > n:
                results.append(i)
                break
            if i == len(scores) - 1:
                print("No episode where rolling average is above ", n, " for file ", file)    
    mean = np.mean(results)
    std = np.std(results)
    return mean, std

if __name__ == "__main__":
    dirMountainCarZero = "results/MountainCarZeroEps0995"
    dirMountainCar10 = "results/MountainCar10ImitationEps0995"
    dirMountainCar100 = "results/MountainCar100ImitationEps0995"

    print("Score from 0 to 1000 for MountainCarZero: ", score_from_n_to_n(dirMountainCarZero, 0, 1000))
    print("Score from 0 to 1000 for MountainCar10: ", score_from_n_to_n(dirMountainCar10, 0, 1000))
    print("Score from 0 to 1000 for MountainCar100: ", score_from_n_to_n(dirMountainCar100, 0, 1000))

    print("First episode where rolling average is above -100 for MountainCarZero: ", first_rolling_average_above_n(dirMountainCarZero, -100))
    print("First episode where rolling average is above -100 for MountainCar10: ", first_rolling_average_above_n(dirMountainCar10, -100))
    print("First episode where rolling average is above -100 for MountainCar100: ", first_rolling_average_above_n(dirMountainCar100, -100))