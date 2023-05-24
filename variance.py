import statistics
import argparse

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Name of the file')

# Parse the arguments
args = parser.parse_args()

# Open the file for reading
with open(args.filename, 'r') as file:

    # Read the lines of the file into a list
    lines = file.readlines()

    # Initialize a list to hold the scores
    scores = []

    # Iterate over the lines
    for line in lines:

        # Split the line into words
        words = line.split()

        # If the line starts with "episode", extract the score
        if words[0] == "episode":
            score = float(words[2])
            scores.append(score)

    std = statistics.stdev(scores)

    # Calculate the variance
    average = statistics.mean(scores)

    # Print the variance and average
    print(f"Average score: {average:.2f} std: {std:.2f}")