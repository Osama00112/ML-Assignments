import numpy as np
import pandas as pd
from math import log2

def entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def information_gain(data, labels, attribute_index):
    # Step 1: Calculate the entropy of the entire dataset
    total_entropy = entropy(labels)
    
    print(f"Total Entropy: {total_entropy}")

    # Step 2: Calculate the entropy of the selected attribute
    attribute_values, value_counts = np.unique(data[:, attribute_index], return_counts=True)
    weighted_attribute_entropy = 0

    for value, count in zip(attribute_values, value_counts):
        subset_labels = labels[data[:, attribute_index] == value]
        weighted_attribute_entropy += (count / len(labels)) * entropy(subset_labels)

    # Step 3: Calculate Information Gain
    information_gain_value = total_entropy - weighted_attribute_entropy
    return information_gain_value

# Example usage
# Assume you have a dataset with features (attributes) in columns and labels in the last column
data = np.array([
    [1, 'A', 'Yes'],
    [2, 'B', 'No'],
    [3, 'A', 'Yes'],
    [4, 'B', 'No'],
    [5, 'A', 'No']
])

data = pd.DataFrame(data)

labels = data[:, -1]

# Choose an attribute index for which you want to calculate Information Gain
attribute_index = 2

# Calculate Information Gain for the selected attribute
ig = information_gain(data, labels, attribute_index)

print(f"Information Gain for attribute at index {attribute_index}: {ig}")
