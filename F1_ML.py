import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("")

# Define the features (X) and target variable (y)
X = df[['driver_performance', 'team_performance']].to_numpy()
y = df['champion'].to_numpy()

# Hyperparameters
epochs = 100 # Fixed number of iterations

# Parameters
w, b = np.array([0.0, 0.0]), 0.0 # np.array is used to define a vector

# Define the d(x) function
def d(x):
    return np.dot(w, x) + b # np.dot is the dot product of vectors

# Define the sign function
def sign(x):
    return 1 if x >= 0 else -1

# Define the h(x) function
def h(x):
    return sign(d(x))

# Calculate the score of the decision boundary
def clf_score(X, y):
    score = 0
    for xi, yi in zip(X, y):
        score += yi * h(xi)
    return score

# Perceptron's pocket algorithm
def PLA_pocket(X, y):
    global epochs, w, b

    w, b = np.array([0.0, 0.0]), 0.0 # np.array is used to define a vector
    best_w, best_b = w, b
    best_cs = clf_score(X, y)
    for _ in range(epochs):
        # Sequentially traverse the dataset X
        for xi, yi in zip(X, y):
            # If there is a misclassified sample
            if yi * d(xi) <= 0:
                # Update the normal vectors w and b
                w, b = w + yi * xi, b + yi
                # Score the newly obtained decision boundary
                cs = clf_score(X, y)
                # If it's better, update it
                if cs > best_cs:
                    best_cs = cs
                    best_w, best_b = w, b
                break

    w, b = best_w, best_b

# The following is the training code
# Use train_test_split to randomly split, dividing into training/validation and test sets at an 8:2 ratio
rs = 42
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
print(f'There are {len(X)} samples in total, among which there are {len(X_tv)} samples in the training/validation set and {len(X_test)} samples in the test set.')

# Perform K-fold cross-validation on X_tv
k = 10
kf = KFold(n_splits=k, random_state=rs, shuffle=True)
val_accuracy = 0
for idx, (train, val) in zip(range(k), kf.split(X_tv)):
    X_train, y_train, X_val, y_val = X_tv[train], y_tv[train], X_tv[val], y_tv[val]
    PLA_pocket(X_train, y_train)
    split_train_accuracy = 1 - (len(X_train) - clf_score(X_train, y_train)) / 2 / len(X_train)
    split_val_accuracy = 1 - (len(X_val) - clf_score(X_val, y_val)) / 2 / len(X_val)
    print(f'Fold {idx + 1}, training set accuracy: {split_train_accuracy:.2%}, validation set accuracy: {split_val_accuracy:.2%}')
    val_accuracy += split_val_accuracy
print(f'With epochs = {epochs}, the average validation set accuracy is {val_accuracy / k:.2%}.')
