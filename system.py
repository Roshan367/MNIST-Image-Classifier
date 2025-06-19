from utils import *
import numpy as np
import pickle
import scipy.linalg


# Defining the global variables
v_global = None
train_mean_global = None
n_global = None

"""
This is the class for the Model that will be saved
during the training section

This class will also be used for the prediction/classification
and for access to some of the variables stored
"""


class Model:
    """
    Initialises the variables that will be used in the class
    """

    def __init__(self):
        self.k = None
        self.eigenvector = None
        self.train_mean = None
        self.train_features = None
        self.train_labels = None
        self.features = None

    """
    Saves the given variables in the class
    """

    def fit(self, k, eigenvector, train_mean, train_features, train_labels, features):
        self.k = k
        self.eigenvector = eigenvector
        self.train_mean = train_mean
        self.train_features = train_features
        self.train_labels = train_labels
        self.features = features

    """
    Calls the classify function to predict the given test data
    """

    def predict(
        self,
        pcatest_data,
    ):
        features = self.features
        return self.classify(pcatest_data, features)

    """
    Uses the KNN function to find the predicted labels
    """

    def classify(self, pcatest_data, features=None):
        labels = kNearestNeighbour(
            self.train_features, self.train_labels, pcatest_data, features, self.k
        )
        return labels

    """
    Used to get some of the saved variables from the trained model
    """

    def get_pca_params(self):
        return self.eigenvector, self.train_mean


# CLASSIFICATION SECTION

"""
Performs the K Nearest Neighbour algorithm using the reduced training
and test data a long with the optimal k value

The cosine distance is used as the distance metric and the closest K
labels are found, the most occuring label is then set 
"""


def kNearestNeighbour(pcatrain_data, train_labels, pcatest_data, features, k):
    # Selects the desired features from the training and test data
    train = pcatrain_data[:, features]
    test = pcatest_data[:, features]

    # Works out the cosine distance
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())

    # Gets the closest K indices and gets the corresponding labels
    top_k_indices = np.argsort(-dist, axis=1)[:, :k]
    top_k_labels = train_labels[top_k_indices]

    # Gets the most occuring label and appending it to the prediction label list
    labels = []
    for label in top_k_labels:
        labels.append(majority_vote(label))

    return labels


"""
Takes in the list of labels and finds the frequency of the
highest appearing label
The label with this frequency is then returned 
If multiple labels have the highest frequency the smallest one
is returned
"""


def majority_vote(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)

    max_count = np.max(counts)

    most_frequent_labels = unique_labels[counts == max_count]

    return np.min(most_frequent_labels)


# DIMENSIONALITY REDUCTION SECTION

"""
This will take in the images and perform Principal Component
Analysis to reduce the features and return the reduced features

There is also a commented out call to the explained variance plot
function, if you wish to view the plot uncomment the line and uncomment
the explained_variance_plot function
"""


def image_to_reduced_feature(images, split=None):
    global v_global
    global train_mean_global

    if split == "train":
        # SHOWS EXPLAINED VARIANCE PLOT
        # UNCOMMENT THIS AND THE FUNCTION TO VIEW
        # explained_variance_plot(images)

        pcaimages = doPca(images, v_global, train_mean_global, split)

        return pcaimages

    else:
        # Loads the training model
        model = load_model()

        # For the test data the training eigenvectors and mean are used
        v, train_mean = model.get_pca_params()

        pcaimages = doPca(images, v, train_mean, split)

        return pcaimages


"""
Performs the Principle Component Analysis

For the training set of data the covariance matrix, eigenvalues
and eigenvectors are calculated

For the testing data it will use the stored mean and eigenvectors
from the trained model
"""


def doPca(data, v, train_mean, split):
    global v_global
    global train_mean_global
    global n_global

    if split == "train":
        # Finds the optimal n components
        optimal_n, cumulative_variance = find_optimal_n_variance(data)
        print(f"Optimal number of components: {optimal_n}")

        # Gets the covariance matrix
        covx = np.cov(data, rowvar=0)
        N = covx.shape[0]

        # Calculates the eigenvalues and eigenvectors
        w, v = scipy.linalg.eigh(covx, subset_by_index=(N - optimal_n, N - 1))

        # Orders the eigenvectors in descending order
        v = np.fliplr(v)
        v.shape

        # Gets the mean
        train_mean_global = np.mean(data)

        # Projects the data onto the principal component axes
        pcaimages = np.dot((data - train_mean_global), v)

        n_global = optimal_n
        v_global = v
    else:
        # Projects the data onto the principal component axes
        pcaimages = np.dot((data - train_mean), v)

    return pcaimages


"""
Given a threshold this will calculate the optimal N number of PCA
components needed to achieve the threshold

The threshold was chosen as 0.91 instead of something higher such as
0.95 or 0.99 which is what you would normally use, this was what I found to
be the best midpoint between the optimal noise and masked thresholds

This is done so when the noisy data is being reduced it doesn't store any
unecessary noise, but this would negatively affect the reduction in clean or masked
data where you would want a higher threshold 
"""


def find_optimal_n_variance(data, threshold=0.91):
    eigenvalues, _ = compute_eigen(data)

    # Computes the cumulative explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find the smallest N components to reach the threshold
    optimal_n = np.argmax(cumulative_variance >= threshold) + 1
    return optimal_n, cumulative_variance


"""
Computes the eigenvalues and eigenvectors and sorts them in descending order
so they can be used to find the optimal n components and help plot the explained
variance graph
"""


def compute_eigen(data):
    # Gets the covariance matrix
    covx = np.cov(data, rowvar=0)

    # Computes eigenvalues and eigenvectors
    w, v = scipy.linalg.eigh(covx)

    # Orders the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(w)[::-1]
    w = w[sorted_indices]
    v = v[:, sorted_indices]
    return w, v


"""
FUNCTION COMMENTED OUT FOR ASSIGNMENT SUBMISSION
UNCOMMENT IF YOU WISH TO VIEW THE PLOTS

This calculates the explained variance ratio for each principal
component and plots it, helping see how many components are actually
necessary for PCA

def explained_variance_plot(data):
    eigenvalues, _ = compute_eigen(data)
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Plot')
    plt.show()
"""


# TRAINING SECTION

"""
Uses cross validation to find the optimal K value for KNN by testing k valuess in the
range 2 - 15 and 
Fits the given data into the Model class which will be saved
"""


def training_model(train_features, train_labels):
    features = range(0, n_global)
    k_range = range(2, 15)

    # Gets all the accuracies from a range of K values using cross validation
    accuracies = cross_validate_knn(train_features, train_labels, k_range, features)

    # Picks the K value that has the highest average accuracy
    optimal_k = k_range[np.argmax(accuracies)]
    print(f"Optimal k: {optimal_k}")

    # Adds necessary variables to the saved model
    model = Model()
    model.fit(
        optimal_k, v_global, train_mean_global, train_features, train_labels, features
    )

    return model


"""
Cross validation is used to find the optimal value for K by using parts of
the training data as test data and analysing and using the accuracies to find
the most accurate K value

The training data is split into 10 folds where each loop of that K value will
have the test data be a section of the 10 portions

The mean of the 10 fold accuracies for that K value will then be the accuracy
of prediction for that K value and the K value with the highest average accuracy
will then be the optimal choice
"""


def cross_validate_knn(train_data, train_labels, k_range, features, num_folds=10):
    n = len(train_data)
    fold_size = n // num_folds
    accuracies = []

    # Shuffles the training data and corresponding labels so it acts as test data
    shuffled_train_data, shuffled_train_labels = shuffle_data_and_labels(
        train_data, train_labels
    )

    for k in k_range:
        fold_accuracies = []
        for i in range(num_folds):
            # Gets the indices for what will act as the testing data
            test_indices = list(range(i * fold_size, (i + 1) * fold_size))

            # The rest of the indices in the data will be the training data
            train_indices = list(set(range(n)) - set(test_indices))

            # Extracts the needed training data and labels from the training indices
            train_fold_data = shuffled_train_data[train_indices]
            train_fold_labels = shuffled_train_labels[train_indices]

            # Extracts the needed testing data and labels from the testing indicies
            test_fold_data = shuffled_train_data[test_indices]
            test_fold_labels = shuffled_train_labels[test_indices]

            # Predict using KNN
            test_predictions = kNearestNeighbour(
                train_fold_data, train_fold_labels, test_fold_data, features, k
            )

            # Compute accuracy for this fold
            accuracy = np.mean(test_predictions == test_fold_labels)

            # Appends the accuracy for this fold
            fold_accuracies.append(accuracy)

        # Appends the average accuracy for all the folds for the specific k value
        accuracies.append(np.mean(fold_accuracies))

    return accuracies


"""
Uses the training data and shuffles the data and corresponding labels
equally to create a 'different' set of data and labels which will be used
to represent a testing set, to be used for the cross validation
"""


def shuffle_data_and_labels(train_data, train_labels):
    # Generate a random permutation of indices
    indices = np.random.permutation(len(train_data))

    # Use the indices to shuffle both train_data and train_labels
    shuffled_train_data = train_data[indices]
    shuffled_train_labels = train_labels[indices]

    return shuffled_train_data, shuffled_train_labels
