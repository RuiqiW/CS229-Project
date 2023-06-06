import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import svm

import open3d as o3d

import os
from PIL import Image

import time
import scipy


random_state = 0
n_components = 10


DATA_LABELS = '../data/train_meshMNIST/labels.txt'
DATA_FORMAT = 'voxel'
ROOT_DIR = '../data/train_{}'.format(DATA_FORMAT)


def get_data(data_path):
    if DATA_FORMAT == 'proj':
        with open(data_path, 'rb') as f:
            data = np.array(Image.open(f).convert('L'))
    elif DATA_FORMAT == 'single_view':
        with open(data_path, 'rb') as f:
            data = np.array(Image.open(f).convert('L'))
    elif DATA_FORMAT == 'voxel':
        with open(data_path, 'rb') as f:
            data = np.load(f).astype(np.float32)
    elif DATA_FORMAT == 'pcd':
        pcd = o3d.io.read_point_cloud(data_path)
        data = np.asarray(pcd.points)
    elif DATA_FORMAT == 'multi_view_upright':
        with open(data_path, 'rb') as f:
            data = np.load(f)[0, :, :]
    return data.flatten()




if __name__ == '__main__':

    if DATA_FORMAT == 'proj' or DATA_FORMAT == 'single_view':
        filename_format = "{:05d}.png" 
    elif DATA_FORMAT == 'voxel' or DATA_FORMAT == 'multi_view_upright':
        filename_format = "{:05d}.npy"
    elif DATA_FORMAT == 'pcd':
        filename_format = "{:05d}.ply"
    else:
        raise NotImplementedError("data format not supported")


    # Load Data
    with open(DATA_LABELS, 'r') as f:
        lines = f.readlines()

    # # Shuffle the lines randomly
    # random.shuffle(lines)

    # Split the data into training, validation, and test sets
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    num_samples = len(lines)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val

    train_lines = lines[:num_train]
    val_lines = lines[num_train:num_train+num_val]
    test_lines = lines[num_train+num_val:]


    X_train, y_train = [], []
    X_test, y_test = [], []




    for line in test_lines:
        data_id, label = line.strip().split(',')
        data_path = os.path.join(ROOT_DIR, filename_format.format(int(data_id)))
        data = get_data(data_path)
        X_test.append(data)
        y_test.append(int(label))

    X_test = np.vstack(X_test)
    print(X_test.shape)


    for line in train_lines:
        data_id, label = line.strip().split(',')
        data_path = os.path.join(ROOT_DIR, filename_format.format(int(data_id)))
        data = get_data(data_path)
        X_train.append(data)
        y_train.append(int(label))

    X_train = np.vstack(X_train)
    print(X_train.shape)


    # ----------------------------------------------------------------
    # Source Code: https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html

    # # Load Digits dataset
    # X, y = datasets.load_digits(return_X_y=True)

    # # Split into train/test
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.5, stratify=y, random_state=random_state
    # )

    dim = X_train.shape[1]
    n_classes = 10

    svm =  svm.SVC(kernel='rbf')

    # Reduce dimension with PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_pipeline = make_pipeline(StandardScaler(), pca)


    # Make a list of the methods to be compared
    dim_reduction_methods = [("PCA", pca_pipeline)]

    # plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        # plt.subplot(1, 3, i + 1, aspect=1)

        print(name)
        # Fit the method's model
        model.fit(X_train, y_train)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)

        # Fit a classifier on the embedded training set
        svm.fit(model.transform(X_train), y_train)

        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X_test)

        # Compute the accuracy on the embedded test set
        acc_svm = svm.score(X_embedded, y_test)

        X_tsne = TSNE(n_components=2, learning_rate='auto').fit_transform(X_embedded)

        # Plot the projected points and show the evaluation score
        plt.figure()
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test, s=30, cmap="Set1")
        plt.title(
            "{}, SVM \nTest accuracy = {:.4f}".format(name, acc_svm)
        )

        # plt.savefig("{}_{}.png".format(name, DATA_FORMAT))
        plt.show()

