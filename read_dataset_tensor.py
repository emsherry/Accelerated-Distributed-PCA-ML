# read_dataset_tensor.py

import numpy as np
import gzip
import pickle
from tensorly.decomposition import tucker
import tensorly as tl

tl.set_backend('numpy')

def read_mnist_tensor():
    with gzip.open('Datasets/raw/mnist_py3k.pkl.gz', 'rb') as f:
        (train, _), (valid, _), _ = pickle.load(f)
    data = np.concatenate((train, valid))  # shape (N, 784)
    T = data.reshape(-1, 28, 28)           # tensor: (N,28,28)
    return T.astype('float64')

def read_cifar10_tensor():
    from read_dataset import unpickle
    parts = [unpickle(f'Datasets/raw/cifar-10-batches-py/data_batch_{i}') for i in range(1,6)]
    flat = np.concatenate([p[b'data'] for p in parts])
    T = flat.reshape(-1, 32, 32, 3)        # tensor: (N,32,32,3)
    return T.astype('float64')

def tensor_to_features(T, ranks):
    from numpy.linalg import norm
    # Tucker decomposition
    core, factors = tucker(T, rank=ranks, init='svd', tol=1e-5)
    # Flatten the core tensor
    features = core.reshape(core.shape[0] * core.shape[1], -1)
    # Normalize across each feature vector (column-wise)
    features = features / (norm(features, axis=0, keepdims=True) + 1e-8)
    # Zero-mean the features
    features = features - np.mean(features, axis=1, keepdims=True)
    return features

def read_data_tensor(dataset, ranks=[10, 10], limit=500):
    if dataset == 'mnist':
        with gzip.open("Datasets/raw/mnist_py3k.pkl.gz", 'rb') as f:
            (train_inputs, _), (_, _), (_, _) = pickle.load(f)
        train_inputs = train_inputs[:limit]
        T = train_inputs.reshape(-1, 28, 28)
    elif dataset == 'cifar10':
        from read_dataset import unpickle
        parts = [unpickle(f'Datasets/raw/cifar-10-batches-py/data_batch_{i}') for i in range(1,6)]
        flat = np.concatenate([p[b'data'] for p in parts])
        flat = flat[:limit]
        T = flat.reshape(-1, 32, 32, 3)
    else:
        raise ValueError("Unsupported dataset")
    
    return tensor_to_features(T, ranks)


