import numpy as np
import pickle
import gzip
import math

def read_data(dataset, limit=5000):
    if dataset == 'mnist':
        return read_mnist(limit=limit)
    elif dataset == 'cifar10':
        return read_cifar10(limit=limit)
    else:
        raise ValueError("Unsupported dataset")


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_mnist(limit=5000):
    """
    Loads up to `limit` samples from the MNIST dataset and performs mean normalization.
    """
    with gzip.open("Datasets/raw/mnist_py3k.pkl.gz", 'rb') as f:
        (train_inputs, _), (_, _), (_, _) = pickle.load(f)

    # ✅ Only take the first `limit` samples to avoid memory overflow
    train_inputs = train_inputs[:limit]  # shape: (limit, 784)

    # Transpose to (features, samples) and normalize
    data = train_inputs.T  # shape: (784, limit)
    mean = np.mean(data, axis=1, keepdims=True)
    data = data - mean

    return data

def read_cifar10(limit=None):
    from read_dataset import unpickle
    data_list = []
    total_loaded = 0
    max_batches = 6  # 5 train + 1 test

    for i in range(1, 6):
        if limit is not None and total_loaded >= limit:
            break
        f = f'Datasets/raw/cifar-10-batches-py/data_batch_{i}'
        batch = unpickle(f)[b'data']
        
        if limit is not None:
            remaining = limit - total_loaded
            if remaining <= 0:
                break
            batch = batch[:remaining]
        
        data_list.append(batch)
        total_loaded += batch.shape[0]

    if limit is None or total_loaded < limit:
        test_batch = unpickle('Datasets/raw/cifar-10-batches-py/test_batch')[b'data']
        if limit is not None:
            remaining = limit - total_loaded
            test_batch = test_batch[:remaining]
        data_list.append(test_batch)

    data_concatenated = np.concatenate(data_list, axis=0)  # (N, 3072)
    data_concatenated = data_concatenated[:, :1024]        # reduce dimension
    data = data_concatenated.T                             # shape: (d, N)
    
    d, N = data.shape

    # Mean normalization
    M = np.mean(data, axis=1, keepdims=True)
    data = data - M

    # Normalize by max eigenvalue
    Cy = (1 / N) * np.dot(data, data.T)
    eigvals = np.linalg.eigvalsh(Cy)
    max_eigval = np.max(eigvals)
    data = data / math.sqrt(max_eigval)

    return data
