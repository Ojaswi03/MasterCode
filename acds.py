# acds.py
# Adaptive Client Data Selection for MNIST (TensorFlow version)

import numpy as np
import tensorflow as tf
import random

def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

def create_acds_splits(x, y, num_clients=10, shard_per_client=2):
    num_shards = num_clients * shard_per_client
    data_per_shard = len(x) // num_shards

    idxs = np.arange(len(x))
    labels = y
    idxs_labels = np.vstack((idxs, labels)).T
    idxs_labels = idxs_labels[idxs_labels[:, 1].argsort()]

    shard_idxs = [idxs_labels[i * data_per_shard:(i + 1) * data_per_shard, 0] for i in range(num_shards)]
    random.shuffle(shard_idxs)

    client_data = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        for j in range(shard_per_client):
            client_data[i].extend(shard_idxs[i * shard_per_client + j])

    return client_data

def get_loader(x, y, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess).shuffle(1000).batch(batch_size)
    return ds

def load_acds_mnist(num_clients=10, shard_per_client=2):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)

    client_indices = create_acds_splits(x_train, y_train, num_clients, shard_per_client)

    train_loaders, memory_loaders, test_loaders = [], [], []

    for indices in client_indices:
        indices = np.array(indices, dtype=np.int32)
        client_x = x_train[indices]
        client_y = y_train[indices]

        x_mem, y_mem = client_x[-500:], client_y[-500:]
        x_train_c, y_train_c = client_x[:-500], client_y[:-500]

        train_loaders.append(get_loader(x_train_c, y_train_c))
        memory_loaders.append(get_loader(x_mem, y_mem))
        test_loaders.append(get_loader(x_test, y_test))  # Shared test

    return train_loaders, test_loaders, memory_loaders
