# # data/data_loader.py

# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Subset, random_split
# import numpy as np
# import os

# def mnist_loaders(num_clients=10, batch_size=64, memory_size=500):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#     # Split train dataset into N client datasets
#     data_per_client = len(train_dataset) // num_clients
#     client_indices = [list(range(i * data_per_client, (i + 1) * data_per_client)) for i in range(num_clients)]

#     train_loaders = []
#     memory_loaders = []
#     for indices in client_indices:
#         client_subset = Subset(train_dataset, indices[:-memory_size])
#         memory_subset = Subset(train_dataset, indices[-memory_size:])

#         train_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
#         memory_loader = DataLoader(memory_subset, batch_size=batch_size, shuffle=False)

#         train_loaders.append(train_loader)
#         memory_loaders.append(memory_loader)

#     # Shared test loader per client
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     test_loaders = [test_loader for _ in range(num_clients)]

#     return train_loaders, test_loaders, memory_loaders



# data/data_loader.py (TensorFlow version for MNIST)

import numpy as np
import tensorflow as tf
import random

def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

def partition_data(x, y, num_clients=10, non_iid=True):
    data_per_client = len(x) // num_clients
    client_data = []

    if non_iid:
        sorted_indices = np.argsort(y)
        x, y = x[sorted_indices], y[sorted_indices]

    for i in range(num_clients):
        start = i * data_per_client
        end = start + data_per_client
        client_x = x[start:end]
        client_y = y[start:end]
        client_data.append((client_x, client_y))

    return client_data

def get_memory_loader(x, y, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess).shuffle(buffer_size=500).batch(batch_size)
    return ds

def get_loader(x, y, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess).shuffle(1000).batch(batch_size)
    return ds

def load_mnist_data(num_nodes=10, non_iid=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)

    clients = partition_data(x_train, y_train, num_clients=num_nodes, non_iid=non_iid)

    train_loaders = []
    memory_loaders = []
    test_loaders = [get_loader(x_test, y_test)] * num_nodes  # Shared test set

    for x_c, y_c in clients:
        x_mem, y_mem = x_c[-500:], y_c[-500:]
        x_train, y_train = x_c[:-500], y_c[:-500]

        train_loaders.append(get_loader(x_train, y_train))
        memory_loaders.append(get_memory_loader(x_mem, y_mem))

    return train_loaders, test_loaders, memory_loaders
