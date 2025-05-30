# File: FedAvg.py

import numpy as np
from data.mnist import load_mnist_data
from sklearn.metrics import accuracy_score
import wandb
# Initialize Weights & Biases

def hinge_loss(w, X, y, lambd):
    margins = 1 - y * (X @ w)
    loss = np.mean(np.maximum(0, margins)) + lambd * np.sum(w ** 2)
    return loss

def compute_gradient(w, X, y, lambd):
    margins = 1 - y * (X @ w)
    indicator = margins > 0
    grad = -np.mean((indicator * y)[:, np.newaxis] * X, axis=0) + 2 * lambd * w
    return grad

def local_update(X, y, w_global, lr, lambd):
    grad = compute_gradient(w_global, X, y, lambd)
    return w_global - lr * grad

def aggregate_weights(weights, sizes):
    total = np.sum(sizes)
    return sum(w * (sz / total) for w, sz in zip(weights, sizes))

def smooth_curve(values, alpha=0.3):
    """Exponential moving average: less aggressive smoothing"""
    smoothed = [values[0]]  # start with the first actual value
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
    return smoothed


def federated_training(num_clients=5, num_rounds=20, lr=0.01, lambd=0.01, sigma=0.0, binary=True, dataset="mnist", smooth=False, alpha=0.3):
    """
    Simulate Federated Averaging (FedAvg) with optional noise on model transmission.

    Args:
        num_clients (int): Number of clients.
        num_rounds (int): Federated training rounds.
        lr (float): Learning rate.
        lambd (float): L2 regularization.
        sigma (float): Stddev of Gaussian noise added to local models.
        binary (bool): Even-vs-odd conversion flag.
        smooth (bool): If True, applies exponential smoothing to output.
        alpha (float): Smoothing factor if smoothing is enabled.

    Returns:
        w_global: Final global model.
        accs, losses: Accuracy and loss history (optionally smoothed).
    """
    print("[INFO] Loading dataset...")
    if dataset == "mnist":
        X_train, X_test, y_train, y_test = load_mnist_data(binary=binary)
        y_train_bin = 2 * y_train - 1
        y_test_bin = 2 * y_test - 1
    else:
        raise ValueError("Unsupported dataset: " + dataset)

    n_samples, n_features = X_train.shape
    w_global = np.zeros(n_features)

    # Split data IID across clients
    indices = np.random.permutation(n_samples)
    splits_X = np.array_split(X_train[indices], num_clients)
    splits_y = np.array_split(y_train_bin[indices], num_clients)
    sizes = [len(x) for x in splits_X]

    accs, losses = [], []

    print("[INFO] Starting federated training...")
    for round in range(num_rounds):
        local_weights = []
        for i in range(num_clients):
            X_local, y_local = splits_X[i], splits_y[i]
            w_local = local_update(X_local, y_local, w_global, lr, lambd)

            # Add communication noise
            if sigma > 0:
                delta = np.random.randn(*w_local.shape)
                delta /= max(np.linalg.norm(delta), 1e-8)  # Normalize
                delta *= sigma * 1.2  # Scale
                w_local += delta

            local_weights.append(w_local)

        # Aggregate models
        w_global = aggregate_weights(local_weights, sizes)

        # Evaluate
        preds = np.sign(X_test @ w_global)
        acc = accuracy_score(y_test_bin, preds)
        loss = hinge_loss(w_global, X_train, y_train_bin, lambd)

        wandb.log({
            "round": round + 1,
            "FedAvg/accuracy": acc,
            "FedAvg/loss": loss
        })

        accs.append(acc)
        losses.append(loss)
        print(f"[Round {round+1}] Test Acc: {acc:.4f}, Loss: {loss:.4f}")

    # Optional smoothing for plotting (not used internally or for wandb)
    if smooth:
        accs = smooth_curve(accs, alpha=alpha)
        losses = smooth_curve(losses, alpha=alpha)

    return w_global, accs, losses
