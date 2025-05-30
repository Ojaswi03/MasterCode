# File: EBM.py

import numpy as np
from data.mnist import load_mnist_data
from sklearn.metrics import accuracy_score
import wandb

def hinge_loss(w, X, y, lambd):
    margins = 1 - y * (X @ w)
    return np.mean(np.maximum(0, margins)) + lambd * np.sum(w**2)

def compute_gradient(w, X, y, lambd):
    margins = 1 - y * (X @ w)
    active = margins > 0
    grad = -np.mean((active * y)[:, np.newaxis] * X, axis=0) + 2 * lambd * w
    return grad

def compute_ebm_gradient(w, X, y, lambd, sigma):
    grad = compute_gradient(w, X, y, lambd)
    reg = 2 * sigma**2 * grad  # ∇(||∇F_j(w)||²) ≈ 2 * grad * ∇F_j(w)
    return grad + reg

def local_update(X, y, w_global, lr, lambd, sigma):
    grad = compute_ebm_gradient(w_global, X, y, lambd, sigma)
    return w_global - lr * grad

def aggregate_weights(weights, sizes):
    total = np.sum(sizes)
    return sum(w * (sz / total) for w, sz in zip(weights, sizes))

def ebm_training(num_clients=5, num_rounds=20, lr=0.01, lambd=0.01, sigma=0.1, binary=True):
    """
    Expectation-Based Model training for Federated Learning.

    Args:
        num_clients (int): Number of clients.
        num_rounds (int): Number of communication rounds.
        lr (float): Learning rate.
        lambd (float): L2 regularization parameter.
        sigma (float): Noise level for EBM regularization.
        binary (bool): Whether to do even-vs-odd binary MNIST classification.

    Returns:
        w_global: Final model weights.
        accs, losses: Lists of accuracy and loss per round.
    """
    print("[INFO] Loading data...")
    X_train, X_test, y_train, y_test = load_mnist_data(binary=binary)
    y_train_bin = 2 * y_train - 1
    y_test_bin = 2 * y_test - 1

    n_samples, n_features = X_train.shape
    w_global = np.zeros(n_features)

    # IID partition
    indices = np.random.permutation(n_samples)
    splits_X = np.array_split(X_train[indices], num_clients)
    splits_y = np.array_split(y_train_bin[indices], num_clients)
    sizes = [len(x) for x in splits_X]

    accs, losses = [], []

    print("[INFO] Starting EBM training...")
    for round in range(num_rounds):
        local_weights = []
        for i in range(num_clients):
            X_local, y_local = splits_X[i], splits_y[i]
            w_local = local_update(X_local, y_local, w_global, lr, lambd, sigma)
            local_weights.append(w_local)

        w_global = aggregate_weights(local_weights, sizes)

        # Evaluate
        preds = np.sign(X_test @ w_global)
        acc = accuracy_score(y_test_bin, preds)
        loss = hinge_loss(w_global, X_train, y_train_bin, lambd)
        wandb.log({
            "round": round + 1,
            "EBM/accuracy": acc,
            "EBM/loss": loss
        })

        accs.append(acc)
        losses.append(loss)
        print(f"[Round {round+1}] Test Acc: {acc:.4f}, Loss: {loss:.4f}")

    return w_global, accs, losses
