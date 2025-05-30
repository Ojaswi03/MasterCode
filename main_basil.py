import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from trainer import train
from data.mnist import load_mnist_data as load_data
from attacks import apply_attack
from models import create_model
from EBM import ebm_training
from WCM import run_wcm
from FedAvg import run_fedavg
from centralized import centralized_svm

print("========== BASIL + NOISY CHANNEL EXPERIMENT ==========")
nodes = int(input("Enter number of nodes [default=10]: ") or 10)
rounds = int(input("Enter number of rounds [default=50]: ") or 50)
lr = float(input("Enter learning rate [default=0.05]: ") or 0.05)
alpha = float(input("Enter mixing coefficient alpha [default=0.1]: ") or 0.1)
sigma = float(input("Enter communication noise std dev sigma [default=0.01]: ") or 0.01)
use_wcm = input("Use Worst-Case Aggregation? (y/n) [default=n]: ") or 'n'
attack_type = input("Enter attack type (none, gaussian, sign_flip, hidden) [default=none]: ") or 'none'
iid = input("Use IID data split? (y/n) [default=n]: ") or 'n'

print("\n========== Starting Training ==========")

train_loaders, test_loaders, memory_loaders = load_data(nodes, iid == 'y')
models = [create_model() for _ in range(nodes)]
neighbors = list(range(nodes))

# Run Basil baseline
acc_list, loss_list = train(models, train_loaders, test_loaders, memory_loaders, neighbors,
                            num_rounds=rounds, alpha=alpha, sigma=0.0, use_wcm=False)

plt.plot(acc_list)
plt.title("Basil Accuracy (No Attack)")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.savefig("basil_accuracy_no_attack.png")
plt.clf()

plt.plot(loss_list)
plt.title("Basil Loss (No Attack)")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.savefig("basil_loss_no_attack.png")
plt.clf()

# Inject attack and noise
attacks = ['none', 'gaussian', 'sign_flip', 'hidden']
for att in attacks:
    print(f"\nRunning training with {att} attack...")
    models = [create_model() for _ in range(nodes)]
    train_loaders_attacked = apply_attack(train_loaders, att, nodes)

    acc_list, loss_list = train(models, train_loaders_attacked, test_loaders, memory_loaders, neighbors,
                                num_rounds=rounds, alpha=alpha, sigma=sigma, use_wcm=(use_wcm == 'y'))

    plt.plot(acc_list)
    plt.title(f"Accuracy with {att} attack")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.savefig(f"accuracy_{att}.png")
    plt.clf()

    plt.plot(loss_list)
    plt.title(f"Loss with {att} attack")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.savefig(f"loss_{att}.png")
    plt.clf()

# Run centralized baseline
print("\nRunning Centralized SVM baseline...")
train_loader_c, test_loader_c, _ = load_data(1, iid == 'y')
centralized_svm(train_loader_c[0], test_loader_c[0], num_features=784, sigma=sigma, lr=lr, epochs=rounds)

# Run Federated EBM
print("\nRunning Federated EBM...")
run_ebm(nodes, rounds, sigma)

# Run Federated WCM
print("\nRunning Federated WCM...")
run_wcm(nodes, rounds, sigma)

# Run FedAvg
print("\nRunning Federated FedAvg...")
run_fedavg(nodes, rounds, sigma)

print("\nAll training completed. Plots saved.")
