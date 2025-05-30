# main_basil.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from models import MNISTModel
from trainer import train
from data.data_loader import load_mnist_data
from acds import load_acds_mnist
from WCM import worst_case_aggregate

def user_input(prompt, default=None, val_type=str):
    try:
        value = input(prompt)
        return val_type(value) if value != '' else default
    except:
        return default

if __name__ == "__main__":
    print("========== BASIL + NOISY CHANNEL EXPERIMENT ==========")
    num_nodes = user_input("Enter number of nodes [default=10]: ", 10, int)
    num_rounds = user_input("Enter number of rounds [default=50]: ", 50, int)
    lr = user_input("Enter learning rate [default=0.05]: ", 0.05, float)
    alpha = user_input("Enter mixing coefficient alpha [default=0.1]: ", 0.1, float)
    sigma = user_input("Enter communication noise std dev sigma [default=0.01]: ", 0.01, float)
    use_wcm = user_input("Use Worst-Case Aggregation? (y/n) [default=n]: ", 'n') == 'y'
    attack_type = user_input("Enter attack type (none, gaussian, sign_flip, hidden) [default=none]: ", 'none')
    iid_choice = user_input("Use IID data split? (y/n) [default=n]: ", 'n') == 'y'

    # Load MNIST data
    if iid_choice:
        train_loaders, test_loaders, memory_loaders = load_mnist_data(num_nodes=num_nodes, non_iid=False)
    else:
        train_loaders, test_loaders, memory_loaders = load_acds_mnist(num_clients=num_nodes)

    # Initialize models for each node
    models = [MNISTModel().model for _ in range(num_nodes)]

    # Define neighbor connections in logical ring
    neighbors = [(i - 1) % num_nodes for i in range(num_nodes)]

    print("\n========== Starting Training ==========")
    acc_list, loss_list = train(
        models=models,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        memory_loaders=memory_loaders,
        neighbors=neighbors,
        num_rounds=num_rounds,
        lr=lr,
        alpha=alpha,
        sigma=sigma,
        use_wcm=use_wcm
    )

    print("\n========== Training Complete ==========")
    for i, (acc, loss) in enumerate(zip(acc_list, loss_list)):
        print(f"Round {i+1}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")

    print("\n========== Summary Across Attacks ==========")
    print(f"Final Accuracy: {acc_list[-1]:.4f}")
    print(f"Final Loss: {loss_list[-1]:.4f}")

    # Save plots to diagram folder
    os.makedirs("diagram", exist_ok=True)
    config_str = f"_{attack_type}_nodes{num_nodes}_rounds{num_rounds}_sigma{sigma:.3f}" + ("_wcm" if use_wcm else "") + ("_iid" if iid_choice else "_acds")

    # Accuracy Plot
    plt.figure()
    plt.plot(range(1, num_rounds + 1), acc_list, label='Accuracy')
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy per Round")
    plt.grid(True)
    plt.savefig(f"diagram/accuracy{config_str}.png")
    plt.close()

    # Loss Plot
    plt.figure()
    plt.plot(range(1, num_rounds + 1), loss_list, label='Loss', color='red')
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title("Test Loss per Round")
    plt.grid(True)
    plt.savefig(f"diagram/loss{config_str}.png")
    plt.close()
