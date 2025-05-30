import numpy as np
from copy import deepcopy
from attacks import apply_attack
from trainer import local_update, evaluate_batch_loss, evaluate
import tensorflow as tf

class BasilNode:
    def __init__(self, node_id, model, data_loader, S):
        self.node_id = node_id
        self.model = model
        self.data_loader = data_loader
        self.S = S
        self.stored_models = []

    def get_weights(self):
        return [w.numpy() for w in self.model.trainable_weights]

    def set_weights(self, weights):
        for var, w in zip(self.model.trainable_weights, weights):
            var.assign(w)

    def store_model(self, model_params):
        self.stored_models.append(deepcopy(model_params))
        if len(self.stored_models) > self.S:
            self.stored_models.pop(0)

    def select_model(self):
        losses = []
        for params in self.stored_models:
            self.model.set_params(params)
            loss = evaluate_batch_loss(self.model, self.data_loader)
            losses.append(loss)
        best_idx = np.argmin(losses)
        return self.stored_models[best_idx]

    def update_model(self, epochs, lr):
        local_update(self.model, self.data_loader, epochs, lr)
        return deepcopy(self.model.get_params())

def generate_adversarial_weights(model, data_loader, epsilon=0.1):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    for X_batch, y_batch in data_loader:
        X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.int32)
        with tf.GradientTape() as tape:
            logits = model(X_batch, training=True)
            loss = loss_fn(y_batch, logits)
        gradients = tape.gradient(loss, model.trainable_weights)
        return [w + epsilon * g for w, g in zip(model.get_weights(), gradients)]
def basil_ring_training(nodes, rounds, epochs, test_loader):
    N = len(nodes)
    initial_model = deepcopy(nodes[0].model.get_params())

    for node in nodes:
        for _ in range(node.S):
            node.store_model(initial_model)

    test_accuracies = []

    for r in range(rounds):
        lr = 0.03 / (1 + 0.03 * r)
        print(f"\n[Round {r + 1}/{rounds}] Starting training step with lr={lr:.4f}...")

        for i in range(N):
            selected = nodes[i].select_model()
            nodes[i].model.set_params(selected)
            model_params = nodes[i].update_model(epochs, lr)

            for s in range(1, nodes[i].S + 1):
                neighbor_idx = (i + s) % N
                nodes[neighbor_idx].store_model(model_params)

        if test_loader is not None:
            round_acc = [evaluate(node.model, test_loader) for node in nodes]
            avg_acc = np.mean(round_acc)
            test_accuracies.append(avg_acc)
            print(f"[Round {r + 1}] Average Test Accuracy: {avg_acc:.4f}")

    return [node.model for node in nodes], test_accuracies

def basil_ring_training_with_attack(nodes, rounds, epochs, test_loader, attack_type, attacker_ids):
    test_accuracies = []
    num_nodes = len(nodes)
    for r in range(rounds):
        lr = 0.03 / (1 + 0.03 * r)
        print(f"\\n[Round {r + 1}/{rounds}] Starting training step with lr={lr:.4f}...")
        for i in range(num_nodes):
            selected = nodes[i].select_model()
            nodes[i].model.set_params(selected)
            model_params = nodes[i].update_model(epochs, lr)

            if attack_type != 'none' and i in attacker_ids:
                weights = nodes[i].model.get_params()
                if attack_type == "hidden":
                    adv_weights = generate_adversarial_weights(nodes[i].model, nodes[i].data_loader)
                    attacked_weights = apply_attack(weights, attack_type, malicious_weights=adv_weights, blend_ratio=0.5)
                else:
                    attacked_weights = apply_attack(weights, attack_type)
                nodes[i].model.set_params(attacked_weights)
                model_params = attacked_weights

            for s in range(1, nodes[i].S + 1):
                neighbor_idx = (i + s) % num_nodes
                nodes[neighbor_idx].store_model(model_params)

        if test_loader is not None:
            round_acc = [evaluate(node.model, test_loader) for node in nodes]
            avg_acc = np.mean(round_acc)
            test_accuracies.append(avg_acc)
            print(f"[Round {r + 1}] Average Test Accuracy: {avg_acc:.4f}")
    return [node.model for node in nodes], test_accuracies
