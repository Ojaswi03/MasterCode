# trainer.py (TensorFlow + Basil + Noisy Channel + WCM Ready)

import tensorflow as tf
import numpy as np
import copy
from WCM import worst_case_aggregate  # Ensure this is TensorFlow-compatible

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def evaluate(model, test_loader):
    correct, total = 0, 0
    for x_batch, y_batch in test_loader:
        logits = model(x_batch, training=False)
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        correct += tf.reduce_sum(tf.cast(predictions == tf.cast(y_batch, tf.int32), tf.int32)).numpy()
        total += y_batch.shape[0]
    return correct / total


def local_update(model, train_loader, lr):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    for x_batch, y_batch in train_loader:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return model


def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])


def unflatten_weights(flattened, model_shapes):
    new_weights, idx = [], 0
    for shape in model_shapes:
        size = np.prod(shape)
        new_weights.append(flattened[idx:idx+size].reshape(shape))
        idx += size
    return new_weights


def add_noise(weights, sigma):
    return [w + np.random.normal(0, sigma, size=w.shape) for w in weights]


def aggregate_models(local_model, neighbor_model, memory_loader, use_wcm=False, sigma=0.01, alpha=0.1):
    local_weights = local_model.get_weights()
    neighbor_weights = neighbor_model.get_weights()

    if use_wcm:
        # Flatten weights for worst-case aggregation
        local_flat = flatten_weights(local_weights)
        neighbor_flat = flatten_weights(neighbor_weights)
        shapes = [w.shape for w in local_weights]

        new_flat = worst_case_aggregate(
            local_weights=local_flat,
            neighbor_weights=neighbor_flat,
            local_loss_fn=loss_fn,
            data_loader=memory_loader,
            sigma=sigma,
            alpha=alpha
        )
        new_weights = unflatten_weights(new_flat, shapes)
    else:
        # Add noise to neighbor model before aggregation
        noisy_neighbor_weights = add_noise(neighbor_weights, sigma)
        new_weights = [(1 - alpha) * lw + alpha * nw for lw, nw in zip(local_weights, noisy_neighbor_weights)]

    local_model.set_weights(new_weights)
    return local_model


def train(models, train_loaders, test_loaders, memory_loaders, neighbors,
          num_rounds, lr, alpha, sigma, use_wcm):

    num_nodes = len(models)
    acc_list = []

    for rnd in range(num_rounds):
        print(f"\n[Round {rnd+1}/{num_rounds}] Starting training step with lr={lr:.4f}...")

        # Local update
        for i in range(num_nodes):
            models[i] = local_update(models[i], train_loaders[i], lr)

        # Aggregation
        new_models = []
        for i in range(num_nodes):
            local_model = models[i]
            neighbor_model = copy.deepcopy(models[neighbors[i]])
            memory_loader = memory_loaders[i]

            updated_model = aggregate_models(
                local_model,
                neighbor_model,
                memory_loader,
                use_wcm=use_wcm,
                sigma=sigma,
                alpha=alpha
            )
            new_models.append(updated_model)

        models = new_models

        # Evaluation
        avg_acc = np.mean([evaluate(models[i], test_loaders[i]) for i in range(num_nodes)])
        print(f"[Round {rnd+1}] Average Test Accuracy: {avg_acc:.5f}")
        acc_list.append(avg_acc)

    return acc_list
