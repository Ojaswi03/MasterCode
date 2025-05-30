# trainer.py

import numpy as np
import tensorflow as tf
from WCM import worst_case_aggregate
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def evaluate(model, test_loader):
    correct, total = 0, 0
    for x_batch, y_batch in test_loader:
        logits = model(x_batch, training=False)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        correct += tf.reduce_sum(tf.cast(preds == tf.cast(y_batch, tf.int32), tf.int32)).numpy()
        total += y_batch.shape[0]
    return correct / total

def add_noise(weights, sigma):
    return [w + tf.random.normal(w.shape, stddev=sigma) for w in weights]

def local_update(model, train_loader, lr):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    for x_batch, y_batch in train_loader:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return model

def aggregate_models(local_model, neighbor_model, memory_loader, use_wcm=False, sigma=0.01, alpha=0.1):
    local_weights = local_model.get_weights()
    neighbor_weights = neighbor_model.get_weights()

    if use_wcm:
        from WCM import worst_case_aggregate
        return worst_case_aggregate(local_model, neighbor_model, memory_loader, sigma, alpha)
    else:
        noisy_neighbor_weights = add_noise(neighbor_weights, sigma)
        new_weights = [(1 - alpha) * lw + alpha * nw for lw, nw in zip(local_weights, noisy_neighbor_weights)]
        local_model.set_weights(new_weights)
        return local_model

def train(models, train_loaders, test_loaders, memory_loaders, neighbors,
          num_rounds, alpha, sigma, use_wcm):

    num_nodes = len(models)
    acc_list = []
    loss_list = []

    for rnd in range(num_rounds):
        # Dynamically decay learning rate as per Basil paper: lr_t = 0.3 / (1 + 0.3 * t)
        lr_t = 0.3 / (1 + 0.3 * rnd)
        print(f"\n[Round {rnd+1}/{num_rounds}] Starting training step with lr={lr_t:.4f}, alpha={alpha:.4f} and sigma={sigma:.4f}...")

        # Local update
        for i in range(num_nodes):
            models[i] = local_update(models[i], train_loaders[i], lr_t)

        # Aggregation
        new_models = []
        for i in range(num_nodes):
            local_model = tf.keras.models.clone_model(models[i])
            local_model.set_weights(models[i].get_weights())

            neighbor_model = tf.keras.models.clone_model(models[neighbors[i]])
            neighbor_model.set_weights(models[neighbors[i]].get_weights())

            memory_loader = memory_loaders[i]

            if use_wcm:
                updated_weights = worst_case_aggregate(
                    local_model.get_weights(),
                    neighbor_model.get_weights(),
                    model_template=models[i],
                    data_loader=memory_loader,
                    loss_fn=loss_fn,
                    sigma=sigma,
                    alpha=alpha
                )
                local_model.set_weights(updated_weights)
                updated_model = local_model
            else:
                updated_model = aggregate_models(local_model, neighbor_model, memory_loader, use_wcm=False, sigma=sigma, alpha=alpha)

            new_models.append(updated_model)

        models = new_models

        # Evaluation
        avg_acc = np.mean([evaluate(models[i], test_loaders[i]) for i in range(num_nodes)])
        avg_loss = np.mean([
            np.mean([loss_fn(y_batch, models[i](x_batch, training=False)).numpy()
                     for x_batch, y_batch in test_loaders[i]])
            for i in range(num_nodes)
        ])

        print(f"[Round {rnd+1}] Average Test Accuracy: {avg_acc:.5f} | Loss: {avg_loss:.5f}")
        acc_list.append(avg_acc)
        loss_list.append(avg_loss)

    return acc_list, loss_list
