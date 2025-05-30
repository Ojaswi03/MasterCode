# WCM.py (TensorFlow version)

import numpy as np
import tensorflow as tf

def evaluate_model_loss(weights, model, data_loader, loss_fn):
    model.set_weights(weights)
    total_loss, total_samples = 0.0, 0
    for x_batch, y_batch in data_loader:
        logits = model(x_batch, training=False)
        loss = loss_fn(y_batch, logits)
        total_loss += loss.numpy() * x_batch.shape[0]
        total_samples += x_batch.shape[0]
    return total_loss / total_samples

def worst_case_aggregate(local_weights, neighbor_weights, model_template, data_loader,
                          loss_fn, sigma=0.01, alpha=0.1):
    """
    TensorFlow-compatible worst-case aggregation under communication noise.

    Args:
        local_weights (list of np.ndarray): Local model weights
        neighbor_weights (list of np.ndarray): Received neighbor model weights
        model_template (tf.keras.Model): Untrained copy of model architecture
        data_loader (tf.data.Dataset or generator): Memory/validation data loader
        loss_fn (callable): Keras loss function
        sigma (float): Std dev of noise threshold
        alpha (float): Update mixing factor

    Returns:
        list of np.ndarray: Aggregated model weights
    """
    local_loss = evaluate_model_loss(local_weights, model_template, data_loader, loss_fn)
    neighbor_loss = evaluate_model_loss(neighbor_weights, model_template, data_loader, loss_fn)

    if neighbor_loss > local_loss + sigma:
        return local_weights  # Reject neighbor
    else:
        return [(1 - alpha) * lw + alpha * nw for lw, nw in zip(local_weights, neighbor_weights)]
