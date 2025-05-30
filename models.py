import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class MNISTModel:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.model = models.Sequential([
            layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)  # No softmax – logits used for training
        ])

    def get_params(self):
        return self.model.get_weights()

    def set_params(self, weights):
        self.model.set_weights(weights)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def __call__(self, x, training=False):
        return self.model(x, training=training)


class CIFARModel:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(num_classes)
        ])

    def get_params(self):
        return self.model.get_weights()

    def set_params(self, weights):
        self.model.set_weights(weights)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights
    @property
    def trainable_variables(self):
        return self.model.trainable_variables  
    
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def __call__(self, x, training=False):
        return self.model(x, training=training)
