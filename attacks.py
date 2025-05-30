import numpy as np

def gaussian_attack(weights, std):
    return [w + np.random.normal(0, std, w.shape) for w in weights]

def sign_flip_attack(weights):
    return [-w for w in weights]

def hidden_attack(weights, malicious_weights, blend_ratio):
    return [(1 - blend_ratio) * w + blend_ratio * mw for w, mw in zip(weights, malicious_weights)]

def apply_attack(weights, attack_type, std=0.01, malicious_weights=None, blend_ratio=0.5):
    if attack_type == "none":
        return weights
    if attack_type == "gaussian":
        return gaussian_attack(weights, std)
    if attack_type == "sign_flip":
        return sign_flip_attack(weights)
    if attack_type == "hidden":
        if malicious_weights is None:
            raise ValueError("Hidden attack requires 'malicious_weights'.")
        return hidden_attack(weights, malicious_weights, blend_ratio)
    raise ValueError(f"Unknown attack type: {attack_type}")
