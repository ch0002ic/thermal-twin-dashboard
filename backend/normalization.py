import numpy as np

X_MIN = np.array([0, 0.01, 0, 0.1], dtype=np.float32)
X_MAX = np.array([1000, 10, 50, 10], dtype=np.float32)
Y_MIN = 0.0
Y_MAX = 200.0

def normalize_X(X):
    return (X - X_MIN) / (X_MAX - X_MIN)

def denormalize_X(Xn):
    return Xn * (X_MAX - X_MIN) + X_MIN

def normalize_Y(Y):
    return (Y - Y_MIN) / (Y_MAX - Y_MIN)

def denormalize_Y(Yn):
    return Yn * (Y_MAX - Y_MIN) + Y_MIN
