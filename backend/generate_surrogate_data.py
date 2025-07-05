import numpy as np
from backend.thermal_model import ThermalModel
import os

# Settings
nx = 50
n_samples = 10000
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

X = []
Y = []
for _ in range(n_samples):
    pl = np.random.uniform(0, 1000)
    af = np.random.uniform(0.01, 10)
    at = np.random.uniform(0, 100)
    tc = np.random.uniform(0.1, 10)
    model = ThermalModel(nx=nx)
    T = model.solve_heat_equation(pl, af, at, tc)
    X.append([pl, af, at, tc])
    Y.append(T)
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)
np.savez(os.path.join(BACKEND_DIR, 'surrogate_train_data.npz'), X=X, Y=Y)
print(f"Saved surrogate_train_data.npz with {n_samples} samples in {BACKEND_DIR}")
