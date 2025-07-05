import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from .normalization import normalize_Y, denormalize_Y
import json
import itertools
import argparse

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Swish activation ---
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PINNSurrogateProfile(nn.Module):
    """
    Profile-wise PINN surrogate: takes the entire profile as input and predicts the full temperature profile
    Input: [batch, nx, 5] (params expanded for each x, x positions)
    Output: [batch, nx] (temperature profile)
    """
    def __init__(self, input_dim=250, hidden_dim=256, n_layers=8, activation='swish', nx=50):
        super().__init__()
        layers = []
        act = Swish() if activation == 'swish' else nn.Tanh()
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(act)
        layers.append(nn.Linear(hidden_dim, nx))  # Output is the full profile
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        batch, nx_, d = x.shape
        x_flat = x.reshape(batch, nx_ * d)
        out = self.net(x_flat)
        return out  # [batch, nx]

def physics_residual(model, params, x, device):
    """
    Compute the residual of the steady-state 1D heat equation at points x
    params: [power_load, airflow_rate, ambient_temp, tim_conductivity] (batch)
    x: positions (batch, nx)
    """
    power_load = params[:, 0:1]  # (batch, 1)
    airflow_rate = params[:, 1:2]
    ambient_temp = params[:, 2:3]
    tim_conductivity = params[:, 3:4]
    batch_size, nx = x.shape
    params_expanded = params.unsqueeze(1).expand(-1, nx, -1)  # (batch, nx, 4)
    x_ = x.clone().detach().requires_grad_(True)  # (batch, nx)
    inp = torch.cat([params_expanded, x_.unsqueeze(-1)], dim=-1)  # (batch, nx, 5)
    T_pred = model(inp)  # [batch, nx]
    dT_dx = torch.autograd.grad(T_pred, x_, grad_outputs=torch.ones_like(T_pred),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x_, grad_outputs=torch.ones_like(dT_dx),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    length = 0.1
    q = power_load / length  # (batch, 1)
    q = q.expand(-1, nx)
    k = tim_conductivity.expand(-1, nx)
    residual = k * d2T_dx2 + q
    return residual

def save_plot(fig, filename):
    plot_path = os.path.join(BACKEND_DIR, filename)
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

def plot_profiles(model, params, T_true, nx, device, title=None, filename=None):
    model.eval()
    with torch.no_grad():
        x = torch.linspace(0, 0.1, nx, device=device).unsqueeze(0)
        params_exp = params.unsqueeze(1).expand(-1, nx, -1)
        inp = torch.cat([params_exp, x.unsqueeze(-1)], dim=-1)
        T_pred = model(inp).cpu().numpy()
    fig = plt.figure(figsize=(7,4))
    plt.plot(np.linspace(0, 0.1, nx), T_true.cpu().numpy().flatten(), label='Physics')
    plt.plot(np.linspace(0, 0.1, nx), T_pred.flatten(), '--', label='PINN')
    plt.xlabel('Position')
    plt.ylabel('Temperature (°C)')
    plt.title(title or 'PINN vs physics')
    plt.legend()
    plt.tight_layout()
    if filename:
        save_plot(fig, filename)
    else:
        save_plot(fig, 'profile_plot.png')

def generate_targeted_samples(nx=50, n_target=1000, seed=456):
    from .thermal_model import ThermalModel
    np.random.seed(seed)
    X = []
    Y = []
    for _ in range(n_target):
        pl_ = 1000 + np.random.uniform(-10, 10)
        af_ = 0.01 + np.random.uniform(-0.01, 0.1)
        at_ = 50 + np.random.uniform(-1, 1)
        tc_ = 0.1 + np.random.uniform(-0.01, 0.01)
        model = ThermalModel(nx=nx)
        T = model.solve_heat_equation(pl_, af_, at_, tc_)
        X.append([pl_, af_, at_, tc_])
        Y.append(T)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y

def generate_hard_edge_samples(nx=50, n_target=1000, seed=789, edge_values=None):
    from .thermal_model import ThermalModel
    """
    Generate samples at the hardest domain edges/corners for targeted enrichment
    By default, uses min/max for each parameter. Optionally, custom edge_values can be provided as a dict.
    args:
        nx (int): Number of spatial points in the profile
        n_target (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        edge_values (dict, optional): Dict with keys 'power_loads', 'airflow_rates', 'ambient_temps', 'tim_conductivities',
            each mapping to a list of values to sample from. If None, uses default min/max.
    returns:
        X (np.ndarray): [n_target, 4] parameter sets at hard edges
        Y (np.ndarray): [n_target, nx] corresponding temperature profiles
    """
    np.random.seed(seed)
    X = []
    Y = []
    if edge_values is None:
        power_loads = [100, 1000]
        airflow_rates = [0.01, 10]
        ambient_temps = [0, 100]
        tim_conductivities = [0.01, 10]
    else:
        power_loads = edge_values.get('power_loads', [100, 1000])
        airflow_rates = edge_values.get('airflow_rates', [0.01, 10])
        ambient_temps = edge_values.get('ambient_temps', [0, 100])
        tim_conductivities = edge_values.get('tim_conductivities', [0.01, 10])
    model = ThermalModel(nx=nx)
    for _ in range(n_target):
        pl_ = np.random.choice(power_loads)
        af_ = np.random.choice(airflow_rates)
        at_ = np.random.choice(ambient_temps)
        tc_ = np.random.choice(tim_conductivities)
        T = model.solve_heat_equation(pl_, af_, at_, tc_)
        X.append([pl_, af_, at_, tc_])
        Y.append(T)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y

def get_all_hard_edges():
    power_loads = [100, 1000]
    airflow_rates = [0.01, 10]
    ambient_temps = [0, 100]
    tim_conductivities = [0.01, 10]
    return list(itertools.product(power_loads, airflow_rates, ambient_temps, tim_conductivities))

def train_pinn(
    X, Y, nx=50, batch_size=32, n_epochs=1000, lr=1e-3, physics_loss_weight=1.0, val_frac=0.1, plot_every=100, early_stop_patience=50,
    local_loss_weighting=True, edge_weight=10.0, hidden_dim=256, n_layers=8, activation='swish'
):
    from .thermal_model import ThermalModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 5 * nx
    model = PINNSurrogateProfile(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, nx=nx).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    N = X.shape[0]
    Yn = normalize_Y(Y)
    easy_idx = np.where((X[:,0] < 600) & (X[:,1] > 2) & (X[:,3] > 1))[0]
    hard_idx = np.setdiff1d(np.arange(N), easy_idx)
    X_easy, Yn_easy = X[easy_idx], Yn[easy_idx]
    X_hard, Yn_hard = X[hard_idx], Yn[hard_idx]
    X_edge, Y_edge = generate_targeted_samples(nx=nx, n_target=1000)
    Yn_edge = normalize_Y(Y_edge)
    X_hard_edge, Y_hard_edge = generate_hard_edge_samples(nx=nx, n_target=3000)
    Yn_hard_edge = normalize_Y(Y_hard_edge)
    X_train = np.concatenate([X_hard_edge, X_edge, X_hard, X_easy], axis=0)
    Yn_train = np.concatenate([Yn_hard_edge, Yn_edge, Yn_hard, Yn_easy], axis=0)
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    n_val = int(val_frac * len(X_train))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    Xtr, Ytr = X_train[train_idx], Yn_train[train_idx]
    Xval, Yval = X_train[val_idx], Yn_train[val_idx]
    if local_loss_weighting:
        weights = np.ones(len(Xtr), dtype=np.float32)
        for i, x in enumerate(Xtr):
            if any(np.allclose(x, xe, atol=1e-5) for xe in X_hard_edge):
                weights[i] = edge_weight
            elif any(np.allclose(x, xe, atol=1e-5) for xe in X_hard):
                weights[i] = 2.0
    else:
        weights = np.ones(len(Xtr), dtype=np.float32)
    train_losses, val_losses = [], []
    best_val = float('inf')
    patience = 0
    for epoch in range(n_epochs):
        model.train()
        batch_ids = np.random.choice(len(Xtr), batch_size, replace=False)
        params = torch.tensor(Xtr[batch_ids], dtype=torch.float32, device=device)
        T_true = torch.tensor(Ytr[batch_ids], dtype=torch.float32, device=device)
        batch_weights = torch.tensor(weights[batch_ids], dtype=torch.float32, device=device)
        x = torch.linspace(0, 0.1, nx, device=device).unsqueeze(0).repeat(batch_size, 1)
        optimizer.zero_grad()
        params_exp = params.unsqueeze(1).expand(-1, nx, -1)
        inp = torch.cat([params_exp, x.unsqueeze(-1)], dim=-1)
        T_pred = model(inp)
        data_loss = ((T_pred - T_true) ** 2).mean(dim=1)
        data_loss = (data_loss * batch_weights).mean()
        phys_res = physics_residual(model, params, x, device)
        phys_loss = torch.mean(phys_res**2)
        if epoch > 0 and abs(train_losses[-1] - data_loss.item()) < 1e-3:
            physics_loss_weight = min(physics_loss_weight * 1.05, 10.0)
        loss = data_loss + physics_loss_weight * phys_loss
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            params_val = torch.tensor(Xval, dtype=torch.float32, device=device)
            T_val = torch.tensor(Yval, dtype=torch.float32, device=device)
            x_val = torch.linspace(0, 0.1, nx, device=device).unsqueeze(0).repeat(len(Xval), 1)
            params_exp_val = params_val.unsqueeze(1).expand(-1, nx, -1)
            inp_val = torch.cat([params_exp_val, x_val.unsqueeze(-1)], dim=-1)
            T_pred_val = model(inp_val)
            val_loss = nn.MSELoss()(T_pred_val, T_val).item()
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = model.state_dict()
        else:
            patience += 1
        if patience > early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_state)
            break
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: train loss={loss.item():.4e}, val loss={val_loss:.4e}, data loss={data_loss.item():.4e}, phys loss={phys_loss.item():.4e}, phys_wt={physics_loss_weight:.2f}")
        if (epoch+1) % plot_every == 0:
            i = np.random.randint(len(Xval))
            plot_profiles(model, params_val[i:i+1], denormalize_Y(T_val[i:i+1]), nx, device, title=f'Epoch {epoch+1} (val sample)', filename=f'profile_plot_epoch_{epoch+1}.png')
    fig = plt.figure(figsize=(7,4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN training loss curves')
    plt.legend()
    plt.tight_layout()
    save_plot(fig, 'training_loss_curves.png')
    return model

def train_pinn_ensemble(X, Y, nx=50, n_models=3, **kwargs):
    models = []
    for i in range(n_models):
        print(f"Training ensemble PINN model {i+1}/{n_models}")
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X_shuf, Y_shuf = X[idx], Y[idx]
        model = train_pinn(X_shuf, Y_shuf, nx=nx, **kwargs)
        models.append(model)
    return models

def train_all_local_hard_edge_pinns(nx=50, n_target=1000):
    from .thermal_model import ThermalModel
    hard_edges = get_all_hard_edges()
    for edge in hard_edges:
        pl, af, at, tc = edge
        print(f"Training local PINN for hard edge: power_load={pl}, airflow_rate={af}, ambient_temp={at}, tim_conductivity={tc}")
        model = ThermalModel(nx=nx)
        X_edge = np.array([[pl, af, at, tc]] * n_target, dtype=np.float32)
        Y_edge = np.array([model.solve_heat_equation(pl, af, at, tc) for _ in range(n_target)], dtype=np.float32)
        local_model = train_pinn(X_edge, Y_edge, nx=nx, batch_size=32, n_epochs=1000, lr=1e-3, physics_loss_weight=1.0, plot_every=100, early_stop_patience=75, edge_weight=10.0)
        fname = f'pinn_local_model_{pl}_{af}_{at}_{tc}.pt'
        pinn_local_model_path = os.path.join(BACKEND_DIR, fname)
        torch.save(local_model.state_dict(), pinn_local_model_path)
        arch = {
            'hidden_dim': local_model.net[0].out_features,
            'n_layers': sum(isinstance(l, nn.Linear) for l in local_model.net) - 1,
            'activation': 'swish' if any(isinstance(l, Swish) for l in local_model.net) else 'tanh'
        }
        with open(pinn_local_model_path + '.json', 'w') as f:
            json.dump(arch, f)
        print(f"Local PINN for hard edge saved as '{pinn_local_model_path}'.")

def train_all_small_edge_pinns(nx=50, n_target=20000, n_epochs=800, hidden_dim=32, n_layers=2, activation='tanh', lr=1e-2, batch_size=64):
    from .thermal_model import ThermalModel
    """
    Train a small/shallow profile-wise PINN for every edge/corner (all combinations in the grid),
    using only data from that edge/corner, and save each model and its .json arch file
    """
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    # Use the exact values from the grid for all edge/corner cases
    power_loads = [0, 100, 500, 1000]
    airflow_rates = [0.01, 5, 10]
    ambient_temps = [0, 25, 50, 100]
    tim_conductivities = [0.1, 5, 10]  # match the grid, not just 0.01/10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for pl in power_loads:
        for af in airflow_rates:
            for at in ambient_temps:
                for tc in tim_conductivities:
                    # Only train for edge/corner values (at min or max of each param)
                    is_edge = (
                        pl in [min(power_loads), max(power_loads)] and
                        af in [min(airflow_rates), max(airflow_rates)] and
                        at in [min(ambient_temps), max(ambient_temps)] and
                        tc in [min(tim_conductivities), max(tim_conductivities)]
                    )
                    if not is_edge:
                        continue
                    print(f"Training small edge PINN: pl={pl}, af={af}, at={at}, tc={tc}")
                    X_edge = np.array([[pl, af, at, tc]] * n_target, dtype=np.float32)
                    model = ThermalModel(nx=nx)
                    Y_edge = np.array([model.solve_heat_equation(pl, af, at, tc) for _ in range(n_target)], dtype=np.float32)
                    input_dim = 5 * nx
                    pinn = PINNSurrogateProfile(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, nx=nx).to(device)
                    optimizer = torch.optim.AdamW(pinn.parameters(), lr=lr)
                    losses = []
                    for epoch in range(1, n_epochs+1):
                        pinn.train()
                        idx = np.random.choice(len(X_edge), batch_size, replace=False)
                        params = torch.tensor(X_edge[idx], dtype=torch.float32, device=device)
                        T_true = torch.tensor(Y_edge[idx], dtype=torch.float32, device=device)
                        x = torch.linspace(0, 0.1, nx, device=device).unsqueeze(0).repeat(batch_size, 1)
                        optimizer.zero_grad()
                        params_exp = params.unsqueeze(1).expand(-1, nx, -1)
                        inp = torch.cat([params_exp, x.unsqueeze(-1)], dim=-1)
                        T_pred = pinn(inp)
                        data_loss = ((T_pred - T_true) ** 2).mean()
                        data_loss.backward()
                        optimizer.step()
                        losses.append(data_loss.item())
                        if epoch % 50 == 0:
                            print(f"Epoch {epoch}: data loss={data_loss.item():.4e}")
                    # Save model and arch
                    pl_str = _format_param(pl)
                    af_str = _format_param(af)
                    at_str = _format_param(at)
                    tc_str = _format_param(tc)
                    fname = f'pinn_local_model_{pl_str}_{af_str}_{at_str}_{tc_str}_small.pt'
                    pinn_local_model_path = os.path.join(BACKEND_DIR, fname)
                    torch.save(pinn.state_dict(), pinn_local_model_path)
                    arch = {
                        'hidden_dim': hidden_dim,
                        'n_layers': n_layers,
                        'activation': activation
                    }
                    with open(pinn_local_model_path + '.json', 'w') as f:
                        json.dump(arch, f)
                    print(f"Saved small edge PINN to {pinn_local_model_path} and .json.")

def train_all_grid_local_pinns(nx=50, n_target=20000, n_epochs=800, hidden_dim=32, n_layers=2, activation='tanh', lr=1e-2, batch_size=64):
    from .thermal_model import ThermalModel
    """
    Train a small/shallow profile-wise PINN for every grid point (all combinations in the grid),
    using only data from that grid point, and save each model and its .json arch file
    """
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    power_loads = [0, 500, 1000]
    airflow_rates = [0.01, 5, 10]
    ambient_temps = [0, 25, 50]
    tim_conductivities = [0.1, 5, 10]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for pl in power_loads:
        for af in airflow_rates:
            for at in ambient_temps:
                for tc in tim_conductivities:
                    print(f"Training grid local PINN: pl={pl}, af={af}, at={at}, tc={tc}")
                    X_grid = np.array([[pl, af, at, tc]] * n_target, dtype=np.float32)
                    model = ThermalModel(nx=nx)
                    Y_grid = np.array([model.solve_heat_equation(pl, af, at, tc) for _ in range(n_target)], dtype=np.float32)
                    input_dim = 5 * nx
                    pinn = PINNSurrogateProfile(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, nx=nx).to(device)
                    optimizer = torch.optim.AdamW(pinn.parameters(), lr=lr)
                    for epoch in range(1, n_epochs+1):
                        pinn.train()
                        idx = np.random.choice(len(X_grid), batch_size, replace=False)
                        params = torch.tensor(X_grid[idx], dtype=torch.float32, device=device)
                        T_true = torch.tensor(Y_grid[idx], dtype=torch.float32, device=device)
                        x = torch.linspace(0, 0.1, nx, device=device).unsqueeze(0).repeat(batch_size, 1)
                        optimizer.zero_grad()
                        params_exp = params.unsqueeze(1).expand(-1, nx, -1)
                        inp = torch.cat([params_exp, x.unsqueeze(-1)], dim=-1)
                        T_pred = pinn(inp)
                        data_loss = ((T_pred - T_true) ** 2).mean()
                        data_loss.backward()
                        optimizer.step()
                        if epoch % 50 == 0:
                            print(f"Epoch {epoch}: data loss={data_loss.item():.4e}")
                    # Save model and arch
                    pl_str = _format_param(pl)
                    af_str = _format_param(af)
                    at_str = _format_param(at)
                    tc_str = _format_param(tc)
                    fname = f'pinn_local_model_{pl_str}_{af_str}_{at_str}_{tc_str}_grid.pt'
                    pinn_local_model_path = os.path.join(BACKEND_DIR, fname)
                    torch.save(pinn.state_dict(), pinn_local_model_path)
                    arch = {
                        'hidden_dim': hidden_dim,
                        'n_layers': n_layers,
                        'activation': activation
                    }
                    with open(pinn_local_model_path + '.json', 'w') as f:
                        json.dump(arch, f)
                    print(f"Saved grid local PINN to {pinn_local_model_path} and .json.")

def _format_param(val):
    # Use up to 3 decimal places, strip trailing zeros, avoid scientific notation
    if isinstance(val, int) or float(val).is_integer():
        return str(int(val))
    return f"{val:.3f}".rstrip('0').rstrip('.')

def select_pinn_for_params(params, nx=50, backend_dir=None, tol=1e-6, debug=True):
    """
    Select and load the correct PINN model for the given params
    If params match an edge/corner, load the corresponding small/shallow PINN
    Otherwise, load the main/ensemble PINN (default: 'pinn_model_1.pt')
    Returns: (model, arch_dict, model_path)
    """
    if backend_dir is None:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
    # Edge/corner values: use all grid edges, not just min/max
    power_loads = [0, 100, 1000]
    airflow_rates = [0.01, 10]
    ambient_temps = [0, 100]
    tim_conductivities = [0.1, 10]
    pl, af, at, tc = params
    def is_close(val, arr):
        return any(abs(float(val) - float(a)) < tol for a in arr)
    if is_close(pl, power_loads) and is_close(af, airflow_rates) and is_close(at, ambient_temps) and is_close(tc, tim_conductivities):
        # Use small edge/corner PINN with robust float formatting
        pl_str = _format_param(pl)
        af_str = _format_param(af)
        at_str = _format_param(at)
        tc_str = _format_param(tc)
        fname = f'pinn_local_model_{pl_str}_{af_str}_{at_str}_{tc_str}_small.pt'
        model_path = os.path.join(backend_dir, fname)
        json_path = model_path + '.json'
        if not os.path.exists(model_path):
            if debug:
                print(f"[select_pinn_for_params] Edge/corner PINN not found: {model_path}")
            raise FileNotFoundError(f"Edge/corner PINN not found: {model_path}")
        if debug:
            print(f"[select_pinn_for_params] Using SMALL EDGE PINN: {fname}")
        with open(json_path, 'r') as f:
            arch = json.load(f)
        model = PINNSurrogateProfile(input_dim=5*nx, hidden_dim=arch['hidden_dim'], n_layers=arch['n_layers'], activation=arch['activation'], nx=nx)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, arch, model_path
    # Try to match a local grid PINN for any exact grid point
    grid_power_loads = [0, 500, 1000]
    grid_airflow_rates = [0.01, 5, 10]
    grid_ambient_temps = [0, 25, 50]
    grid_tim_conductivities = [0.1, 5, 10]
    if (
        is_close(pl, grid_power_loads) and
        is_close(af, grid_airflow_rates) and
        is_close(at, grid_ambient_temps) and
        is_close(tc, grid_tim_conductivities)
    ):
        pl_str = _format_param(pl)
        af_str = _format_param(af)
        at_str = _format_param(at)
        tc_str = _format_param(tc)
        fname = f'pinn_local_model_{pl_str}_{af_str}_{at_str}_{tc_str}_grid.pt'
        model_path = os.path.join(backend_dir, fname)
        json_path = model_path + '.json'
        if os.path.exists(model_path):
            if debug:
                print(f"[select_pinn_for_params] Using GRID LOCAL PINN: {fname}")
            with open(json_path, 'r') as f:
                arch = json.load(f)
            model = PINNSurrogateProfile(input_dim=5*nx, hidden_dim=arch['hidden_dim'], n_layers=arch['n_layers'], activation=arch['activation'], nx=nx)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model, arch, model_path
    # Otherwise, use main/ensemble PINN (default: pinn_model_1.pt)
    model_path = os.path.join(backend_dir, 'pinn_model_1.pt')
    json_path = model_path + '.json'
    if debug:
        print(f"[select_pinn_for_params] Using MAIN/ENSEMBLE PINN: pinn_model_1.pt")
    with open(json_path, 'r') as f:
        arch = json.load(f)
    model = PINNSurrogateProfile(input_dim=5*nx, hidden_dim=arch['hidden_dim'], n_layers=arch['n_layers'], activation=arch['activation'], nx=nx)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, arch, model_path

nx = 50  # Default number of spatial points for all training functions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN surrogate training utility")
    parser.add_argument("--train_ensemble", action="store_true", help="Train ensemble PINN models")
    parser.add_argument("--train_local_hard_edge", action="store_true", help="Train local PINN for hardest edge region")
    parser.add_argument("--train_all_local_hard_edge_pinns", action="store_true", help="Train all local hard edge PINNs")
    parser.add_argument("--train_all_small_edge_pinns", action="store_true", help="Train all small/shallow edge PINNs for every edge/corner case")
    parser.add_argument("--train_all_grid_local_pinns", action="store_true", help="Train all small/shallow local PINNs for every grid point (not just edges/corners)")
    parser.add_argument("--train_all", action="store_true", help="Train all possible PINN surrogates (ensemble, all edge, all grid, etc.)")
    args = parser.parse_args()

    data_path = os.path.join(BACKEND_DIR, 'surrogate_train_data.npz')
    if args.train_all or args.train_ensemble:
        data = np.load(data_path)
        X = data['X']
        Y = data['Y']
        N, nx = Y.shape
        ensemble_models = train_pinn_ensemble(X, Y, nx=nx, batch_size=64, n_epochs=1000, lr=1e-3, physics_loss_weight=1.0, plot_every=100, early_stop_patience=75)
        for i, model in enumerate(ensemble_models):
            pinn_model_path = os.path.join(BACKEND_DIR, f'pinn_model_{i+1}.pt')
            torch.save(model.state_dict(), pinn_model_path)
            arch = {
                'hidden_dim': model.net[0].out_features,
                'n_layers': sum(isinstance(l, nn.Linear) for l in model.net) - 1,
                'activation': 'swish' if any(isinstance(l, Swish) for l in model.net) else 'tanh'
            }
            with open(pinn_model_path + '.json', 'w') as f:
                json.dump(arch, f)
            print(f"Trained ensemble PINN model {i+1} saved as '{pinn_model_path}'.")
    if args.train_all or args.train_local_hard_edge:
        print("Training local PINN surrogate for hardest edge region...")
        X_hard_edge, Y_hard_edge = generate_hard_edge_samples(nx=nx, n_target=3000)
        local_model = train_pinn(X_hard_edge, Y_hard_edge, nx=nx, batch_size=32, n_epochs=1000, lr=1e-3, physics_loss_weight=1.0, plot_every=100, early_stop_patience=75, edge_weight=10.0)
        pinn_local_model_path = os.path.join(BACKEND_DIR, 'pinn_local_model.pt')
        torch.save(local_model.state_dict(), pinn_local_model_path)
        arch = {
            'hidden_dim': local_model.net[0].out_features,
            'n_layers': sum(isinstance(l, nn.Linear) for l in local_model.net) - 1,
            'activation': 'swish' if any(isinstance(l, Swish) for l in local_model.net) else 'tanh'
        }
        with open(pinn_local_model_path + '.json', 'w') as f:
            json.dump(arch, f)
        print(f"Local PINN surrogate for hardest edge region saved as '{pinn_local_model_path}'.")
    if args.train_all or args.train_all_local_hard_edge_pinns:
        train_all_local_hard_edge_pinns(nx=nx, n_target=1000)
    if args.train_all or args.train_all_small_edge_pinns:
        train_all_small_edge_pinns(nx=nx, n_target=20000, n_epochs=800, hidden_dim=32, n_layers=2, activation='tanh', lr=1e-2, batch_size=64)
    if args.train_all or args.train_all_grid_local_pinns:
        train_all_grid_local_pinns(nx=nx, n_target=20000, n_epochs=800, hidden_dim=32, n_layers=2, activation='tanh', lr=1e-2, batch_size=64)
    if not any([args.train_ensemble, args.train_local_hard_edge, args.train_all_local_hard_edge_pinns, args.train_all_small_edge_pinns, args.train_all_grid_local_pinns, args.train_all]):
        parser.print_help()
