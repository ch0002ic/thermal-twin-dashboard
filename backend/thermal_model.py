import numpy as np
import torch
import os
from backend.pinn_surrogate import select_pinn_for_params

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

def predict_pinn_surrogate(params, nx=50, device='cpu', debug=False, return_model_path=False):
    """Run the PINN surrogate for given params, return prediction (and optionally model path/arch)"""
    pl = params['power_load']
    af = params['airflow_rate']
    at = params['ambient_temp']
    tc = params['tim_conductivity']
    model, arch, model_path = select_pinn_for_params((pl, af, at, tc), nx=nx, backend_dir=BACKEND_DIR, debug=debug)
    x = torch.linspace(0, 0.1, nx, device=device).unsqueeze(0)
    params_tensor = torch.tensor([[pl, af, at, tc]], dtype=torch.float32, device=device)
    params_exp = params_tensor.unsqueeze(1).expand(-1, nx, -1)
    inp = torch.cat([params_exp, x.unsqueeze(-1)], dim=-1)
    with torch.no_grad():
        T_pred = model(inp).cpu().numpy().flatten()
    if return_model_path:
        return T_pred, model_path, arch
    return T_pred

# --- PINN surrogate unified interface ---
def predict_pinn(params, ensemble=None, local=None, local_hard_edge_models=None, nx=50, device='cpu'):
    """Predict using the best available PINN surrogate (ensemble/local/hard edge)"""
    return predict_pinn_surrogate(params, ensemble=ensemble, local=local, local_hard_edge_models=local_hard_edge_models, nx=nx, device=device)

class ThermalModel:
    def __init__(self, length=0.1, nx=50, base_thermal_diffusivity=1e-5, rho=1000, cp=900):
        self.length = length
        self.nx = nx
        self.dx = length / (nx - 1)
        self.base_alpha = base_thermal_diffusivity
        self.rho = rho
        self.cp = cp

    def solve_heat_equation(self, power_load, airflow_rate, ambient_temp, tim_conductivity, time_steps=100, boundary='convective', uniform_source=False, mode='transient', tol=1e-8):
        # Thermal diffusivity definition
        alpha = self.base_alpha * (tim_conductivity / 5.0)
        
        if mode == 'analytical':
            # Direct analytical solution for steady state
            return self._solve_steady_state_analytical(power_load, ambient_temp, tim_conductivity, boundary, uniform_source)
        
        # Time step for numerical stability
        dt = 0.8 * self.dx**2 / (2 * alpha)
        T = np.ones(self.nx) * ambient_temp
        
        # Heat source formulation
        if uniform_source:
            heat_source_vol = power_load / self.length  # W/m³
        else:
            heat_source_vol = np.zeros(self.nx)
            source_start = int(0.4 * self.nx)
            source_end = int(0.6 * self.nx)
            heat_source_vol[source_start:source_end] = power_load / (self.length * 0.2)
        
        h_conv = 10 + airflow_rate * 20
        
        # Transient solution
        for step in range(time_steps):
            T_new = T.copy()
            for i in range(1, self.nx - 1):
                # Heat equation: ∂T/∂t = α∇²T + q/(ρcp)
                diffusion = alpha * (T[i+1] - 2*T[i] + T[i-1]) / self.dx**2
                if uniform_source:
                    source = heat_source_vol / (self.rho * self.cp)
                else:
                    source = heat_source_vol[i] / (self.rho * self.cp)
                
                T_new[i] = T[i] + dt * (diffusion + source)
            
            # Boundary conditions
            if boundary == 'convective':
                T_new[0] = ambient_temp + (T_new[1] - ambient_temp) * np.exp(-h_conv * dt / (self.rho * self.cp))
                T_new[-1] = ambient_temp + (T_new[-2] - ambient_temp) * np.exp(-h_conv * dt / (self.rho * self.cp))
            elif boundary == 'insulated':
                T_new[0] = T_new[1]  # dT/dx = 0
                T_new[-1] = T_new[-2]  # dT/dx = 0
            
            if np.max(np.abs(T_new - T)) < tol:
                break
            T = T_new
        
        return T
    
    def _solve_steady_state_analytical(self, power_load, ambient_temp, tim_conductivity, boundary='insulated', uniform_source=True):
        """Direct analytical solution for steady-state: d²T/dx² + q/k = 0"""
        x = np.linspace(0, self.length, self.nx)
        
        if uniform_source and boundary == 'insulated':
            # Analytical solution: T(x) = T_ambient + (q/(2k)) * (xL - x²)
            q_vol = power_load / self.length  # W/m³
            k = tim_conductivity
            T = ambient_temp + (q_vol / (2 * k)) * (x * self.length - x**2)
        else:
            # Numerical solution for all other cases
            T = self.solve_heat_equation(power_load, 0, ambient_temp, tim_conductivity,
                                         time_steps=10000, boundary=boundary,
                                         uniform_source=uniform_source, mode='transient')
        return T

# If PINN surrogate prediction is needed, use the new profile-wise PINN logic as in compare_models.py
