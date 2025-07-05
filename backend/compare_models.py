import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from .thermal_model import ThermalModel
from .pinn_surrogate import select_pinn_for_params

# NOTE: All PINN model selection and prediction is handled via select_pinn_for_params

# Parameter grid (low, mid, high for each)
power_loads = [0, 500, 1000]
airflow_rates = [0.01, 5, 10]
ambient_temps = [0, 25, 50]
tim_conductivities = [0.1, 5, 10]
nx = 50
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print("Comparing profile-wise PINN surrogate (ensemble/local) to physics model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    for (power, air, amb, tim) in itertools.product(power_loads, airflow_rates, ambient_temps, tim_conductivities):
        params = {
            'power_load': power,
            'airflow_rate': air,
            'ambient_temp': amb,
            'tim_conductivity': tim
        }
        # Physics model
        model = ThermalModel(nx=nx)
        T_physics = model.solve_heat_equation(power, air, amb, tim)
        # Use select_pinn_for_params to get the correct PINN model for these params
        pinn_model, arch, model_path = select_pinn_for_params([power, air, amb, tim], nx=nx, backend_dir=BACKEND_DIR)
        # Prepare input for PINN
        x_n = np.array([power, air, amb, tim], dtype=np.float32)
        positions = np.linspace(0, 0.1, nx).astype(np.float32)
        X_full = np.tile(x_n, (nx, 1))
        X_full = np.concatenate([X_full, positions[:, None]], axis=1)
        X_full = X_full[None, :, :]  # [1, nx, 5]
        X_tensor = torch.tensor(X_full, dtype=torch.float32, device='cpu')
        with torch.no_grad():
            T_model = pinn_model(X_tensor).cpu().numpy().flatten()
        # Metrics
        mae = np.mean(np.abs(T_physics - T_model))
        temp_range = np.max(T_physics) - np.min(T_physics)
        rel_mae = mae / temp_range if temp_range > 0 else 0
        results.append({
            'power_load': power,
            'airflow_rate': air,
            'ambient_temp': amb,
            'tim_conductivity': tim,
            'mae': mae,
            'rel_mae': rel_mae
        })
        print(f"Params: {params} | MAE: {mae:.4f} | Rel MAE: {rel_mae:.4f}")

    # Summary
    maes = [r['mae'] for r in results]
    rel_maes = [r['rel_mae'] for r in results]
    print(f"\nGrid search complete.\nMean MAE: {np.mean(maes):.4f}, Max MAE: {np.max(maes):.4f}")
    print(f"Mean rel MAE: {np.mean(rel_maes):.4f}, Max rel MAE: {np.max(rel_maes):.4f}")

    # Save results to CSV for further analysis
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(BACKEND_DIR, 'compare_pinn_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Plot MAE vs. parameter (optional)
    fig = plt.figure(figsize=(8,5))
    plt.plot([r['mae'] for r in results], 'o-')
    plt.title(f"Profile-wise PINN vs physics model: MAE across grid")
    plt.xlabel('Grid point')
    plt.ylabel('MAE (°C)')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(BACKEND_DIR, 'mae_grid_plot.png')
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"MAE grid plot saved to {plot_path}")

    # Print the worst-case (max MAE) parameter set(s)
    max_mae = np.max(maes)
    worst_cases = [r for r in results if np.isclose(r['mae'], max_mae)]
    print("\nWorst-case parameter set(s):")
    for wc in worst_cases:
        print(wc)

    # Print the top 5 highest MAE parameter sets
    sorted_results = sorted(results, key=lambda r: r['mae'], reverse=True)
    print("\nTop 5 highest MAE parameter sets:")
    for r in sorted_results[:5]:
        print(r)

    # Visualize the worst-case (max MAE) profile
    if results:
        max_mae = max(r['mae'] for r in results)
        worst_cases = [r for r in results if np.isclose(r['mae'], max_mae)]
        print("\nWorst-case parameter set(s):")
        for wc in worst_cases:
            print(wc)
        if worst_cases:
            wc = worst_cases[0]
            print(f"\nPlotting worst-case profile: {wc}")
            model = ThermalModel(nx=nx)
            T_physics = model.solve_heat_equation(wc['power_load'], wc['airflow_rate'], wc['ambient_temp'], wc['tim_conductivity'])
            params_wc = {
                'power_load': wc['power_load'],
                'airflow_rate': wc['airflow_rate'],
                'ambient_temp': wc['ambient_temp'],
                'tim_conductivity': wc['tim_conductivity']
            }
            # Use select_pinn_for_params for the worst-case
            pinn_model, arch, model_path = select_pinn_for_params([
                wc['power_load'], wc['airflow_rate'], wc['ambient_temp'], wc['tim_conductivity']
            ], nx=nx, backend_dir=BACKEND_DIR)
            x_n = np.array([
                wc['power_load'], wc['airflow_rate'], wc['ambient_temp'], wc['tim_conductivity']
            ], dtype=np.float32)
            positions = np.linspace(0, model.length, nx)
            X_full = np.tile(x_n, (nx, 1))
            X_full = np.concatenate([X_full, positions[:, None]], axis=1)
            X_full = X_full[None, :, :]
            X_tensor = torch.tensor(X_full, dtype=torch.float32, device='cpu')
            with torch.no_grad():
                T_model = pinn_model(X_tensor).cpu().numpy().flatten()
            fig = plt.figure(figsize=(8,5))
            plt.plot(positions, T_physics, label='Physics model', lw=2)
            plt.plot(positions, T_model, '--', label='Profile-wise PINN', lw=2)
            plt.title(f"Worst-case profile (MAE={wc['mae']:.3f})\n{params_wc}")
            plt.xlabel('Position')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(BACKEND_DIR, 'worst_case_profile.png')
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Worst-case profile plot saved to {plot_path}")
