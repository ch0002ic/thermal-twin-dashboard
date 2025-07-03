import numpy as np
import pytest
from backend.thermal_model import ThermalModel

@pytest.fixture
def model():
    return ThermalModel()

def test_steady_state_no_heat_source(model):
    # No heat source, should remain at ambient
    T = model.solve_heat_equation(
        power_load=0, airflow_rate=1.0, ambient_temp=25, tim_conductivity=5.0, time_steps=200
    )
    assert np.allclose(T, 25, atol=0.5)

def test_boundary_condition_convection(model):
    # High airflow should keep boundaries close to ambient
    T = model.solve_heat_equation(
        power_load=100, airflow_rate=5.0, ambient_temp=20, tim_conductivity=5.0, time_steps=200
    )
    assert abs(T[0] - 20) < 2
    assert abs(T[-1] - 20) < 2

def test_heat_source_peak(model):
    # With heat source, center should be hottest
    T = model.solve_heat_equation(
        power_load=100, airflow_rate=1.0, ambient_temp=20, tim_conductivity=5.0, time_steps=200
    )
    center = T[len(T)//2]
    assert center == pytest.approx(np.max(T), abs=1e-6)
    assert center > T[0] and center > T[-1]

def test_airflow_cooling_effect(model):
    # Higher airflow should reduce max temperature
    T_low = model.solve_heat_equation(100, 0.5, 20, 5.0, 200)
    T_high = model.solve_heat_equation(100, 5.0, 20, 5.0, 200)
    assert np.max(T_high) < np.max(T_low)

def test_tim_conductivity_effect(model):
    # Higher TIM conductivity should lower max temperature
    T_low = model.solve_heat_equation(100, 1.0, 20, 2.0, 200)
    T_high = model.solve_heat_equation(100, 1.0, 20, 10.0, 200)
    assert np.max(T_high) < np.max(T_low)

def test_numerical_stability(model):
    # This test ensures the model is numerically stable (without NaN nor inf values) under typical conditions
    # and does not produce unrealistic temperature values.
    # It also checks that the temperature values are within a reasonable range.
    # For example, we expect temperatures to be within -100 to 200 degrees Celsius.
    # This range can be adjusted based on the expected physical limits of the system.
    T = model.solve_heat_equation(100, 1.0, 20, 5.0, 200)
    assert np.all(np.isfinite(T))
    assert np.all((T > -100) & (T < 200))

def analytical_steady_state(length, nx, power_load, ambient_temp, k=5.0):
    """Analytical solution: T(x) = T_ambient + (q/(2k)) * (xL - x²)"""
    x = np.linspace(0, length, nx)
    q = power_load / length  # W/m³
    T = ambient_temp + (q / (2 * k)) * (x * length - x**2)
    return T

def test_analytical_comparison(model):
    """Compare numerical with analytical steady-state solution"""
    length = model.length
    nx = model.nx
    power_load = 100
    ambient_temp = 25
    tim_conductivity = 5.0
    
    # Numerical solution using corrected analytical mode
    T_num = model.solve_heat_equation(
        power_load, 0, ambient_temp, tim_conductivity, 
        boundary='insulated', uniform_source=True, mode='analytical'
    )
    
    # Analytical solution
    T_ana = analytical_steady_state(length, nx, power_load, ambient_temp, k=tim_conductivity)
    
    # Error analysis
    mae = np.mean(np.abs(T_num - T_ana))
    max_err = np.max(np.abs(T_num - T_ana))
    temp_range = np.max(T_ana) - np.min(T_ana)
    
    rel_mae = mae / temp_range if temp_range > 0 else 0
    rel_max = max_err / temp_range if temp_range > 0 else 0
    
    # Physics-based validation criteria
    assert rel_mae < 0.05, f"Mean abs error too high: {rel_mae:.4f}"
    assert rel_max < 0.10, f"Max abs error too high: {rel_max:.4f}"
    assert np.allclose(T_num, T_ana, rtol=0.10, atol=0.01)

# Grid convergence test: error should decrease with finer mesh
def test_grid_convergence():
    power_load = 100
    ambient_temp = 25
    tim_conductivity = 5.0
    errors = []
    for nx in [20, 50, 100]:
        model = ThermalModel(nx=nx)
        T_num = model.solve_heat_equation(
            power_load, 0, ambient_temp, tim_conductivity, time_steps=2000, boundary='insulated', uniform_source=True, mode='transient'
        )
        T_ana = analytical_steady_state(model.length, nx, power_load, ambient_temp, k=tim_conductivity)
        mae = np.mean(np.abs(T_num - T_ana))
        errors.append(mae)
    assert errors[2] < errors[1] < errors[0], f"Grid convergence failed: {errors}"
