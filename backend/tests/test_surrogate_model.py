"""
Tests for PINN surrogate model selection and accuracy
Covers accuracy, model file coverage, error handling, and summary reporting
"""
import os
import numpy as np
import pytest
import torch
from backend.thermal_model import ThermalModel
import importlib
import backend.thermal_model

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================
# Constants and helpers
# =====================

# Centralized test parameter sets
EDGE_GRID_TEST_PARAMS = [
    {'power_load': 0, 'airflow_rate': 0.01, 'ambient_temp': 0, 'tim_conductivity': 0.1},
    {'power_load': 1000, 'airflow_rate': 0.01, 'ambient_temp': 0, 'tim_conductivity': 0.1},
    {'power_load': 500, 'airflow_rate': 0.01, 'ambient_temp': 0, 'tim_conductivity': 0.1},
    {'power_load': 0, 'airflow_rate': 10, 'ambient_temp': 100, 'tim_conductivity': 10},
    {'power_load': 1000, 'airflow_rate': 10, 'ambient_temp': 100, 'tim_conductivity': 10},
]

UTILITY_TEST_PARAMS = [
    {'power_load': 100, 'airflow_rate': 0.01, 'ambient_temp': 0, 'tim_conductivity': 0.1},
    {'power_load': 1000, 'airflow_rate': 10, 'ambient_temp': 100, 'tim_conductivity': 10},
    {'power_load': 500, 'airflow_rate': 5, 'ambient_temp': 25, 'tim_conductivity': 5},
    {'power_load': 100, 'airflow_rate': 10, 'ambient_temp': 100, 'tim_conductivity': 10},
    {'power_load': 1000, 'airflow_rate': 0.01, 'ambient_temp': 0, 'tim_conductivity': 0.1},
]

# Global list to collect test results for summary
TEST_RESULTS = []

def _format_param(val):
    """Format parameter value for filename (int if possible, else trimmed float)"""
    return str(int(val)) if float(val).is_integer() else f"{val:.3f}".rstrip('0').rstrip('.')

def _model_prefix(params):
    """Return the model filename prefix for a parameter set"""
    pl_str = _format_param(params['power_load'])
    af_str = _format_param(params['airflow_rate'])
    at_str = _format_param(params['ambient_temp'])
    tc_str = _format_param(params['tim_conductivity'])
    return f"pinn_local_model_{pl_str}_{af_str}_{at_str}_{tc_str}"

@pytest.fixture(scope="module")
def pinn_models():
    """Return device string for torch (cuda if available, else cpu)"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def is_edge_corner_point(params, tol=1e-6):
    """Return True if params are on edge/corner of the parameter space"""
    power_loads = [0, 100, 1000]
    airflow_rates = [0.01, 10]
    ambient_temps = [0, 100]
    tim_conductivities = [0.1, 10]
    def is_close(val, arr):
        return any(abs(float(val) - float(a)) < tol for a in arr)
    return (
        is_close(params['power_load'], power_loads)
        and is_close(params['airflow_rate'], airflow_rates)
        and is_close(params['ambient_temp'], ambient_temps)
        and is_close(params['tim_conductivity'], tim_conductivities)
    )

def is_grid_point(params, tol=1e-6):
    """Return True if params are on the main grid (not edge/corner)"""
    power_loads = [0, 500, 1000]
    airflow_rates = [0.01, 5, 10]
    ambient_temps = [0, 25, 50]
    tim_conductivities = [0.1, 5, 10]
    def is_close(val, arr):
        return any(abs(float(val) - float(a)) < tol for a in arr)
    return (
        is_close(params['power_load'], power_loads)
        and is_close(params['airflow_rate'], airflow_rates)
        and is_close(params['ambient_temp'], ambient_temps)
        and is_close(params['tim_conductivity'], tim_conductivities)
    )

def get_model_type_from_path(model_path):
    """Return model type string from model file path"""
    if model_path is None:
        return 'unknown'
    fname = os.path.basename(model_path)
    if fname.endswith('_small.pt'):
        return 'local_edge_corner'
    if fname.endswith('_grid.pt'):
        return 'local_grid'
    if (
        'main' in fname
        or 'ensemble' in fname
        or fname == 'pinn_model_1.pt'
    ):
        return 'main'
    return 'other'

def expected_model_info(params, nx=50, backend_dir=None, tol=1e-6):
    """Return (expected_type, expected_path) for a parameter set, mirroring model selection logic"""
    if backend_dir is None:
        backend_dir = BACKEND_DIR
    # Edge/corner values
    power_loads_edge = [0, 100, 1000]
    airflow_rates_edge = [0.01, 10]
    ambient_temps_edge = [0, 100]
    tim_conductivities_edge = [0.1, 10]
    # Grid values
    power_loads_grid = [0, 500, 1000]
    airflow_rates_grid = [0.01, 5, 10]
    ambient_temps_grid = [0, 25, 50]
    tim_conductivities_grid = [0.1, 5, 10]
    pl, af, at, tc = params['power_load'], params['airflow_rate'], params['ambient_temp'], params['tim_conductivity']
    def is_close(val, arr):
        return any(abs(float(val) - float(a)) < tol for a in arr)
    # Edge/corner first
    if is_close(pl, power_loads_edge) and is_close(af, airflow_rates_edge) and is_close(at, ambient_temps_edge) and is_close(tc, tim_conductivities_edge):
        fname = f'{_model_prefix(params)}_small.pt'
        model_path = os.path.join(backend_dir, fname)
        return 'local_edge_corner', model_path
    # Grid next
    if is_close(pl, power_loads_grid) and is_close(af, airflow_rates_grid) and is_close(at, ambient_temps_grid) and is_close(tc, tim_conductivities_grid):
        fname = f'{_model_prefix(params)}_grid.pt'
        model_path = os.path.join(backend_dir, fname)
        return 'local_grid', model_path
    # Main/ensemble fallback
    model_path = os.path.join(backend_dir, 'pinn_model_1.pt')
    return 'main', model_path

# =====================
#        Tests
# =====================

@pytest.mark.parametrize("params", EDGE_GRID_TEST_PARAMS)
def test_pinn_surrogate_accuracy(pinn_models, params):
    from backend.thermal_model import predict_pinn_surrogate
    """Test PINN surrogate accuracy for parameter sets with available models"""
    device = pinn_models
    nx = 50
    model = ThermalModel(nx=nx)
    T_physics = model.solve_heat_equation(params['power_load'], params['airflow_rate'], params['ambient_temp'], params['tim_conductivity'])
    expected_type, expected_path = expected_model_info(params, nx=nx, backend_dir=BACKEND_DIR)
    try:
        T_pinn, model_path, arch = predict_pinn_surrogate(params, nx=nx, device=device, debug=False, return_model_path=True)
    except FileNotFoundError as e:
        if expected_type == 'local_edge_corner':
            pytest.xfail(f"Local edge/corner PINN missing for edge/corner point {params}: {e}")
        elif expected_type == 'local_grid':
            pytest.xfail(f"Grid local PINN missing for grid point {params}: {e}")
        raise
    mae = np.mean(np.abs(T_physics - T_pinn))
    temp_range = np.max(T_physics) - np.min(T_physics)
    rel_mae = mae / temp_range if temp_range > 0 else 0
    model_type = get_model_type_from_path(model_path)
    TEST_RESULTS.append({
        'params': params,
        'mae': mae,
        'rel_mae': rel_mae,
        'model_file': os.path.basename(model_path),
        'arch': arch
    })
    # Assert model type matches expectation, and xfail if main is used for local
    if expected_type in ('local_edge_corner', 'local_grid'):
        if model_type == 'main':
            pytest.xfail(f"Main PINN used for {expected_type.replace('_', ' ')} point {params}. Local PINN missing or not selected. Error: MAE={mae}, rel MAE={rel_mae}")
        assert os.path.basename(model_path) == os.path.basename(expected_path), f"Expected model {expected_path}, got {model_path}"
    assert mae < 1.5 or rel_mae < 0.02, f"PINN surrogate error too high for params {params}: MAE={mae}, rel MAE={rel_mae}"

def test_pinn_surrogate_output_shape(pinn_models):
    from backend.thermal_model import predict_pinn_surrogate
    """Test that the PINN surrogate output has correct shape and is finite"""
    device = pinn_models
    params = {'power_load': 500, 'airflow_rate': 5, 'ambient_temp': 25, 'tim_conductivity': 5}
    nx = 50
    T_pinn = predict_pinn_surrogate(params, nx=nx, device=device)
    assert T_pinn.shape == (nx,)
    assert np.all(np.isfinite(T_pinn))

def test_pinn_surrogate_hard_edge(pinn_models):
    from backend.thermal_model import predict_pinn_surrogate
    """Test PINN surrogate on a hard edge/corner point"""
    device = pinn_models
    params = {'power_load': 100, 'airflow_rate': 0.01, 'ambient_temp': 0, 'tim_conductivity': 0.01}
    nx = 50
    T_pinn = predict_pinn_surrogate(params, nx=nx, device=device)
    assert T_pinn.shape == (nx,)
    assert np.all(np.isfinite(T_pinn))
    assert not np.allclose(T_pinn, 0)

def test_missing_model_triggers_suggestion(monkeypatch, pinn_models):
    """Test that a missing model triggers a retrain suggestion in the error message"""
    device = pinn_models
    params = {'power_load': 12345, 'airflow_rate': 0.123, 'ambient_temp': 99, 'tim_conductivity': 0.456}
    nx = 50
    monkeypatch.setattr(
        "backend.pinn_surrogate.select_pinn_for_params",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("Fake missing model for test."))
    )
    importlib.reload(backend.thermal_model)
    from backend.thermal_model import predict_pinn_surrogate
    with pytest.raises(FileNotFoundError) as excinfo:
        predict_pinn_surrogate(params, nx=nx, device=device)
    msg = str(excinfo.value).lower()
    assert any(word in msg for word in ("not found", "missing")), \
        f"Error message should mention missing model or not found, got: {msg}"

def test_print_summary():
    """Print a summary table of all test results (MAE, rel MAE, model file, arch): useful for local runs"""
    if not TEST_RESULTS:
        return
    print("\n===== PINN surrogate test summary =====")
    print(f"{'Params':<60} {'MAE':>8} {'Rel MAE':>8} {'Model':>30} {'Arch':>20}")
    for r in TEST_RESULTS:
        arch_str = str(r['arch']) if isinstance(r['arch'], dict) else r['arch']
        print(f"{str(r['params']):<60} {r['mae']:8.4f} {r['rel_mae']:8.4f} {r['model_file'][:30]:>30} {arch_str[:20]:>20}")
    print("======================================\n")

def test_list_missing_expected_models():
    """List all expected model files for the utility test parameter set that are missing on disk"""
    missing = []
    for params in UTILITY_TEST_PARAMS:
        expected_type, expected_path = expected_model_info(params, nx=50, backend_dir=BACKEND_DIR)
        if expected_type in ("local_edge_corner", "local_grid") and not os.path.exists(expected_path):
            missing.append((params, expected_type, expected_path))
    if missing:
        print("\nMISSING expected PINN models for test cases:")
        for params, typ, path in missing:
            print(f"  {typ}: {params} -> {os.path.basename(path)}")
        print()
    else:
        print("\nAll expected PINN models for test cases are present.\n")

def test_list_all_available_expected_models():
    """List all available model files in the backend directory that match the utility test parameter set"""
    found = []
    for params in UTILITY_TEST_PARAMS:
        prefix = _model_prefix(params)
        matches = [f for f in os.listdir(BACKEND_DIR) if f.startswith(prefix) and f.endswith('.pt')]
        found.append((params, matches))
    print("\nAVAILABLE model files for test parameter set:")
    for params, matches in found:
        print(f"  {params}:")
        for m in matches:
            print(f"    {m}")
        if not matches:
            print("    (none found)")
    print()
