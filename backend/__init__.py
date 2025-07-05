import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from .thermal_model import ThermalModel, predict_pinn_surrogate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Root route for status
    @app.route('/')
    def index():
        return 'Thermal twin dashboard backend is running normally!'

    def validate_thermal_params(data):
        """Validate and sanitize thermal parameters with physical constraints"""
        try:
            power_load = float(data.get('power_load', 50))
            airflow_rate = float(data.get('airflow_rate', 1.0))
            ambient_temp = float(data.get('ambient_temp', 25))
            tim_conductivity = float(data.get('tim_conductivity', 5.0))
            nx = int(data.get('nx', 50))
        except Exception as e:
            raise ValueError(f"Invalid parameter type: {e}")
        # Clamp to physical ranges
        if not (0 <= power_load <= 10000):
            raise ValueError("power_load must be between 0 and 10,000 W")
        if not (0.01 <= airflow_rate <= 100):
            raise ValueError("airflow_rate must be between 0.01 and 100 m³/min")
        if not (-50 <= ambient_temp <= 100):
            raise ValueError("ambient_temp must be between -50 and 100 °C")
        if not (0.1 <= tim_conductivity <= 100):
            raise ValueError("tim_conductivity must be between 0.1 and 100 W/mK")
        if not (5 <= nx <= 1000):
            raise ValueError("nx must be between 5 and 1000")
        return dict(
            power_load=power_load,
            airflow_rate=airflow_rate,
            ambient_temp=ambient_temp,
            tim_conductivity=tim_conductivity,
            nx=nx
        )

    @app.before_request
    def log_request_info():
        logging.info(f"Request: {request.method} {request.path} | Data: {request.get_json(silent=True)}")

    @app.after_request
    def log_response_info(response):
        # Avoid logging body for direct passthrough responses (e.g., static files)
        if not getattr(response, 'direct_passthrough', False):
            logging.info(f"Response: {response.status} | {response.get_data(as_text=True)[:200]}")
        else:
            logging.info(f"Response: {response.status} | [direct_passthrough]")
        return response

    # API endpoint for thermal prediction
    @app.route('/api/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            params = validate_thermal_params(data)
            model = ThermalModel(nx=params['nx'])
            temperatures = model.solve_heat_equation(
                params['power_load'], params['airflow_rate'], params['ambient_temp'], params['tim_conductivity']
            )
            positions = np.linspace(0, model.length, model.nx)
            temp_profile = {
                'positions': positions.tolist(),
                'temperatures': temperatures.tolist()
            }
            return jsonify(temp_profile)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # API endpoint for surrogate model prediction
    @app.route('/api/predict_surrogate', methods=['POST'])
    def predict_surrogate_api():
        try:
            data = request.get_json()
            params = validate_thermal_params(data)
            # Use profile-wise canonical PINN surrogate logic directly
            y_pred = predict_pinn_surrogate(params, nx=params['nx'])
            length = 0.1  # Default length, or use params if available
            positions = np.linspace(0, length, params['nx'])
            return jsonify({
                'positions': positions.tolist(),
                'temperatures': y_pred.tolist()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # Health check endpoint
    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok', 'message': 'Thermal twin backend healthy.'}), 200

    # Serve surrogate model metadata JSON
    @app.route('/surrogate_model_meta.json')
    def surrogate_model_meta():
        return send_from_directory(
            os.path.dirname(os.path.abspath(__file__)),
            'surrogate_model_meta.json',
            mimetype='application/json'
        )

    return app