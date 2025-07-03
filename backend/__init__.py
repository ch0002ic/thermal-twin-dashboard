import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from .thermal_model import ThermalModel

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
        return 'Thermal twin dashboard backend is running.'

    # API endpoint for thermal prediction
    @app.route('/api/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        power_load = float(data.get('power_load', 50))
        airflow_rate = float(data.get('airflow_rate', 1.0))
        ambient_temp = float(data.get('ambient_temp', 25))
        tim_conductivity = float(data.get('tim_conductivity', 5.0))
        model = ThermalModel()
        temperatures = model.solve_heat_equation(power_load, airflow_rate, ambient_temp, tim_conductivity)
        positions = np.linspace(0, model.length, model.nx)
        temp_profile = {
            'positions': positions.tolist(),
            'temperatures': temperatures.tolist()
        }
        return jsonify(temp_profile)

    return app