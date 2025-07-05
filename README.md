# Thermal twin dashboard

## Project overview
An interactive dashboard for thermal system simulation, featuring a React frontend and a Flask backend. Users can adjust system parameters and instantly visualize predicted temperature profiles, demonstrating digital twin and physics-informed AI concepts for thermal management.

---

## Getting started
### Prerequisites
- **Python 3.8+**
- **Node.js** (v16+ recommended) & npm

---

## Backend setup (Flask)
1. **Create and activate a virtual environment (recommended)**
   ```sh
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install dependencies**
   ```sh
   pip install flask flask-cors plotly
   ```
3. **Run the Flask server**
   ```sh
   export FLASK_APP=__init__.py
   export FLASK_ENV=development
   flask run
   ```
   The backend will be available at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

   **Notes**
   - The backend serves surrogate model metadata at `/surrogate_model_meta.json` for the frontend dashboard.
   - Ensure all required model files (e.g., `pinn_model_1.pt`, local/grid models, and their `.json` arch files) and `surrogate_model_meta.json` are present in the `backend/` directory. The training scripts save them there by default.
   - If you see 404/500 errors for surrogate predictions or metadata, check that these files exist in `backend/`.

---

## Frontend setup (React)
1. **Install dependencies**
   ```sh
   cd frontend
   npm install
   ```
2. **Start the React development server**
   ```sh
   npm start
   ```
   The frontend will be available at [http://localhost:3000/](http://localhost:3000/).

---

## Usage
- Open the React app in your browser.
- Adjust parameters (power load, airflow rate, ambient temp, etc.) in the dashboard form.
- Select the model (physics or surrogate) and click **Predict** to visualize the temperature profile returned by the backend.
- When the surrogate model is selected, the dashboard displays model training info (date, epochs, etc.) fetched from the backend.

---

## Testing & validation
This project uses a comprehensive pytest suite to validate the physics and numerical accuracy of the backend model.

### Run all tests
1. From the project root, ensure your backend dependencies are installed and (optionally) activate your virtual environment
   ```sh
   cd backend
   source venv/bin/activate  # if using a virtual environment
   pip install -r requirements.txt  # or install pytest, numpy, etc. as needed
   cd ..
   ```
2. Run all tests from the project root
   ```sh
   PYTHONPATH=. pytest backend/tests/
   ```
   This will execute all backend tests, including:
   - Steady-state and boundary condition checks
   - Heat source and airflow effects
   - TIM conductivity and numerical stability
   - Analytical validation (comparison to theoretical solution)
   - Grid convergence (numerical accuracy with mesh refinement)

All tests should pass before making further changes or deploying the backend.

---

## Model assumptions & parameters
- **1D heat conduction**: Simulates a 1D rod with user-adjustable length, discretized into `nx` nodes.
- **Material properties**: Density (`rho`), specific heat (`cp`), and thermal conductivity (`tim_conductivity`) are parameterized and can be tuned for different materials.
- **Boundary conditions**: Supports insulated and convective boundaries. Airflow rate affects convective heat transfer at boundaries.
- **Heat source**: Uniform or localized power input (W) can be specified. For uniform source, the model assumes even distribution along the rod.
- **Numerical method**: Explicit finite-difference time-marching for transient and steady-state solutions. Analytical solution is used for validation.
- **Units**: SI units throughout (meters, seconds, Watts, Kelvin, etc.).

---

## Notes
- The backend now uses a unified, production-ready PINN surrogate logic for all surrogate predictions. No legacy or mock code remains.
- For production, configure CORS and deployment settings as needed.
