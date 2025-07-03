# Thermal twin dashboard
## Project overview
This project is an interactive "thermal twin" dashboard, featuring a React frontend and a Flask backend. Users can adjust thermal system parameters and instantly visualize predicted temperature profiles, demonstrating digital twin and physics-informed AI concepts for thermal management.
---
## Getting started
### Prerequisites
- Python 3.8+
- Node.js (v16+ recommended) & npm
---
## Backend setup (Flask)
1. **Create and activate a virtual environment (optional but recommended)**
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
   The backend will be available at `http://127.0.0.1:5000/`.
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
   The frontend will be available at `http://localhost:3000/`.
---
## Usage
- Open the React app in your browser.
- Adjust parameters (power load, airflow rate, ambient temp, etc.) in the dashboard form.
- Click "Predict" to visualize the temperature profile returned by the backend.
---
## Testing & validation
This project uses a comprehensive pytest suite to validate the physics and numerical accuracy of the backend model.

### Run all tests
1. From the project root, ensure your backend dependencies are installed and (optionally) activate your virtual environment:
   ```sh
   cd backend
   source venv/bin/activate  # if using a virtual environment
   pip install -r requirements.txt  # or install pytest, numpy, etc. as needed
   cd ..
   ```
2. Run all tests from the project root:
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
## Notes
- The backend currently returns a mock temperature profile. Replace the logic in `backend/__init__.py` with your surrogate model or PINN for real predictions.
- For production, configure CORS and deployment settings as needed.
---
## License
MIT
