import { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

function App() {
  const [params, setParams] = useState({
    power_load: 50,
    airflow_rate: 1.0,
    ambient_temp: 25,
    sensor_placement: 'default',
    tim_conductivity: 5.0,
    microchannel_dim: 'default',
  });
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setParams((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post('/api/predict', {
        ...params,
        power_load: parseFloat(params.power_load),
        airflow_rate: parseFloat(params.airflow_rate),
        ambient_temp: parseFloat(params.ambient_temp),
        tim_conductivity: parseFloat(params.tim_conductivity),
      });
      setProfile(res.data);
    } catch (err) {
      setError('Prediction failed.');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Thermal twin dashboard</h1>
        <form onSubmit={handleSubmit} style={{ marginBottom: 24 }}>
          <label>
            Power load (W):
            <input type="number" name="power_load" value={params.power_load} onChange={handleChange} step="0.1" />
          </label>
          <label>
            Airflow rate (m³/min):
            <input type="number" name="airflow_rate" value={params.airflow_rate} onChange={handleChange} step="0.1" />
          </label>
          <label>
            Ambient temp (°C):
            <input type="number" name="ambient_temp" value={params.ambient_temp} onChange={handleChange} step="0.1" />
          </label>
          <label>
            Sensor placement:
            <input type="text" name="sensor_placement" value={params.sensor_placement} onChange={handleChange} />
          </label>
          <label>
            TIM conductivity (W/mK):
            <input type="number" name="tim_conductivity" value={params.tim_conductivity} onChange={handleChange} step="0.1" />
          </label>
          <label>
            Microchannel dim:
            <input type="text" name="microchannel_dim" value={params.microchannel_dim} onChange={handleChange} />
          </label>
          <button type="submit" disabled={loading} style={{ marginLeft: 16 }}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>
        {error && <div style={{ color: 'red' }}>{error}</div>}
        {profile && (
          <Plot
            data={[
              {
                x: profile.positions,
                y: profile.temperatures,
                type: 'scatter',
                mode: 'lines+markers',
                marker: { color: 'orange' },
                name: 'Temperature',
              },
            ]}
            layout={{
              title: 'Predicted temperature profile',
              xaxis: { title: 'Position' },
              yaxis: { title: 'Temperature (°C)' },
              autosize: true,
              paper_bgcolor: '#222',
              plot_bgcolor: '#222',
              font: { color: '#fff' },
            }}
            style={{ width: '100%', maxWidth: 600, height: 400 }}
            config={{ responsive: true }}
          />
        )}
      </header>
    </div>
  );
}

export default App;
