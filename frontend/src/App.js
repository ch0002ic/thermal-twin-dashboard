import { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

// Config for parameter fields
const PARAM_FIELDS = [
  { name: 'power_load', label: 'Power load (W)', type: 'number', step: 0.1, default: 50 },
  { name: 'airflow_rate', label: 'Airflow rate (m³/min)', type: 'number', step: 0.1, default: 1.0 },
  { name: 'ambient_temp', label: 'Ambient temp (°C)', type: 'number', step: 0.1, default: 25 },
  { name: 'sensor_placement', label: 'Sensor placement', type: 'text', default: 'default' },
  { name: 'tim_conductivity', label: 'TIM conductivity (W/mK)', type: 'number', step: 0.1, default: 5.0 },
  { name: 'microchannel_dim', label: 'Microchannel dim', type: 'text', default: 'default' },
];

function getInitialParams() {
  const obj = {};
  PARAM_FIELDS.forEach(f => { obj[f.name] = f.default; });
  return obj;
}

function App() {
  const [params, setParams] = useState(getInitialParams());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelType, setModelType] = useState('physics'); // 'physics' or 'surrogate'
  const [positions, setPositions] = useState([]);
  const [temperatures, setTemperatures] = useState([]);
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    if (modelType === 'surrogate') {
      axios.get('/surrogate_model_meta.json')
        .then(res => setMeta(res.data))
        .catch(() => setMeta(null));
    } else {
      setMeta(null);
    }
  }, [modelType]);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setParams((prev) => ({
      ...prev,
      [name]: type === 'number' ? value : value,
    }));
    setError(null);
  };

  const handlePredict = async () => {
    const endpoint =
      modelType === 'physics'
        ? '/api/predict'
        : '/api/predict_surrogate';
    setError(null);
    setLoading(true);
    try {
      const reqParams = { ...params };
      // Convert number fields
      PARAM_FIELDS.forEach(f => {
        if (f.type === 'number') reqParams[f.name] = parseFloat(params[f.name]);
      });
      const response = await axios.post(endpoint, reqParams);
      setPositions(response.data.positions);
      setTemperatures(response.data.temperatures);
      setError(null);
    } catch (err) {
      setPositions([]);
      setTemperatures([]);
      setError(err.response?.data?.error || 'Prediction failed.');
    }
    setLoading(false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handlePredict();
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Thermal twin dashboard</h1>
        <form onSubmit={handleSubmit} style={{ marginBottom: 24, opacity: loading ? 0.6 : 1, pointerEvents: loading ? 'none' : 'auto' }}>
          {PARAM_FIELDS.map(f => (
            <label key={f.name}>
              {f.label}:
              <input
                type={f.type}
                name={f.name}
                value={params[f.name]}
                onChange={handleChange}
                step={f.step}
                disabled={loading}
              />
            </label>
          ))}
          <label>
            Model:
            <select value={modelType} onChange={e => setModelType(e.target.value)} disabled={loading}>
              <option value="physics">Physics model</option>
              <option value="surrogate">Surrogate model</option>
            </select>
          </label>
          <button type="submit" disabled={loading} style={{ marginLeft: 16 }}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>
        {modelType === 'surrogate' && meta && (
          <div style={{ color: '#ccc', marginBottom: 12, fontSize: 14, background: '#333', padding: 8, borderRadius: 6 }}>
            <b>Surrogate model info:</b><br />
            Trained: {meta.date}<br />
            Epochs: {meta.epochs}, Batch size: {meta.batch_size}, LR: {meta.lr}<br />
            PyTorch: {meta.pytorch_version}, Numpy: {meta.numpy_version}
          </div>
        )}
        {error && <div style={{ color: 'red', fontWeight: 600, marginBottom: 8 }}>{error}</div>}
        {positions.length > 0 && temperatures.length > 0 && (
          <>
            <div style={{ color: '#fff', marginBottom: 8 }}>
              Showing results from: <b>{modelType === 'physics' ? 'Physics model' : 'Surrogate model'}</b>
            </div>
            <Plot
              data={[
                {
                  x: positions,
                  y: temperatures,
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
          </>
        )}
        {loading && (
          <div style={{ color: '#fff', marginTop: 16 }}>Loading...</div>
        )}
      </header>
    </div>
  );
}

export default App;
