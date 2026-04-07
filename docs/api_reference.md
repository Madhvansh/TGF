# TGF Dashboard API Reference

The TGF dashboard runs on FastAPI at `http://localhost:8000` by default.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard overview (embedded HTML + Chart.js) |
| `/api/status` | GET | System status and uptime |
| `/api/readings/latest` | GET | Latest sensor readings |
| `/api/readings/history` | GET | Historical readings (query by time range) |
| `/api/anomalies` | GET | Anomaly event history |
| `/api/dosing/current` | GET | Current dosing state and chemical residuals |
| `/api/dosing/history` | GET | Historical dosing decisions |
| `/api/chemicals` | GET | Chemical inventory and tracking |
| `/api/alerts` | GET | Active alerts |
| `/api/alerts/history` | GET | Alert history |
| `/api/metrics` | GET | System performance metrics |
| `/api/risk` | GET | Current risk assessment (scaling, corrosion, biofouling) |
| `/api/forecast` | GET | Sensor forecasts (Chronos-2 predictions) |
| `/api/cascade` | GET | Cascade detector state |
| `/api/config` | GET | Tower configuration |
| `/api/health` | GET | Health check |
| `/api/report` | GET | Full system report |
| `/api/calibration` | POST | Submit lab calibration data |

## Running the Dashboard

```bash
# With dashboard enabled (default)
python -m tgf_dosing.main --data data/Parameters_5K.csv --cycles 500 --speed 0

# Custom port
python -m tgf_dosing.main --data data/Parameters_5K.csv --port 9000

# Without dashboard (headless mode)
python -m tgf_dosing.main --data data/Parameters_5K.csv --no-api
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TGF_API_PORT` | 8000 | Dashboard port |
| `TGF_DATA_PATH` | - | Path to sensor data CSV |
| `TGF_DB_PATH` | `tgf_data.db` | SQLite database path |
| `TGF_MOMENT_CHECKPOINT` | - | MOMENT model checkpoint path |
