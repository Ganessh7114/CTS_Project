# Predictive Maintenance System

A production-grade predictive maintenance system for device fleet management that classifies device safety states and predicts Remaining Useful Life (RUL) with ≥90% accuracy while prioritizing safety-critical predictions.

## 🎯 Project Overview

This system provides real-time predictive maintenance capabilities for industrial devices, enabling early detection of degrading/unsafe devices to prevent failures and reduce downtime.

### Key Features

- **State Classification**: Classifies devices into three safety states (Critical/Warning/Safe)
- **RUL Prediction**: Predicts remaining useful life in hours and risk buckets
- **Temporal Data Handling**: Proper time-series validation without data leakage
- **Safety-First Design**: Optimized to minimize false negatives on critical states
- **Production-Ready API**: FastAPI server with monitoring and alerting
- **Comprehensive Evaluation**: Detailed metrics, plots, and safety analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Producer │───▶│  Feature Service │───▶│  Model Server   │───▶│   Monitoring    │
│  (Simulator/    │    │  (Rolling        │    │  (FastAPI       │    │  (Drift/        │
│   IoT Devices)  │    │   Features)      │    │   Endpoint)     │    │   Performance)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │    │  Feature Store   │    │  Model Registry │    │  Metrics Store  │
│  (JSON Stream)  │    │  (Per-UDI State) │    │  (Versioned)    │    │  (Prometheus)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Data Schema

### Input Data Format
```json
{
  "udi": "DEV_000001",
  "deviceType": "pump",
  "deviceName": "pump_DEV_000001",
  "timestamp": "2024-01-01T10:00:00Z",
  "runtimeHours": 100.5,
  "temperatureC": 45.2,
  "pressureKPa": 150.0,
  "vibrationMM_S": 0.3,
  "performanceScore": 0.92
}
```

### Output Predictions
```json
{
  "udi": "DEV_000001",
  "timestamp": "2024-01-01T10:00:00Z",
  "state_prediction": {
    "predicted_state": "II",
    "state_probabilities": {"I": 0.1, "II": 0.7, "III": 0.2},
    "confidence": 0.7
  },
  "rul_prediction": {
    "predicted_rul_hours": 85.5,
    "rul_bucket": "Warning",
    "bucket_probabilities": {"Urgent": 0.1, "Warning": 0.8, "Safe": 0.1}
  },
  "risk_score": 0.65,
  "recommendations": [
    "Schedule maintenance within 24-48 hours",
    "Monitor device closely"
  ]
}
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python main_training_pipeline.py
```

This will:
- Generate synthetic device data
- Train state classification and RUL regression models
- Tune decision thresholds for safety
- Generate comprehensive evaluation reports
- Set up monitoring system

### 3. Start API Server

```bash
python src/api_server.py
```

The API will be available at `http://localhost:8080`

### 4. Test Predictions

```bash
# Test state prediction
curl -X POST "http://localhost:8080/predict/state" \
  -H "Content-Type: application/json" \
  -d '{
    "udi": "DEV_000001",
    "deviceType": "pump",
    "deviceName": "test_pump",
    "timestamp": "2024-01-01T10:00:00Z",
    "runtimeHours": 100.5,
    "temperatureC": 45.2,
    "pressureKPa": 150.0,
    "vibrationMM_S": 0.3,
    "performanceScore": 0.92
  }'

# Test combined prediction
curl -X POST "http://localhost:8080/predict/combined" \
  -H "Content-Type: application/json" \
  -d '{
    "udi": "DEV_000001",
    "deviceType": "pump",
    "deviceName": "test_pump",
    "timestamp": "2024-01-01T10:00:00Z",
    "runtimeHours": 100.5,
    "temperatureC": 45.2,
    "pressureKPa": 150.0,
    "vibrationMM_S": 0.3,
    "performanceScore": 0.92
  }'
```

## 📁 Project Structure

```
predictive-maintenance/
├── src/                          # Source code
│   ├── data_simulation.py        # Synthetic data generation
│   ├── data_splitting.py         # Temporal data splitting
│   ├── feature_engineering.py    # Feature engineering pipeline
│   ├── models.py                 # Model training and prediction
│   ├── threshold_tuning.py       # Safety-optimized threshold tuning
│   ├── evaluation.py             # Comprehensive evaluation
│   ├── api_server.py             # FastAPI server
│   └── monitoring.py             # Drift detection and alerting
├── main_training_pipeline.py     # Complete training pipeline
├── requirements.txt              # Python dependencies
├── data/                         # Generated data
├── models/                       # Trained models
├── plots/                        # Evaluation plots
├── reports/                      # Evaluation reports
└── README.md                     # This file
```

## 🔧 Configuration

### Pipeline Configuration

The system can be configured through the `config` dictionary in `main_training_pipeline.py`:

```python
config = {
    'data': {
        'num_devices': 50,                    # Number of devices to simulate
        'output_path': 'data/synthetic_fleet_data.csv'
    },
    'splitting': {
        'test_size': 0.2,                    # Test set size (temporal split)
        'n_splits': 5                        # Cross-validation folds
    },
    'features': {
        'rolling_windows': [5, 30, 120],     # Rolling feature windows
        'delta_periods': [1, 5, 10]          # Delta feature periods
    },
    'thresholds': {
        'accuracy_target': 0.90,             # Minimum accuracy requirement
        'max_false_negative_rate': 0.05      # Max FN rate for safety classes
    }
}
```

### API Configuration

The API server can be configured through environment variables:

```bash
export MODEL_PATH="models/predictive_maintenance_models.pkl"
export REDIS_URL="redis://localhost:6379"
export API_HOST="0.0.0.0"
export API_PORT="8080"
```

## 📈 Model Performance

### State Classification
- **Overall Accuracy**: ≥90%
- **Class I (Critical) Recall**: ≥95% (safety-critical)
- **Class II (Warning) Recall**: ≥90%
- **False Negative Rate**: <5% for safety classes

### RUL Regression
- **MAE**: <20 hours
- **RMSE**: <30 hours
- **R²**: >0.8

### Safety Metrics
- **Overall Safety Score**: Weighted recall for critical classes
- **Risk Assessment**: Combined state and RUL risk scoring
- **Alert Generation**: Automated recommendations based on predictions

## 🔍 Evaluation & Monitoring

### Comprehensive Evaluation
- Per-class precision, recall, F1 scores
- ROC curves and precision-recall curves
- Confusion matrices
- Calibration plots
- Safety-specific metrics

### Monitoring & Alerting
- **Data Drift Detection**: Feature distribution monitoring
- **Performance Degradation**: Model performance tracking
- **Alert Management**: Configurable alert handlers
- **Metrics Collection**: Prometheus integration

### Generated Reports
- `reports/summary_report.txt`: Human-readable summary
- `reports/evaluation_results.json`: Detailed metrics
- `reports/pipeline_results.json`: Complete pipeline results
- `plots/`: Visualization plots

## 🛡️ Safety & Reliability

### Safety-First Design
- **Threshold Optimization**: Grid search optimizing for safety
- **Class Weights**: Balanced training for imbalanced classes
- **False Negative Minimization**: Priority on catching critical states
- **Confidence Scoring**: Uncertainty quantification

### Production Reliability
- **Temporal Validation**: No data leakage in time-series
- **Online/Offline Parity**: Consistent feature engineering
- **Model Versioning**: Reproducible training and deployment
- **Error Handling**: Robust API with proper error responses

## 🔌 API Endpoints

### Health Check
```http
GET /health
```

### State Prediction
```http
POST /predict/state
Content-Type: application/json

{
  "udi": "string",
  "deviceType": "string",
  "deviceName": "string",
  "timestamp": "ISO-8601",
  "runtimeHours": float,
  "temperatureC": float,
  "pressureKPa": float,
  "vibrationMM_S": float,
  "performanceScore": float
}
```

### RUL Prediction
```http
POST /predict/rul
```

### Combined Prediction
```http
POST /predict/combined
```

### Metrics
```http
GET /metrics
```

## 🚨 Monitoring & Alerts

### Drift Detection
- Feature distribution monitoring
- Statistical drift detection
- Configurable thresholds

### Performance Monitoring
- Model accuracy tracking
- Prediction latency monitoring
- Error rate tracking

### Alert Types
- **INFO**: Normal operational alerts
- **WARNING**: Performance degradation detected
- **CRITICAL**: Safety-critical issues detected

### Alert Handlers
- Email notifications
- Slack webhooks
- Custom handlers

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### API Tests
```bash
python -m pytest tests/api/
```

## 📊 Performance Benchmarks

### Training Performance
- **Data Generation**: ~30 seconds for 50 devices
- **Feature Engineering**: ~60 seconds for 50K records
- **Model Training**: ~120 seconds for complete suite
- **Threshold Tuning**: ~300 seconds (grid search)

### Inference Performance
- **Single Prediction**: <100ms
- **Batch Processing**: <50ms per prediction
- **API Latency**: <200ms end-to-end

## 🔄 Deployment

### Docker Deployment
```bash
# Build image
docker build -t predictive-maintenance .

# Run container
docker run -p 8080:8080 predictive-maintenance
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: predictive-maintenance
spec:
  replicas: 3
  selector:
    matchLabels:
      app: predictive-maintenance
  template:
    metadata:
      labels:
        app: predictive-maintenance
    spec:
      containers:
      - name: api
        image: predictive-maintenance:latest
        ports:
        - containerPort: 8080
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the documentation in `docs/`
- Review the example notebooks in `notebooks/`

## 🔮 Roadmap

- [ ] Real-time streaming integration
- [ ] Advanced drift detection algorithms
- [ ] A/B testing framework
- [ ] Model interpretability tools
- [ ] Multi-device type support
- [ ] Edge deployment capabilities
- [ ] Advanced alerting rules
- [ ] Performance optimization

---

**Note**: This is a production-ready system designed for industrial predictive maintenance. The safety-critical nature of the application requires thorough testing and validation before deployment in production environments.