"""
FastAPI Server for Predictive Maintenance
Serves state classification and RUL prediction models with proper schemas.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import joblib
import redis
import json
import time
from datetime import datetime
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made', ['model_type', 'endpoint'])
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Time spent on predictions', ['model_type'])
ERROR_COUNTER = Counter('prediction_errors_total', 'Total prediction errors', ['model_type', 'error_type'])

class DeviceReading(BaseModel):
    """Schema for device reading input."""
    udi: str = Field(..., description="Unique Device Identifier")
    deviceType: str = Field(..., description="Type of device (pump, compressor, etc.)")
    deviceName: str = Field(..., description="Human-readable device name")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    runtimeHours: float = Field(..., description="Device runtime in hours")
    temperatureC: float = Field(..., description="Temperature in Celsius")
    pressureKPa: float = Field(..., description="Pressure in kPa")
    vibrationMM_S: float = Field(..., description="Vibration in mm/s")
    performanceScore: float = Field(..., description="Performance score (0-1)")

class StatePrediction(BaseModel):
    """Schema for state classification prediction."""
    udi: str
    timestamp: str
    predicted_state: str = Field(..., description="Predicted state: I (Critical), II (Warning), III (Safe)")
    state_probabilities: Dict[str, float] = Field(..., description="Probability for each state")
    confidence: float = Field(..., description="Confidence in prediction (0-1)")
    model_version: str = Field(..., description="Model version used")

class RULPrediction(BaseModel):
    """Schema for RUL prediction."""
    udi: str
    timestamp: str
    predicted_rul_hours: float = Field(..., description="Predicted remaining useful life in hours")
    rul_bucket: str = Field(..., description="RUL bucket: Urgent (<30h), Warning (30-100h), Safe (≥100h)")
    bucket_probabilities: Dict[str, float] = Field(..., description="Probability for each bucket")
    confidence_interval: Dict[str, float] = Field(..., description="95% confidence interval")
    model_version: str = Field(..., description="Model version used")

class CombinedPrediction(BaseModel):
    """Schema for combined state and RUL prediction."""
    udi: str
    timestamp: str
    state_prediction: StatePrediction
    rul_prediction: RULPrediction
    risk_score: float = Field(..., description="Overall risk score (0-1)")
    recommendations: List[str] = Field(..., description="Maintenance recommendations")

class HealthCheck(BaseModel):
    """Schema for health check response."""
    status: str
    timestamp: str
    model_versions: Dict[str, str]
    uptime_seconds: float

class FeatureCache:
    """Cache for device features to maintain stateful rolling features."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize feature cache with Redis backend."""
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.redis_client = None
            self.memory_cache = {}
    
    def get_device_history(self, udi: str) -> List[Dict[str, Any]]:
        """Get device history for feature engineering."""
        if self.redis_client:
            try:
                history = self.redis_client.get(f"device_history:{udi}")
                return json.loads(history) if history else []
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                return []
        else:
            return self.memory_cache.get(udi, [])
    
    def update_device_history(self, udi: str, reading: Dict[str, Any]):
        """Update device history with new reading."""
        history = self.get_device_history(udi)
        history.append(reading)
        
        # Keep only last 200 readings per device
        if len(history) > 200:
            history = history[-200:]
        
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"device_history:{udi}", 
                    3600,  # 1 hour TTL
                    json.dumps(history)
                )
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        else:
            self.memory_cache[udi] = history

class PredictiveMaintenanceAPI:
    """Main API class for predictive maintenance."""
    
    def __init__(self, model_path: str = "models/predictive_maintenance_models.pkl"):
        """Initialize API with loaded models."""
        self.model_path = model_path
        self.feature_cache = FeatureCache()
        self.start_time = time.time()
        
        # Load models
        try:
            self.load_models()
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def load_models(self):
        """Load trained models and preprocessing components."""
        model_data = joblib.load(self.model_path)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.class_weights = model_data['class_weights']
        
        # Load feature pipeline
        from feature_engineering import create_feature_pipeline
        self.feature_pipeline = create_feature_pipeline()
        
        logger.info(f"Loaded models: {list(self.models.keys())}")
    
    def engineer_features(self, device_history: List[Dict[str, Any]]) -> pd.DataFrame:
        """Engineer features from device history."""
        if not device_history:
            raise ValueError("No device history available for feature engineering")
        
        # Convert to DataFrame
        df = pd.DataFrame(device_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Apply feature pipeline
        features_df = self.feature_pipeline.transform(df)
        
        # Get latest features only
        latest_features = features_df.iloc[-1:].copy()
        
        return latest_features
    
    def predict_state(self, udi: str, reading: DeviceReading) -> StatePrediction:
        """Predict device state."""
        start_time = time.time()
        
        try:
            # Get device history
            device_history = self.feature_cache.get_device_history(udi)
            
            # Add current reading
            reading_dict = reading.dict()
            device_history.append(reading_dict)
            
            # Engineer features
            features_df = self.engineer_features(device_history)
            
            # Make prediction
            predictions, probabilities = self.models['xgb'].predict(features_df)
            predicted_state = predictions[0]
            state_probs = probabilities[0]
            
            # Calculate confidence
            confidence = np.max(state_probs)
            
            # Update cache
            self.feature_cache.update_device_history(udi, reading_dict)
            
            # Record metrics
            duration = time.time() - start_time
            PREDICTION_DURATION.labels(model_type='state').observe(duration)
            PREDICTION_COUNTER.labels(model_type='state', endpoint='/predict_state').inc()
            
            return StatePrediction(
                udi=udi,
                timestamp=reading.timestamp,
                predicted_state=predicted_state,
                state_probabilities={
                    'I': float(state_probs[0]),
                    'II': float(state_probs[1]),
                    'III': float(state_probs[2])
                },
                confidence=float(confidence),
                model_version="v1.0"
            )
            
        except Exception as e:
            ERROR_COUNTER.labels(model_type='state', error_type=type(e).__name__).inc()
            logger.error(f"State prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_rul(self, udi: str, reading: DeviceReading) -> RULPrediction:
        """Predict remaining useful life."""
        start_time = time.time()
        
        try:
            # Get device history
            device_history = self.feature_cache.get_device_history(udi)
            
            # Add current reading
            reading_dict = reading.dict()
            device_history.append(reading_dict)
            
            # Engineer features
            features_df = self.engineer_features(device_history)
            
            # Make RUL prediction
            rul_prediction = self.models['xgb_reg'].predict(features_df)[0]
            
            # Make bucket prediction
            bucket_prediction, bucket_probs = self.models['xgb_bucket'].predict(features_df)
            bucket_prediction = bucket_prediction[0]
            bucket_probs = bucket_probs[0]
            
            # Map bucket to label
            bucket_labels = ['Urgent', 'Warning', 'Safe']
            bucket_label = bucket_labels[bucket_prediction]
            
            # Calculate confidence interval (simplified)
            confidence_interval = {
                'lower': max(0, rul_prediction - 20),
                'upper': rul_prediction + 20
            }
            
            # Update cache
            self.feature_cache.update_device_history(udi, reading_dict)
            
            # Record metrics
            duration = time.time() - start_time
            PREDICTION_DURATION.labels(model_type='rul').observe(duration)
            PREDICTION_COUNTER.labels(model_type='rul', endpoint='/predict_rul').inc()
            
            return RULPrediction(
                udi=udi,
                timestamp=reading.timestamp,
                predicted_rul_hours=float(rul_prediction),
                rul_bucket=bucket_label,
                bucket_probabilities={
                    'Urgent': float(bucket_probs[0]),
                    'Warning': float(bucket_probs[1]),
                    'Safe': float(bucket_probs[2])
                },
                confidence_interval=confidence_interval,
                model_version="v1.0"
            )
            
        except Exception as e:
            ERROR_COUNTER.labels(model_type='rul', error_type=type(e).__name__).inc()
            logger.error(f"RUL prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_combined(self, udi: str, reading: DeviceReading) -> CombinedPrediction:
        """Make combined state and RUL prediction."""
        start_time = time.time()
        
        try:
            # Get both predictions
            state_pred = self.predict_state(udi, reading)
            rul_pred = self.predict_rul(udi, reading)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(state_pred, rul_pred)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(state_pred, rul_pred)
            
            # Record metrics
            duration = time.time() - start_time
            PREDICTION_DURATION.labels(model_type='combined').observe(duration)
            PREDICTION_COUNTER.labels(model_type='combined', endpoint='/predict_combined').inc()
            
            return CombinedPrediction(
                udi=udi,
                timestamp=reading.timestamp,
                state_prediction=state_pred,
                rul_prediction=rul_pred,
                risk_score=risk_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            ERROR_COUNTER.labels(model_type='combined', error_type=type(e).__name__).inc()
            logger.error(f"Combined prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def _calculate_risk_score(self, state_pred: StatePrediction, rul_pred: RULPrediction) -> float:
        """Calculate overall risk score."""
        # State risk weights
        state_risk_weights = {'I': 1.0, 'II': 0.6, 'III': 0.1}
        state_risk = state_risk_weights.get(state_pred.predicted_state, 0.5)
        
        # RUL risk (lower RUL = higher risk)
        rul_risk = max(0, 1 - (rul_pred.predicted_rul_hours / 1000))  # Normalize to 1000 hours
        
        # Combined risk score
        risk_score = 0.7 * state_risk + 0.3 * rul_risk
        return min(1.0, risk_score)
    
    def _generate_recommendations(self, state_pred: StatePrediction, rul_pred: RULPrediction) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []
        
        # State-based recommendations
        if state_pred.predicted_state == 'I':
            recommendations.append("CRITICAL: Immediate maintenance required")
            recommendations.append("Shutdown device if safe to do so")
        elif state_pred.predicted_state == 'II':
            recommendations.append("Schedule maintenance within 24-48 hours")
            recommendations.append("Monitor device closely")
        else:
            recommendations.append("Continue normal operation")
        
        # RUL-based recommendations
        if rul_pred.rul_bucket == 'Urgent':
            recommendations.append("Plan replacement within 30 hours")
        elif rul_pred.rul_bucket == 'Warning':
            recommendations.append("Plan maintenance within 100 hours")
        
        return recommendations
    
    def health_check(self) -> HealthCheck:
        """Health check endpoint."""
        uptime = time.time() - self.start_time
        
        return HealthCheck(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            model_versions={
                "state_classifier": "v1.0",
                "rul_regressor": "v1.0",
                "rul_bucket_classifier": "v1.0"
            },
            uptime_seconds=uptime
        )

# Initialize API
api = PredictiveMaintenanceAPI()

# Create FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="API for device state classification and RUL prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Startup event - start Prometheus metrics server."""
    start_http_server(8000)

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return api.health_check()

@app.post("/predict/state", response_model=StatePrediction)
async def predict_state(reading: DeviceReading):
    """Predict device state."""
    return api.predict_state(reading.udi, reading)

@app.post("/predict/rul", response_model=RULPrediction)
async def predict_rul(reading: DeviceReading):
    """Predict remaining useful life."""
    return api.predict_rul(reading.udi, reading)

@app.post("/predict/combined", response_model=CombinedPrediction)
async def predict_combined(reading: DeviceReading):
    """Make combined state and RUL prediction."""
    return api.predict_combined(reading.udi, reading)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict/state": "Predict device state",
            "/predict/rul": "Predict remaining useful life",
            "/predict/combined": "Combined prediction",
            "/metrics": "Prometheus metrics"
        }
    }

# Example usage
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)