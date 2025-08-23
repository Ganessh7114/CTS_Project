"""
Monitoring Module for Predictive Maintenance
Handles drift detection, performance tracking, and alerting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import time
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert data structure."""
    timestamp: datetime
    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: float
    threshold: float
    device_id: Optional[str] = None

class DriftDetector:
    """Detects data drift and concept drift in predictive maintenance."""
    
    def __init__(self, reference_data: pd.DataFrame, 
                 drift_threshold: float = 0.1,
                 window_size: int = 1000):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference dataset for drift detection
            drift_threshold: Threshold for drift detection
            window_size: Size of sliding window for drift detection
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.reference_stats = self._calculate_reference_stats()
        
    def _calculate_reference_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate reference statistics for drift detection."""
        numeric_columns = self.reference_data.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_columns:
            stats[col] = {
                'mean': float(self.reference_data[col].mean()),
                'std': float(self.reference_data[col].std()),
                'min': float(self.reference_data[col].min()),
                'max': float(self.reference_data[col].max()),
                'q25': float(self.reference_data[col].quantile(0.25)),
                'q75': float(self.reference_data[col].quantile(0.75))
            }
        
        return stats
    
    def detect_feature_drift(self, current_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Detect feature drift in current data.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {}
        
        for feature, ref_stats in self.reference_stats.items():
            if feature not in current_data.columns:
                continue
            
            current_stats = {
                'mean': float(current_data[feature].mean()),
                'std': float(current_data[feature].std()),
                'min': float(current_data[feature].min()),
                'max': float(current_data[feature].max()),
                'q25': float(current_data[feature].quantile(0.25)),
                'q75': float(current_data[feature].quantile(0.75))
            }
            
            # Calculate drift metrics
            mean_drift = abs(current_stats['mean'] - ref_stats['mean']) / (ref_stats['std'] + 1e-8)
            std_drift = abs(current_stats['std'] - ref_stats['std']) / (ref_stats['std'] + 1e-8)
            
            # Distribution drift using KS test approximation
            distribution_drift = self._calculate_distribution_drift(
                current_data[feature], ref_stats
            )
            
            # Determine if drift is detected
            drift_detected = (
                mean_drift > self.drift_threshold or
                std_drift > self.drift_threshold or
                distribution_drift > self.drift_threshold
            )
            
            drift_results[feature] = {
                'drift_detected': drift_detected,
                'mean_drift': mean_drift,
                'std_drift': std_drift,
                'distribution_drift': distribution_drift,
                'reference_stats': ref_stats,
                'current_stats': current_stats,
                'severity': self._calculate_drift_severity(mean_drift, std_drift, distribution_drift)
            }
        
        return drift_results
    
    def _calculate_distribution_drift(self, current_data: pd.Series, 
                                    ref_stats: Dict[str, float]) -> float:
        """Calculate distribution drift using simplified approach."""
        # Simplified distribution drift calculation
        current_mean = current_data.mean()
        current_std = current_data.std()
        
        # Normalized distance from reference distribution
        mean_diff = abs(current_mean - ref_stats['mean']) / (ref_stats['std'] + 1e-8)
        std_diff = abs(current_std - ref_stats['std']) / (ref_stats['std'] + 1e-8)
        
        return (mean_diff + std_diff) / 2
    
    def _calculate_drift_severity(self, mean_drift: float, std_drift: float, 
                                distribution_drift: float) -> AlertSeverity:
        """Calculate drift severity level."""
        max_drift = max(mean_drift, std_drift, distribution_drift)
        
        if max_drift > 2.0 * self.drift_threshold:
            return AlertSeverity.CRITICAL
        elif max_drift > 1.5 * self.drift_threshold:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

class PerformanceMonitor:
    """Monitors model performance and detects degradation."""
    
    def __init__(self, baseline_metrics: Dict[str, float],
                 degradation_threshold: float = 0.1):
        """
        Initialize performance monitor.
        
        Args:
            baseline_metrics: Baseline performance metrics
            degradation_threshold: Threshold for performance degradation
        """
        self.baseline_metrics = baseline_metrics
        self.degradation_threshold = degradation_threshold
        self.performance_history = []
        
    def update_performance(self, metrics: Dict[str, float], timestamp: datetime):
        """Update performance metrics."""
        self.performance_history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def detect_performance_degradation(self) -> Dict[str, Dict[str, Any]]:
        """Detect performance degradation."""
        if len(self.performance_history) < 10:
            return {}
        
        # Get recent performance (last 100 entries)
        recent_performance = self.performance_history[-100:]
        
        degradation_results = {}
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            recent_values = [
                entry['metrics'].get(metric_name, baseline_value) 
                for entry in recent_performance
            ]
            
            if not recent_values:
                continue
            
            current_value = np.mean(recent_values[-10:])  # Last 10 values
            degradation = (baseline_value - current_value) / baseline_value
            
            degradation_detected = degradation > self.degradation_threshold
            
            degradation_results[metric_name] = {
                'degradation_detected': degradation_detected,
                'degradation_percentage': degradation * 100,
                'baseline_value': baseline_value,
                'current_value': current_value,
                'severity': self._calculate_degradation_severity(degradation)
            }
        
        return degradation_results
    
    def _calculate_degradation_severity(self, degradation: float) -> AlertSeverity:
        """Calculate degradation severity."""
        if degradation > 0.3:
            return AlertSeverity.CRITICAL
        elif degradation > 0.15:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts = []
        self.alert_handlers = []
        
    def add_alert(self, alert: Alert):
        """Add a new alert."""
        self.alerts.append(alert)
        logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def add_alert_handler(self, handler):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get recent alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level."""
        return [alert for alert in self.alerts if alert.severity == severity]
    
    def clear_old_alerts(self, days: int = 7):
        """Clear old alerts."""
        cutoff_time = datetime.now() - timedelta(days=days)
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]

class PredictiveMaintenanceMonitor:
    """Main monitoring class for predictive maintenance system."""
    
    def __init__(self, reference_data: pd.DataFrame,
                 baseline_metrics: Dict[str, float],
                 drift_threshold: float = 0.1,
                 degradation_threshold: float = 0.1):
        """
        Initialize monitoring system.
        
        Args:
            reference_data: Reference data for drift detection
            baseline_metrics: Baseline performance metrics
            drift_threshold: Threshold for drift detection
            degradation_threshold: Threshold for performance degradation
        """
        self.drift_detector = DriftDetector(reference_data, drift_threshold)
        self.performance_monitor = PerformanceMonitor(baseline_metrics, degradation_threshold)
        self.alert_manager = AlertManager()
        
        # Monitoring state
        self.last_drift_check = None
        self.last_performance_check = None
        self.monitoring_enabled = True
        
    def check_data_drift(self, current_data: pd.DataFrame, 
                        device_id: Optional[str] = None) -> Dict[str, Any]:
        """Check for data drift and generate alerts."""
        if not self.monitoring_enabled:
            return {}
        
        drift_results = self.drift_detector.detect_feature_drift(current_data)
        
        # Generate alerts for detected drift
        for feature, result in drift_results.items():
            if result['drift_detected']:
                alert = Alert(
                    timestamp=datetime.now(),
                    severity=result['severity'],
                    message=f"Data drift detected in feature '{feature}'",
                    metric_name=f"drift_{feature}",
                    current_value=result['mean_drift'],
                    threshold=self.drift_detector.drift_threshold,
                    device_id=device_id
                )
                self.alert_manager.add_alert(alert)
        
        self.last_drift_check = datetime.now()
        return drift_results
    
    def check_performance_degradation(self) -> Dict[str, Any]:
        """Check for performance degradation and generate alerts."""
        if not self.monitoring_enabled:
            return {}
        
        degradation_results = self.performance_monitor.detect_performance_degradation()
        
        # Generate alerts for detected degradation
        for metric_name, result in degradation_results.items():
            if result['degradation_detected']:
                alert = Alert(
                    timestamp=datetime.now(),
                    severity=result['severity'],
                    message=f"Performance degradation detected in '{metric_name}'",
                    metric_name=f"degradation_{metric_name}",
                    current_value=result['current_value'],
                    threshold=result['baseline_value'] * (1 - self.performance_monitor.degradation_threshold)
                )
                self.alert_manager.add_alert(alert)
        
        self.last_performance_check = datetime.now()
        return degradation_results
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics."""
        self.performance_monitor.update_performance(metrics, datetime.now())
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        recent_alerts = self.alert_manager.get_recent_alerts(24)
        
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'last_drift_check': self.last_drift_check.isoformat() if self.last_drift_check else None,
            'last_performance_check': self.last_performance_check.isoformat() if self.last_performance_check else None,
            'recent_alerts_count': len(recent_alerts),
            'critical_alerts_count': len(self.alert_manager.get_alerts_by_severity(AlertSeverity.CRITICAL)),
            'warning_alerts_count': len(self.alert_manager.get_alerts_by_severity(AlertSeverity.WARNING)),
            'info_alerts_count': len(self.alert_manager.get_alerts_by_severity(AlertSeverity.INFO))
        }
    
    def add_alert_handler(self, handler):
        """Add custom alert handler."""
        self.alert_manager.add_alert_handler(handler)
    
    def enable_monitoring(self):
        """Enable monitoring."""
        self.monitoring_enabled = True
        logger.info("Monitoring enabled")
    
    def disable_monitoring(self):
        """Disable monitoring."""
        self.monitoring_enabled = False
        logger.info("Monitoring disabled")

def create_monitoring_system(reference_data: pd.DataFrame,
                           baseline_metrics: Dict[str, float]) -> PredictiveMaintenanceMonitor:
    """
    Create monitoring system for predictive maintenance.
    
    Args:
        reference_data: Reference data for drift detection
        baseline_metrics: Baseline performance metrics
        
    Returns:
        Configured monitoring system
    """
    return PredictiveMaintenanceMonitor(
        reference_data=reference_data,
        baseline_metrics=baseline_metrics
    )

def email_alert_handler(alert: Alert):
    """Example email alert handler."""
    # This would integrate with your email service
    logger.info(f"EMAIL ALERT: {alert.severity.value} - {alert.message}")

def slack_alert_handler(alert: Alert):
    """Example Slack alert handler."""
    # This would integrate with Slack webhook
    logger.info(f"SLACK ALERT: {alert.severity.value} - {alert.message}")

if __name__ == "__main__":
    # Example usage
    from data_simulation import create_synthetic_dataset
    from data_splitting import create_temporal_splits
    from feature_engineering import create_feature_pipeline
    from models import train_complete_model_suite
    from evaluation import create_evaluation_report
    
    # Generate and prepare data
    df = create_synthetic_dataset()
    splits = create_temporal_splits(df)
    
    # Create features
    pipeline = create_feature_pipeline()
    pipeline.fit(splits['train_df'])
    
    X_train = pipeline.transform(splits['train_df'])
    X_test = pipeline.transform(splits['test_df'])
    
    # Prepare targets
    y_train_class = splits['train_df']['stateClass']
    y_train_rul = splits['train_df']['lifeRemaining']
    y_test_class = splits['test_df']['stateClass']
    y_test_rul = splits['test_df']['lifeRemaining']
    
    # Train models
    model_suite = train_complete_model_suite(X_train, y_train_class, y_train_rul)
    
    # Get predictions
    y_pred_class, y_prob_class = model_suite.predict_state(X_test, 'xgb')
    y_pred_rul = model_suite.predict_rul(X_test, 'xgb_reg')
    
    # Create evaluation report
    evaluator = create_evaluation_report(
        y_test_class, y_pred_class, y_prob_class,
        y_test_rul, y_pred_rul,
        save_plots=False
    )
    
    # Create monitoring system
    baseline_metrics = {
        'state_accuracy': evaluator.evaluation_results['state_classification']['accuracy'],
        'rul_mae': evaluator.evaluation_results['rul_regression']['mae'],
        'rul_r2': evaluator.evaluation_results['rul_regression']['r2']
    }
    
    monitor = create_monitoring_system(
        reference_data=X_train,
        baseline_metrics=baseline_metrics
    )
    
    # Add alert handlers
    monitor.add_alert_handler(email_alert_handler)
    monitor.add_alert_handler(slack_alert_handler)
    
    # Simulate monitoring
    print("Monitoring system initialized")
    print(f"Baseline metrics: {baseline_metrics}")
    
    # Check for drift in test data
    drift_results = monitor.check_data_drift(X_test)
    print(f"Drift detection results: {len([r for r in drift_results.values() if r['drift_detected']])} features with drift")
    
    # Get monitoring summary
    summary = monitor.get_monitoring_summary()
    print(f"Monitoring summary: {summary}")