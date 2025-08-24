"""
Model Training for Predictive Maintenance
Implements three classifiers, RUL regression, and RUL bucket classifier with proper handling.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from typing import Dict, Any, Tuple, List, Optional
import joblib
import warnings
import os

class PredictiveMaintenanceModels:
    """
    Complete model suite for predictive maintenance.
    Handles state classification, RUL regression, and RUL bucket classification.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model suite.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = None
        self.class_weights = None
        
    def prepare_data(self, X: pd.DataFrame, y_class: pd.Series = None, 
                    y_rul: pd.Series = None, fit: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for modeling.
        
        Args:
            X: Feature matrix
            y_class: State classification labels
            y_rul: RUL regression targets
            fit: Whether to fit scalers and encoders
            
        Returns:
            X_scaled, y_class_encoded, y_rul: Prepared data
        """
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
        
        # Handle NaN values by filling with 0
        X_clean = X.fillna(0)
        
        # Scale features
        if fit:
            self.scalers['features'] = StandardScaler()
            X_scaled = self.scalers['features'].fit_transform(X_clean)
        else:
            X_scaled = self.scalers['features'].transform(X_clean)
        
        # Encode classification labels
        if y_class is not None:
            if fit:
                self.label_encoders['class'] = LabelEncoder()
                y_class_encoded = self.label_encoders['class'].fit_transform(y_class)
            else:
                y_class_encoded = self.label_encoders['class'].transform(y_class)
        else:
            y_class_encoded = None
        
        # RUL doesn't need encoding
        y_rul_processed = y_rul.values if y_rul is not None else None
        
        return X_scaled, y_class_encoded, y_rul_processed
    
    def calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights to handle imbalance.
        
        Args:
            y: Classification labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        return dict(zip(classes, weights))
    
    def train_state_classifiers(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train state classification models.
        
        Args:
            X: Feature matrix
            y: State labels (I, II, III)
            
        Returns:
            Dictionary with trained models and metadata
        """
        print("Training state classification models...")
        
        # Prepare data
        X_scaled, y_encoded, _ = self.prepare_data(X, y_class=y, fit=True)
        
        # Calculate class weights
        self.class_weights = self.calculate_class_weights(y_encoded)
        print(f"Class weights: {self.class_weights}")
        
        # Train Logistic Regression (fast baseline)
        print("Training Logistic Regression...")
        lr = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=500  # Reduced for speed
        )
        lr.fit(X_scaled, y_encoded)
        
        # Calibrate LR (simplified)
        lr_calibrated = CalibratedClassifierCV(lr, cv=3, method='isotonic')  # Reduced CV
        lr_calibrated.fit(X_scaled, y_encoded)
        
        # Train Random Forest (simplified)
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=50,  # Reduced from 100
            max_depth=8,      # Reduced from 10
            min_samples_split=20,  # Increased for speed
            min_samples_leaf=10,   # Increased for speed
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_scaled, y_encoded)
        
        # Train XGBoost (simplified)
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,  # Reduced from 100
            max_depth=4,      # Reduced from 6
            learning_rate=0.2,  # Increased for faster convergence
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_encoded[y_encoded == 0]) / len(y_encoded[y_encoded == 1]),
            random_state=self.random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        xgb_model.fit(X_scaled, y_encoded)
        
        # Store models
        self.models['lr'] = lr_calibrated
        self.models['rf'] = rf
        self.models['xgb'] = xgb_model
        
        return {
            'models': {
                'lr': lr_calibrated,
                'rf': rf,
                'xgb': xgb_model
            },
            'class_weights': self.class_weights,
            'feature_names': self.feature_names
        }
    
    def train_rul_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train RUL regression models.
        
        Args:
            X: Feature matrix
            y: RUL values (continuous)
            
        Returns:
            Dictionary with trained models and metadata
        """
        print("Training RUL regression models...")
        
        # Prepare data
        X_scaled, _, y_rul = self.prepare_data(X, y_rul=y, fit=True)
        
        # Train Random Forest Regressor (simplified)
        print("Training Random Forest Regressor...")
        rf_reg = RandomForestRegressor(
            n_estimators=50,  # Reduced from 100
            max_depth=8,      # Reduced from 10
            min_samples_split=20,  # Increased for speed
            min_samples_leaf=10,   # Increased for speed
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_reg.fit(X_scaled, y_rul)
        
        # Train XGBoost Regressor (simplified)
        print("Training XGBoost Regressor...")
        xgb_reg = xgb.XGBRegressor(
            n_estimators=50,  # Reduced from 100
            max_depth=4,      # Reduced from 6
            learning_rate=0.2,  # Increased for faster convergence
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='rmse'
        )
        xgb_reg.fit(X_scaled, y_rul)
        
        # Store models
        self.models['rf_reg'] = rf_reg
        self.models['xgb_reg'] = xgb_reg
        
        return {
            'models': {
                'rf_reg': rf_reg,
                'xgb_reg': xgb_reg
            },
            'feature_names': self.feature_names
        }
    
    def create_rul_buckets(self, y_rul: pd.Series, 
                          thresholds: List[float] = [30, 100]) -> np.ndarray:
        """
        Create RUL bucket labels.
        
        Args:
            y_rul: RUL values
            thresholds: Bucket thresholds [urgent_threshold, warning_threshold]
            
        Returns:
            Bucket labels (0=Urgent, 1=Warning, 2=Safe)
        """
        buckets = np.zeros(len(y_rul), dtype=int)
        
        # Urgent: < 30 hours
        buckets[y_rul < thresholds[0]] = 0
        
        # Warning: 30-100 hours
        buckets[(y_rul >= thresholds[0]) & (y_rul < thresholds[1])] = 1
        
        # Safe: >= 100 hours
        buckets[y_rul >= thresholds[1]] = 2
        
        return buckets
    
    def train_rul_bucket_classifier(self, X: pd.DataFrame, y_rul: pd.Series,
                                  thresholds: List[float] = [30, 100]) -> Dict[str, Any]:
        """
        Train RUL bucket classification model.
        
        Args:
            X: Feature matrix
            y_rul: RUL values
            thresholds: Bucket thresholds
            
        Returns:
            Dictionary with trained model and metadata
        """
        print("Training RUL bucket classifier...")
        
        # Create bucket labels
        y_buckets = self.create_rul_buckets(y_rul, thresholds)
        
        # Prepare data
        X_scaled, y_encoded, _ = self.prepare_data(X, y_class=pd.Series(y_buckets), fit=True)
        
        # Calculate class weights for buckets
        bucket_weights = self.calculate_class_weights(y_encoded)
        print(f"Bucket class weights: {bucket_weights}")
        
        # Train XGBoost for bucket classification (simplified)
        xgb_bucket = xgb.XGBClassifier(
            n_estimators=50,  # Reduced from 100
            max_depth=4,      # Reduced from 6
            learning_rate=0.2,  # Increased for faster convergence
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        xgb_bucket.fit(X_scaled, y_encoded)
        
        # Store model
        self.models['xgb_bucket'] = xgb_bucket
        
        return {
            'model': xgb_bucket,
            'thresholds': thresholds,
            'class_weights': bucket_weights,
            'feature_names': self.feature_names
        }
    
    def predict_state(self, X: pd.DataFrame, model_name: str = 'xgb') -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict device states.
        
        Args:
            X: Feature matrix
            model_name: Model to use ('lr', 'rf', 'xgb')
            
        Returns:
            predictions, probabilities: Predicted classes and probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        # Prepare data
        X_scaled, _, _ = self.prepare_data(X, fit=False)
        
        # Predict
        model = self.models[model_name]
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Decode predictions
        if hasattr(self.label_encoders['class'], 'inverse_transform'):
            predictions = self.label_encoders['class'].inverse_transform(predictions)
        
        return predictions, probabilities
    
    def predict_rul(self, X: pd.DataFrame, model_name: str = 'xgb_reg') -> np.ndarray:
        """
        Predict RUL values.
        
        Args:
            X: Feature matrix
            model_name: Model to use ('rf_reg', 'xgb_reg')
            
        Returns:
            RUL predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        # Prepare data
        X_scaled, _, _ = self.prepare_data(X, fit=False)
        
        # Predict
        model = self.models[model_name]
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def predict_rul_buckets(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict RUL buckets.
        
        Args:
            X: Feature matrix
            
        Returns:
            predictions, probabilities: Predicted buckets and probabilities
        """
        if 'xgb_bucket' not in self.models:
            raise ValueError("RUL bucket model not trained")
        
        # Prepare data
        X_scaled, _, _ = self.prepare_data(X, fit=False)
        
        # Predict
        model = self.models['xgb_bucket']
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def evaluate_classification(self, X: pd.DataFrame, y_true: pd.Series, 
                              model_name: str = 'xgb') -> Dict[str, Any]:
        """
        Evaluate classification model.
        
        Args:
            X: Feature matrix
            y_true: True labels
            model_name: Model to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predict
        y_pred, y_prob = self.predict_state(X, model_name)
        
        # Encode true labels for comparison
        y_true_encoded = self.label_encoders['class'].transform(y_true)
        y_pred_encoded = self.label_encoders['class'].transform(y_pred)
        
        # Calculate metrics
        report = classification_report(
            y_true_encoded, y_pred_encoded, 
            target_names=self.label_encoders['class'].classes_,
            output_dict=True
        )
        
        # Calculate ROC AUC for each class
        roc_auc = {}
        for i, class_name in enumerate(self.label_encoders['class'].classes_):
            if len(np.unique(y_true_encoded)) > 1:
                roc_auc[class_name] = roc_auc_score(
                    (y_true_encoded == i).astype(int), 
                    y_prob[:, i]
                )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def evaluate_regression(self, X: pd.DataFrame, y_true: pd.Series,
                          model_name: str = 'xgb_reg') -> Dict[str, Any]:
        """
        Evaluate regression model.
        
        Args:
            X: Feature matrix
            y_true: True RUL values
            model_name: Model to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predict
        y_pred = self.predict_rul(X, model_name)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
    
    def save_models(self, filepath: str):
        """Save all models and metadata."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'class_weights': self.class_weights
        }
        
        try:
            joblib.dump(model_data, filepath)
            print(f"Models saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving models: {e}")
            raise
    
    def load_models(self, filepath: str):
        """Load models and metadata."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.class_weights = model_data['class_weights']
        print(f"Models loaded from {filepath}")

def train_complete_model_suite(X_train: pd.DataFrame, y_train_class: pd.Series, 
                              y_train_rul: pd.Series) -> PredictiveMaintenanceModels:
    """
    Train complete model suite for predictive maintenance.
    
    Args:
        X_train: Training features
        y_train_class: Training state labels
        y_train_rul: Training RUL values
        
    Returns:
        Trained model suite
    """
    # Initialize model suite
    model_suite = PredictiveMaintenanceModels()
    
    # Train state classifiers
    model_suite.train_state_classifiers(X_train, y_train_class)
    
    # Train RUL regression
    model_suite.train_rul_regression(X_train, y_train_rul)
    
    # Train RUL bucket classifier
    model_suite.train_rul_bucket_classifier(X_train, y_train_rul)
    
    return model_suite

if __name__ == "__main__":
    # Example usage
    from data_simulation import create_synthetic_dataset
    from data_splitting import create_temporal_splits
    from feature_engineering import create_feature_pipeline
    
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
    
    # Evaluate
    class_results = model_suite.evaluate_classification(X_test, y_test_class)
    reg_results = model_suite.evaluate_regression(X_test, y_test_rul)
    
    print("Classification Results:")
    print(f"Overall Accuracy: {class_results['classification_report']['accuracy']:.3f}")
    print(f"Class I Recall: {class_results['classification_report']['I']['recall']:.3f}")
    
    print("\nRegression Results:")
    print(f"MAE: {reg_results['mae']:.2f}")
    print(f"RMSE: {reg_results['rmse']:.2f}")
    print(f"R²: {reg_results['r2']:.3f}")