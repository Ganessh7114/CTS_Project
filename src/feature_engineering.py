"""
Feature Engineering for Predictive Maintenance
Builds rolling features without leakage and ensures online/offline parity.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any, Optional
import warnings

class RollingFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates rolling features for device maintenance prediction.
    Ensures no future information leakage by only using past data.
    """
    
    def __init__(self, windows: List[int] = [5, 30, 120], 
                 features: List[str] = None,
                 agg_functions: List[str] = None):
        """
        Initialize rolling feature transformer.
        
        Args:
            windows: List of window sizes (in time steps)
            features: Features to create rolling statistics for
            agg_functions: Aggregation functions to apply
        """
        self.windows = windows
        
        if features is None:
            self.features = ['temperatureC', 'pressureKPa', 'vibrationMM_S', 'performanceScore']
        else:
            self.features = features
            
        if agg_functions is None:
            self.agg_functions = ['mean', 'std', 'min', 'max', 'slope']
        else:
            self.agg_functions = agg_functions
        
        # Store feature names for transform
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no fitting needed for rolling features)."""
        # Generate feature names for consistency
        self.feature_names_ = self._get_feature_names()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by adding rolling features.
        
        Args:
            X: DataFrame with device data (must have 'udi' and 'timestamp' columns)
            
        Returns:
            DataFrame with original + rolling features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Ensure required columns exist
        required_cols = ['udi', 'timestamp'] + self.features
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by device and time to ensure proper rolling
        X_sorted = X.sort_values(['udi', 'timestamp']).reset_index(drop=True)
        
        # Create copy to avoid modifying original
        X_transformed = X_sorted.copy()
        
        # Add rolling features for each device
        for udi in X_sorted['udi'].unique():
            device_mask = X_sorted['udi'] == udi
            device_data = X_sorted[device_mask].copy()
            
            # Create rolling features for this device
            device_features = self._create_device_rolling_features(device_data)
            
            # Update the transformed dataframe
            X_transformed.loc[device_mask, device_features.columns] = device_features
        
        return X_transformed
    
    def _create_device_rolling_features(self, device_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling features for a single device.
        
        Args:
            device_data: DataFrame for single device (sorted by time)
            
        Returns:
            DataFrame with rolling features
        """
        features_df = pd.DataFrame(index=device_data.index)
        
        for feature in self.features:
            if feature not in device_data.columns:
                continue
                
            feature_series = device_data[feature]
            
            for window in self.windows:
                for agg_func in self.agg_functions:
                    if agg_func == 'slope':
                        # Calculate slope over window (rate of change)
                        rolling_slope = self._calculate_rolling_slope(feature_series, window)
                        col_name = f"{feature}_slope_{window}"
                        features_df[col_name] = rolling_slope
                    else:
                        # Standard rolling statistics
                        rolling_stat = feature_series.rolling(window=window, min_periods=1).agg(agg_func)
                        col_name = f"{feature}_{agg_func}_{window}"
                        features_df[col_name] = rolling_stat
        
        return features_df
    
    def _calculate_rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate rolling slope (rate of change) over window.
        
        Args:
            series: Time series data
            window: Window size
            
        Returns:
            Series with rolling slopes
        """
        slopes = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            if i < window - 1:
                # Not enough data for full window, use available data
                available_data = series.iloc[:i+1]
                if len(available_data) >= 2:
                    x = np.arange(len(available_data))
                    y = available_data.values
                    slope = np.polyfit(x, y, 1)[0]
                else:
                    slope = 0.0
            else:
                # Full window available
                window_data = series.iloc[i-window+1:i+1]
                x = np.arange(len(window_data))
                y = window_data.values
                slope = np.polyfit(x, y, 1)[0]
            
            slopes.iloc[i] = slope
        
        return slopes
    
    def _get_feature_names(self) -> List[str]:
        """Generate list of feature names for consistency."""
        feature_names = []
        
        for feature in self.features:
            for window in self.windows:
                for agg_func in self.agg_functions:
                    col_name = f"{feature}_{agg_func}_{window}"
                    feature_names.append(col_name)
        
        return feature_names

class DeltaFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates delta features (changes over time) for device maintenance prediction.
    """
    
    def __init__(self, features: List[str] = None, periods: List[int] = [1, 5, 10]):
        """
        Initialize delta feature transformer.
        
        Args:
            features: Features to create deltas for
            periods: Periods to calculate deltas over
        """
        if features is None:
            self.features = ['temperatureC', 'pressureKPa', 'vibrationMM_S', 'performanceScore']
        else:
            self.features = features
            
        self.periods = periods
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no fitting needed for delta features)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by adding delta features.
        
        Args:
            X: DataFrame with device data (must have 'udi' and 'timestamp' columns)
            
        Returns:
            DataFrame with original + delta features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Ensure required columns exist
        required_cols = ['udi', 'timestamp'] + self.features
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by device and time
        X_sorted = X.sort_values(['udi', 'timestamp']).reset_index(drop=True)
        X_transformed = X_sorted.copy()
        
        # Add delta features for each device
        for udi in X_sorted['udi'].unique():
            device_mask = X_sorted['udi'] == udi
            device_data = X_sorted[device_mask].copy()
            
            # Create delta features for this device
            device_features = self._create_device_delta_features(device_data)
            
            # Update the transformed dataframe
            X_transformed.loc[device_mask, device_features.columns] = device_features
        
        return X_transformed
    
    def _create_device_delta_features(self, device_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create delta features for a single device.
        
        Args:
            device_data: DataFrame for single device (sorted by time)
            
        Returns:
            DataFrame with delta features
        """
        features_df = pd.DataFrame(index=device_data.index)
        
        for feature in self.features:
            if feature not in device_data.columns:
                continue
                
            feature_series = device_data[feature]
            
            for period in self.periods:
                # Calculate delta (change from t-period to t)
                delta = feature_series.diff(periods=period)
                col_name = f"{feature}_delta_{period}"
                features_df[col_name] = delta
                
                # Calculate percentage change
                pct_change = feature_series.pct_change(periods=period)
                col_name_pct = f"{feature}_pct_change_{period}"
                features_df[col_name_pct] = pct_change
        
        return features_df

class DeviceFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Complete feature pipeline for device maintenance prediction.
    Combines rolling and delta features with proper temporal handling.
    """
    
    def __init__(self, rolling_windows: List[int] = [5, 30, 120],
                 delta_periods: List[int] = [1, 5, 10],
                 features: List[str] = None):
        """
        Initialize feature pipeline.
        
        Args:
            rolling_windows: Window sizes for rolling features
            delta_periods: Periods for delta features
            features: Base features to engineer
        """
        if features is None:
            self.features = ['temperatureC', 'pressureKPa', 'vibrationMM_S', 'performanceScore']
        else:
            self.features = features
            
        self.rolling_windows = rolling_windows
        self.delta_periods = delta_periods
        
        # Initialize transformers
        self.rolling_transformer = RollingFeatureTransformer(
            windows=rolling_windows, 
            features=self.features
        )
        self.delta_transformer = DeltaFeatureTransformer(
            features=self.features,
            periods=delta_periods
        )
        
        # Store feature names
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature pipeline."""
        # Fit transformers
        self.rolling_transformer.fit(X)
        self.delta_transformer.fit(X)
        
        # Generate feature names
        self.feature_names_ = self._get_feature_names()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data through complete feature pipeline.
        
        Args:
            X: DataFrame with device data
            
        Returns:
            DataFrame with engineered features
        """
        # Apply rolling features
        X_rolling = self.rolling_transformer.transform(X)
        
        # Apply delta features
        X_delta = self.delta_transformer.transform(X)
        
        # Combine features
        X_combined = pd.concat([
            X_rolling,
            X_delta.drop(columns=['udi', 'timestamp'] + self.features, errors='ignore')
        ], axis=1)
        
        return X_combined
    
    def _get_feature_names(self) -> List[str]:
        """Get all feature names from the pipeline."""
        feature_names = []
        
        # Original features
        feature_names.extend(self.features)
        
        # Rolling feature names
        feature_names.extend(self.rolling_transformer._get_feature_names())
        
        # Delta feature names
        for feature in self.features:
            for period in self.delta_periods:
                feature_names.extend([
                    f"{feature}_delta_{period}",
                    f"{feature}_pct_change_{period}"
                ])
        
        return feature_names

def create_feature_pipeline(rolling_windows: List[int] = [5, 30, 120],
                           delta_periods: List[int] = [1, 5, 10]) -> DeviceFeaturePipeline:
    """
    Create a feature pipeline for device maintenance prediction.
    
    Args:
        rolling_windows: Window sizes for rolling features
        delta_periods: Periods for delta features
        
    Returns:
        Configured feature pipeline
    """
    return DeviceFeaturePipeline(
        rolling_windows=rolling_windows,
        delta_periods=delta_periods
    )

if __name__ == "__main__":
    # Example usage
    from data_simulation import create_synthetic_dataset
    
    # Generate synthetic data
    df = create_synthetic_dataset()
    
    # Create feature pipeline
    pipeline = create_feature_pipeline()
    
    # Fit and transform
    pipeline.fit(df)
    df_features = pipeline.transform(df)
    
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_features.columns)}")
    print(f"Feature names: {pipeline.feature_names_[:10]}...")  # Show first 10