"""
Temporal Data Splitting for Predictive Maintenance
Implements proper blocked time splits and GroupKFold validation to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from typing import Tuple, List, Dict, Any
from datetime import datetime
import warnings

class TemporalDataSplitter:
    """
    Handles temporal data splitting for device maintenance prediction.
    Ensures no device appears in both train and test at overlapping time windows.
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.2, n_splits: int = 5):
        """
        Initialize splitter with temporal constraints.
        
        Args:
            test_size: Fraction of latest time period for holdout test
            val_size: Fraction of training data for validation (within GroupKFold)
            n_splits: Number of folds for GroupKFold cross-validation
        """
        self.test_size = test_size
        self.val_size = val_size
        self.n_splits = n_splits
        
    def split_by_time(self, df: pd.DataFrame, time_col: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by time to create train/test sets.
        
        Args:
            df: DataFrame with temporal data
            time_col: Column name for timestamp
            
        Returns:
            train_df, test_df: Split datasets
        """
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Sort by time
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        
        # Find the cutoff time for test set (latest test_size fraction)
        total_time_range = df_sorted[time_col].max() - df_sorted[time_col].min()
        test_cutoff = df_sorted[time_col].max() - total_time_range * self.test_size
        
        # Split data
        train_df = df_sorted[df_sorted[time_col] < test_cutoff].copy()
        test_df = df_sorted[df_sorted[time_col] >= test_cutoff].copy()
        
        print(f"Time-based split:")
        print(f"  Train: {len(train_df)} records ({train_df[time_col].min()} to {train_df[time_col].max()})")
        print(f"  Test:  {len(test_df)} records ({test_df[time_col].min()} to {test_df[time_col].max()})")
        print(f"  Test cutoff: {test_cutoff}")
        
        # Verify no device overlap in time windows
        train_devices = set(train_df['udi'].unique())
        test_devices = set(test_df['udi'].unique())
        overlap_devices = train_devices.intersection(test_devices)
        
        if overlap_devices:
            print(f"  Warning: {len(overlap_devices)} devices appear in both train and test")
            print(f"  This is expected for temporal splits - devices can appear in both sets at different times")
        
        return train_df, test_df
    
    def create_group_kfold_splits(self, train_df: pd.DataFrame, 
                                 group_col: str = 'udi') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create GroupKFold splits for cross-validation.
        Each fold contains complete devices (no device split across folds).
        
        Args:
            train_df: Training DataFrame
            group_col: Column to group by (typically device ID)
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        # Get unique groups
        groups = train_df[group_col].values
        
        # Create GroupKFold splits
        gkf = GroupKFold(n_splits=self.n_splits)
        splits = list(gkf.split(train_df, groups=groups))
        
        print(f"GroupKFold splits (n_splits={self.n_splits}):")
        for i, (train_idx, val_idx) in enumerate(splits):
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            overlap = train_groups.intersection(val_groups)
            
            print(f"  Fold {i+1}: Train={len(train_idx)} samples ({len(train_groups)} devices), "
                  f"Val={len(val_idx)} samples ({len(val_groups)} devices)")
            
            if overlap:
                warnings.warn(f"Fold {i+1} has {len(overlap)} overlapping devices!")
        
        return splits
    
    def get_device_statistics(self, df: pd.DataFrame, group_col: str = 'udi') -> Dict[str, Any]:
        """
        Get statistics about device distribution in dataset.
        
        Args:
            df: DataFrame
            group_col: Column to group by
            
        Returns:
            Dictionary with device statistics
        """
        device_stats = df.groupby(group_col).agg({
            'timestamp': ['count', 'min', 'max'],
            'stateClass': lambda x: x.value_counts().to_dict(),
            'lifeRemaining': ['mean', 'min', 'max']
        }).round(2)
        
        device_stats.columns = ['_'.join(col).strip() for col in device_stats.columns]
        
        return {
            'total_devices': len(device_stats),
            'total_records': len(df),
            'avg_records_per_device': device_stats['timestamp_count'].mean(),
            'device_stats': device_stats
        }
    
    def validate_split_integrity(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                time_col: str = 'timestamp', group_col: str = 'udi') -> bool:
        """
        Validate that the split maintains temporal integrity.
        
        Args:
            train_df, test_df: Split datasets
            time_col: Timestamp column
            group_col: Device ID column
            
        Returns:
            True if split is valid, False otherwise
        """
        # Check 1: No future data leakage in train
        train_max_time = train_df[time_col].max()
        test_min_time = test_df[time_col].min()
        
        if train_max_time >= test_min_time:
            print("❌ ERROR: Train set contains data after test set start time")
            return False
        
        # Check 2: All devices in test have some history in train (realistic scenario)
        train_devices = set(train_df[group_col].unique())
        test_devices = set(test_df[group_col].unique())
        
        devices_without_history = test_devices - train_devices
        if devices_without_history:
            print(f"⚠️  WARNING: {len(devices_without_history)} devices in test have no history in train")
            print(f"   This may be realistic for new devices but could affect model performance")
        
        # Check 3: Temporal consistency for overlapping devices
        overlap_devices = train_devices.intersection(test_devices)
        temporal_violations = 0
        
        for device in overlap_devices:
            device_train = train_df[train_df[group_col] == device]
            device_test = test_df[test_df[group_col] == device]
            
            if device_train[time_col].max() >= device_test[time_col].min():
                temporal_violations += 1
        
        if temporal_violations > 0:
            print(f"❌ ERROR: {temporal_violations} devices have temporal violations")
            return False
        
        print("✅ Split validation passed")
        return True

def create_temporal_splits(df: pd.DataFrame, 
                          test_size: float = 0.2,
                          n_splits: int = 5) -> Dict[str, Any]:
    """
    Create complete temporal splits for predictive maintenance.
    
    Args:
        df: Input DataFrame with temporal device data
        test_size: Fraction of latest time for test set
        n_splits: Number of GroupKFold splits
        
    Returns:
        Dictionary containing train/test splits and GroupKFold indices
    """
    
    # Initialize splitter
    splitter = TemporalDataSplitter(test_size=test_size, n_splits=n_splits)
    
    # Create time-based train/test split
    train_df, test_df = splitter.split_by_time(df)
    
    # Validate split integrity
    is_valid = splitter.validate_split_integrity(train_df, test_df)
    if not is_valid:
        raise ValueError("Temporal split validation failed!")
    
    # Create GroupKFold splits for training
    cv_splits = splitter.create_group_kfold_splits(train_df)
    
    # Get statistics
    train_stats = splitter.get_device_statistics(train_df)
    test_stats = splitter.get_device_statistics(test_df)
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'cv_splits': cv_splits,
        'train_stats': train_stats,
        'test_stats': test_stats,
        'splitter': splitter
    }

if __name__ == "__main__":
    # Example usage
    from data_simulation import create_synthetic_dataset
    
    # Generate synthetic data
    df = create_synthetic_dataset()
    
    # Create temporal splits
    splits = create_temporal_splits(df)
    
    print("\n" + "="*50)
    print("TEMPORAL SPLIT SUMMARY")
    print("="*50)
    print(f"Train devices: {splits['train_stats']['total_devices']}")
    print(f"Test devices: {splits['test_stats']['total_devices']}")
    print(f"Train records: {splits['train_stats']['total_records']}")
    print(f"Test records: {splits['test_stats']['total_records']}")
    print(f"CV folds: {len(splits['cv_splits'])}")