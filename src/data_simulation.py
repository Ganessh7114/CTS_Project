"""
Data Simulation for Predictive Maintenance System
Generates realistic device data with lifecycle transitions and temporal patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any
import random

class DeviceDataSimulator:
    """Simulates realistic device data with degradation patterns."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Device lifecycle parameters
        self.device_types = ['pump', 'compressor', 'turbine', 'motor']
        self.max_life_hours = 10000  # Maximum device life
        
        # State transition probabilities (realistic degradation)
        self.transition_matrix = {
            'III': {'III': 0.95, 'II': 0.05, 'I': 0.00},  # Safe -> Warning
            'II': {'III': 0.10, 'II': 0.80, 'I': 0.10},   # Warning -> Critical or recovery
            'I': {'III': 0.05, 'II': 0.15, 'I': 0.80}     # Critical -> mostly stays critical
        }
        
        # Feature ranges by state
        self.feature_ranges = {
            'III': {  # Safe
                'temperatureC': (20, 60),
                'pressureKPa': (100, 200),
                'vibrationMM_S': (0.1, 0.5),
                'performanceScore': (0.85, 1.0)
            },
            'II': {   # Warning
                'temperatureC': (50, 80),
                'pressureKPa': (180, 250),
                'vibrationMM_S': (0.4, 1.2),
                'performanceScore': (0.60, 0.85)
            },
            'I': {    # Critical
                'temperatureC': (70, 120),
                'pressureKPa': (220, 350),
                'vibrationMM_S': (1.0, 3.0),
                'performanceScore': (0.30, 0.65)
            }
        }
    
    def generate_device_lifecycle(self, udi: str, device_type: str, 
                                start_time: datetime, hours_per_reading: float = 1.0) -> pd.DataFrame:
        """Generate complete lifecycle for a single device."""
        
        # Randomize device characteristics
        base_life = np.random.normal(self.max_life_hours, self.max_life_hours * 0.2)
        base_life = max(1000, min(base_life, self.max_life_hours))
        
        # Generate time series
        total_readings = int(base_life / hours_per_reading)
        timestamps = [start_time + timedelta(hours=i * hours_per_reading) 
                     for i in range(total_readings)]
        
        # Initialize state progression
        current_state = 'III'
        state_changes = []
        life_remaining = base_life
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            # Update state based on transition matrix
            if i > 0 and i % 100 == 0:  # State can change every 100 readings
                probs = self.transition_matrix[current_state]
                current_state = np.random.choice(list(probs.keys()), p=list(probs.values()))
                if current_state != state_changes[-1] if state_changes else 'III':
                    state_changes.append(current_state)
            
            # Calculate life remaining (decreases faster in worse states)
            degradation_rate = {'III': 1.0, 'II': 1.5, 'I': 3.0}[current_state]
            life_remaining -= hours_per_reading * degradation_rate
            
            # Generate features based on current state
            ranges = self.feature_ranges[current_state]
            
            # Add some temporal correlation and noise
            if i == 0:
                temp = np.random.uniform(*ranges['temperatureC'])
                pressure = np.random.uniform(*ranges['pressureKPa'])
                vibration = np.random.uniform(*ranges['vibrationMM_S'])
                performance = np.random.uniform(*ranges['performanceScore'])
            else:
                # Add temporal correlation (features don't jump randomly)
                temp = np.clip(temp + np.random.normal(0, 2), *ranges['temperatureC'])
                pressure = np.clip(pressure + np.random.normal(0, 5), *ranges['pressureKPa'])
                vibration = np.clip(vibration + np.random.normal(0, 0.1), *ranges['vibrationMM_S'])
                performance = np.clip(performance + np.random.normal(0, 0.02), *ranges['performanceScore'])
            
            # Add some realistic anomalies and recoveries
            if np.random.random() < 0.01:  # 1% chance of temporary spike
                temp *= np.random.uniform(1.1, 1.3)
                pressure *= np.random.uniform(1.05, 1.15)
            
            data.append({
                'udi': udi,
                'deviceType': device_type,
                'deviceName': f"{device_type}_{udi}",
                'timestamp': timestamp,
                'runtimeHours': i * hours_per_reading,
                'temperatureC': round(temp, 2),
                'pressureKPa': round(pressure, 1),
                'vibrationMM_S': round(vibration, 3),
                'performanceScore': round(performance, 3),
                'lifeRemaining': max(0, round(life_remaining, 1)),
                'stateClass': current_state
            })
        
        return pd.DataFrame(data)
    
    def generate_fleet_data(self, num_devices: int = 100, 
                           start_date: datetime = None) -> pd.DataFrame:
        """Generate data for entire fleet."""
        
        if start_date is None:
            start_date = datetime(2024, 1, 1)
        
        all_data = []
        
        for i in range(num_devices):
            udi = f"DEV_{i:06d}"
            device_type = np.random.choice(self.device_types)
            
            # Stagger device start times
            device_start = start_date + timedelta(hours=np.random.randint(0, 1000))
            
            device_data = self.generate_device_lifecycle(udi, device_type, device_start)
            all_data.append(device_data)
        
        fleet_data = pd.concat(all_data, ignore_index=True)
        fleet_data = fleet_data.sort_values(['udi', 'timestamp']).reset_index(drop=True)
        
        return fleet_data
    
    def stream_data_generator(self, fleet_data: pd.DataFrame, 
                            stream_rate: float = 1.0) -> Dict[str, Any]:
        """Generator for streaming data simulation."""
        
        for _, row in fleet_data.iterrows():
            # Convert to streaming format (exclude ground truth)
            stream_record = {
                "udi": row['udi'],
                "deviceType": row['deviceType'],
                "deviceName": row['deviceName'],
                "timestamp": row['timestamp'].isoformat(),
                "runtimeHours": float(row['runtimeHours']),
                "temperatureC": float(row['temperatureC']),
                "pressureKPa": float(row['pressureKPa']),
                "vibrationMM_S": float(row['vibrationMM_S']),
                "performanceScore": float(row['performanceScore'])
            }
            
            yield stream_record
            
            # Simulate streaming rate
            if stream_rate > 0:
                import time
                time.sleep(1.0 / stream_rate)

def create_synthetic_dataset(output_path: str = "data/synthetic_fleet_data.csv"):
    """Create and save synthetic dataset."""
    
    print("Generating synthetic fleet data...")
    simulator = DeviceDataSimulator(seed=42)
    
    # Generate fleet data
    fleet_data = simulator.generate_fleet_data(num_devices=50)
    
    # Create output directory
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save dataset
    fleet_data.to_csv(output_path, index=False)
    
    print(f"Generated {len(fleet_data)} records for {fleet_data['udi'].nunique()} devices")
    print(f"Data saved to: {output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Date range: {fleet_data['timestamp'].min()} to {fleet_data['timestamp'].max()}")
    print(f"State distribution: {fleet_data['stateClass'].value_counts().to_dict()}")
    print(f"Life remaining stats: {fleet_data['lifeRemaining'].describe()}")
    
    return fleet_data

if __name__ == "__main__":
    create_synthetic_dataset()