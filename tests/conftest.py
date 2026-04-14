import pytest
import pandas as pd
import numpy as np
import os

@pytest.fixture
def dummy_air_quality_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'id': range(1, 101),
        'date_hour': dates,
        'temperature_c': np.random.uniform(10, 35, size=100),
        'humidity_pct': np.random.uniform(30, 90, size=100),
        'pressure_hpa': np.random.uniform(1000, 1025, size=100),
        'wind_speed_ms': np.random.uniform(0, 15, size=100),
        'wind_direction_deg': np.random.uniform(0, 360, size=100),
        'cloudiness_pct': np.random.uniform(0, 100, size=100),
        'aqi_general': np.random.uniform(1, 5, size=100),
        'co': np.random.uniform(200, 800, size=100),
        'no2': np.random.uniform(10, 100, size=100),
        'o3': np.random.uniform(20, 120, size=100),
        'pm2_5': np.random.uniform(5, 50, size=100),
        'pm10': np.random.uniform(10, 100, size=100),
        'risk_category': ['Moderate'] * 100,
        'risk_severity': np.random.randint(1, 6, size=100),
        'main_pollutant': ['pm2_5'] * 100,
        'register_date': dates
    })
    return df

@pytest.fixture
def mock_schema():
    return {
        'columns': {
            'date_hour': 'object', 'temperature_c': 'float64',
            'humidity_pct': 'float64', 'risk_severity': 'int64'
        },
        'target_column': 'risk_severity',
        'drop_columns': ['id', 'register_date', 'aqi_general', 'risk_category', 'main_pollutant']
    }