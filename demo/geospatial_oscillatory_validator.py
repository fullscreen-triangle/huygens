"""
Geospatial Oscillatory Validator
Comprehensive validation system for biological-technological meta-oscillatory coupling
using real GPS coordinate data and biometric measurements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats, optimize
from scipy.fft import fft, fftfreq
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta, timezone
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Import our custom modules
try:
    from .gps_oscillatory_analysis import GPSOscillatoryAnalyzer
    from .technological_infrastructure_model import TechnologicalOscillatoryInfrastructure
    from .spatiotemporal_coupling_validator import SpatiotemporalCouplingValidator
except ImportError:
    # Fallback for running as standalone script
    import sys
    sys.path.append(os.path.dirname(__file__))
    from gps_oscillatory_analysis import GPSOscillatoryAnalyzer
    from technological_infrastructure_model import TechnologicalOscillatoryInfrastructure
    from spatiotemporal_coupling_validator import SpatiotemporalCouplingValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeospatialOscillatoryValidator:
    """
    Comprehensive validator for biological-technological meta-oscillatory coupling
    using real-world GPS coordinate data, biometric measurements, and infrastructure modeling
    """
    
    def __init__(self):
        self.gps_analyzer = GPSOscillatoryAnalyzer()
        self.infrastructure_model = TechnologicalOscillatoryInfrastructure()
        self.coupling_validator = SpatiotemporalCouplingValidator()
        
        # Validation parameters
        self.validation_config = {
            'min_data_days': 2,                    # Minimum days of data for validation
            'min_gps_points': 1000,               # Minimum GPS data points
            'coupling_threshold': 0.3,            # Minimum coupling strength
            'significance_level': 0.05,           # Statistical significance threshold
            'coherence_threshold': 0.5,           # Minimum coherence for validation
            'infrastructure_analysis': True,      # Include infrastructure analysis
            'biometric_analysis': True,           # Include biometric analysis
            'generate_synthetic_data': True       # Generate synthetic data if needed
        }
        
        # Comprehensive oscillatory frequency database
        self.frequency_database = {
            # Biological frequencies (Hz)
            'circadian': 1.157e-5,
            'ultradian_90min': 1.85e-4,
            'ultradian_120min': 1.39e-4,
            'cardiac_rest': 1.0,
            'cardiac_active': 2.5,
            'respiratory': 0.25,
            'neural_delta': 2.0,
            'neural_theta': 6.0,
            'neural_alpha': 10.5,
            'neural_beta': 21.5,
            'neural_gamma': 65.0,
            'sleep_cycles': 1.67e-4,
            'meal_timing': 3.472e-5,
            'weekly': 1.653e-6,
            'monthly': 3.86e-7,
            
            # Technological frequencies (Hz)
            'gps_l1': 1.57542e9,
            'gps_l2': 1.22760e9,
            'cesium_atomic': 9.192631770e9,
            'cellular_850': 850e6,
            'cellular_1900': 1.9e9,
            'wifi_2_4': 2.4e9,
            'device_crystal': 32.768e3,
            'satellite_orbit': 1/(12*3600)
        }
        
        # Meta-oscillatory coupling patterns
        self.meta_patterns = {}
        self._initialize_meta_patterns()
    
    def _initialize_meta_patterns(self):
        """Initialize expected meta-oscillatory coupling patterns"""
        
        # GPS satellite orbital coupling with circadian rhythms
        self.meta_patterns['gps_circadian'] = {
            'harmonic_ratio': self.frequency_database['satellite_orbit'] / self.frequency_database['circadian'],
            'expected_coupling': 0.7,
            'coupling_mechanism': 'orbital resonance with biological timing',
            'phase_relationship': 'satellite passes correlate with circadian phase'
        }
        
        # Cellular network coupling with activity patterns
        self.meta_patterns['cellular_activity'] = {
            'expected_coupling': 0.5,
            'coupling_mechanism': 'network usage patterns reflect biological activity',
            'phase_relationship': 'peak usage during circadian active periods'
        }
        
        # Device oscillator entrainment
        self.meta_patterns['device_biological'] = {
            'expected_coupling': 0.4,
            'coupling_mechanism': 'crystal oscillator stability affects biological measurements',
            'phase_relationship': 'measurement precision impacts biological coupling detection'
        }
    
    def load_real_gps_data(self, data_path: str, coordinate_system: str = 'WGS84') -> pd.DataFrame:
        """Load and validate real GPS coordinate data"""
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"GPS data file not found: {data_path}")
        
        logger.info(f"Loading real GPS data from {data_path}")
        
        # Load data based on file format
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                raw_data = json.load(f)
            
            if isinstance(raw_data, list):
                df = pd.DataFrame(raw_data)
            elif isinstance(raw_data, dict):
                df = pd.DataFrame([raw_data])
            else:
                raise ValueError("Invalid JSON format for GPS data")
                
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")
        
        # Validate required columns
        required_columns = ['timestamp', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Parse timestamps with flexible format handling
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        except:
            # Try different timestamp formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S.%f']:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt)
                    if df['timestamp'].dt.tz is None:
                        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                    break
                except:
                    continue
            else:
                raise ValueError("Unable to parse timestamp format")
        
        # Sort by timestamp and remove duplicates
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Validate coordinate ranges
        lat_valid = (-90 <= df['latitude']).all() and (df['latitude'] <= 90).all()
        lon_valid = (-180 <= df['longitude']).all() and (df['longitude'] <= 180).all()
        
        if not lat_valid or not lon_valid:
            raise ValueError("Invalid coordinate values detected")
        
        # Calculate data quality metrics
        data_span = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        data_days = data_span.total_seconds() / (24 * 3600)
        
        logger.info(f"GPS data loaded: {len(df)} points spanning {data_days:.2f} days")
        
        # Validate minimum data requirements
        if len(df) < self.validation_config['min_gps_points']:
            logger.warning(f"GPS data contains only {len(df)} points, minimum recommended: {self.validation_config['min_gps_points']}")
        
        if data_days < self.validation_config['min_data_days']:
            logger.warning(f"GPS data spans only {data_days:.2f} days, minimum recommended: {self.validation_config['min_data_days']}")
        
        return df
    
    def generate_comprehensive_synthetic_data(self, duration_days: int = 7, 
                                            sampling_minutes: int = 5) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive synthetic data for validation testing"""
        
        logger.info(f"Generating {duration_days} days of synthetic multimodal data...")
        
        # Time array
        start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        n_points = duration_days * 24 * (60 // sampling_minutes)
        timestamps = [start_time + timedelta(minutes=i * sampling_minutes) for i in range(n_points)]
        time_seconds = np.array([(ts - start_time).total_seconds() for ts in timestamps])
        
        # Base location (San Francisco)
        base_lat = 37.7749
        base_lon = -122.4194
        
        # Generate realistic movement patterns with multiple oscillatory components
        
        # 1. Circadian movement pattern
        circadian_freq = self.frequency_database['circadian']
        circadian_amp_lat = 0.015  # ~1.5 km commute radius
        circadian_amp_lon = 0.020  # ~2.0 km commute radius
        circadian_phase_lat = np.pi/4  # Offset for realistic commute pattern
        circadian_phase_lon = 0
        
        # 2. Weekly pattern (different weekend behavior)
        weekly_freq = self.frequency_database['weekly']
        weekly_amp = 0.008  # Weekend vs weekday difference
        
        # 3. Ultradian patterns (90-minute cycles)
        ultradian_freq = self.frequency_database['ultradian_90min']
        ultradian_amp = 0.003
        
        # 4. Activity-dependent movement
        activity_pattern = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * circadian_freq * time_seconds + np.pi/2))
        
        # Generate GPS coordinates
        lat = (base_lat + 
               circadian_amp_lat * np.sin(2 * np.pi * circadian_freq * time_seconds + circadian_phase_lat) * activity_pattern +
               weekly_amp * np.sin(2 * np.pi * weekly_freq * time_seconds) +
               ultradian_amp * np.sin(2 * np.pi * ultradian_freq * time_seconds) +
               0.001 * np.random.normal(size=n_points))
        
        lon = (base_lon +
               circadian_amp_lon * np.cos(2 * np.pi * circadian_freq * time_seconds + circadian_phase_lon) * activity_pattern +
               weekly_amp * np.cos(2 * np.pi * weekly_freq * time_seconds + np.pi/3) +
               ultradian_amp * np.cos(2 * np.pi * ultradian_freq * time_seconds + np.pi/6) +
               0.001 * np.random.normal(size=n_points))
        
        # Generate biometric data with realistic coupling to movement and circadian rhythms
        
        # Heart rate with circadian variation and activity coupling
        base_hr = 70
        circadian_hr_variation = 15 * np.sin(2 * np.pi * circadian_freq * time_seconds + np.pi/6)
        activity_hr_coupling = 30 * activity_pattern
        hr_noise = 8 * np.random.normal(size=n_points)
        heart_rate = base_hr + circadian_hr_variation + activity_hr_coupling + hr_noise
        heart_rate = np.clip(heart_rate, 45, 180)  # Physiological limits
        
        # Activity levels coupled to circadian rhythm and GPS movement
        base_activity = 0.3
        circadian_activity = 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * circadian_freq * time_seconds + np.pi/2))
        movement_coupling = 0.2 * activity_pattern
        activity_noise = 0.1 * np.random.normal(size=n_points)
        activity_level = base_activity + circadian_activity + movement_coupling + activity_noise
        activity_level = np.clip(activity_level, 0, 1)
        
        # Sleep data (binary: 1 = asleep, 0 = awake)
        # Sleep typically occurs during circadian minimum
        sleep_probability = 0.9 * (0.5 - 0.5 * np.sin(2 * np.pi * circadian_freq * time_seconds + np.pi/2))
        sleep_probability = np.clip(sleep_probability, 0, 0.95)
        sleep_status = (np.random.random(n_points) < sleep_probability).astype(int)
        
        # Step count data with realistic daily patterns
        base_steps_per_interval = 50  # Steps per 5-minute interval
        circadian_steps_multiplier = 1 + 0.8 * (0.5 + 0.5 * np.sin(2 * np.pi * circadian_freq * time_seconds + np.pi/3))
        activity_steps_coupling = 1 + activity_pattern
        steps_noise = 0.3 * np.random.exponential(1, n_points)  # Exponential noise for step count
        step_count = base_steps_per_interval * circadian_steps_multiplier * activity_steps_coupling * steps_noise
        step_count = np.maximum(0, step_count).astype(int)  # Steps can't be negative
        
        # Create comprehensive dataset
        datasets = {
            'gps': pd.DataFrame({
                'timestamp': timestamps,
                'latitude': lat,
                'longitude': lon,
                'altitude': 10 + 5 * np.random.normal(size=n_points),  # Simulated altitude
                'accuracy': 5 + 2 * np.abs(np.random.normal(size=n_points)),  # GPS accuracy in meters
                'speed': np.concatenate([[0], np.sqrt(np.diff(lat)**2 + np.diff(lon)**2) * 111000 / (sampling_minutes * 60)]),  # Speed in m/s
                'bearing': (180/np.pi) * np.arctan2(np.diff(np.concatenate([[lon[0]], lon])), 
                                                   np.diff(np.concatenate([[lat[0]], lat]))) % 360  # Bearing in degrees
            }),
            
            'biometric': pd.DataFrame({
                'timestamp': timestamps,
                'heart_rate': heart_rate,
                'activity_level': activity_level,
                'sleep_status': sleep_status,
                'step_count': step_count,
                'stress_level': 0.3 + 0.2 * (1 - activity_pattern) + 0.1 * np.random.normal(size=n_points),  # Stress inversely related to activity
                'body_temperature': 98.6 + 1.5 * np.sin(2 * np.pi * circadian_freq * time_seconds) + 0.3 * np.random.normal(size=n_points)  # Circadian temperature
            }),
            
            'environmental': pd.DataFrame({
                'timestamp': timestamps,
                'temperature_celsius': 20 + 10 * np.sin(2 * np.pi * circadian_freq * time_seconds + np.pi/8) + 3 * np.random.normal(size=n_points),
                'humidity_percent': 50 + 20 * np.sin(2 * np.pi * circadian_freq * time_seconds + np.pi/4) + 5 * np.random.normal(size=n_points),
                'light_lux': np.maximum(0, 1000 * (0.5 + 0.5 * np.sin(2 * np.pi * circadian_freq * time_seconds + np.pi/2)) + 200 * np.random.normal(size=n_points)),
                'noise_db': 45 + 15 * activity_pattern + 5 * np.random.normal(size=n_points)
            })
        }
        
        # Clean up datasets (clip to reasonable ranges)
        datasets['biometric']['stress_level'] = np.clip(datasets['biometric']['stress_level'], 0, 1)
        datasets['environmental']['temperature_celsius'] = np.clip(datasets['environmental']['temperature_celsius'], -10, 50)
        datasets['environmental']['humidity_percent'] = np.clip(datasets['environmental']['humidity_percent'], 0, 100)
        datasets['environmental']['noise_db'] = np.clip(datasets['environmental']['noise_db'], 30, 90)
        
        logger.info(f"Synthetic data generation complete: {len(datasets)} modalities")
        return datasets
    
    def perform_comprehensive_validation(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Perform comprehensive validation of biological-technological meta-oscillatory coupling"""
        
        logger.info("Starting comprehensive validation of meta-oscillatory coupling...")
        
        validation_results = {
            'data_quality': {},
            'gps_oscillatory_analysis': {},
            'infrastructure_coupling': {},
            'spatiotemporal_coupling': {},
            'biometric_coupling': {},
            'meta_oscillatory_networks': {},
            'overall_validation': {}
        }
        
        gps_df = data['gps']
        biometric_df = data.get('biometric', None)
        
        # 1. Data Quality Assessment
        logger.info("Assessing data quality...")
        validation_results['data_quality'] = self._assess_data_quality(data)
        
        # 2. GPS Oscillatory Pattern Analysis
        logger.info("Analyzing GPS oscillatory patterns...")
        gps_movement_df = self.gps_analyzer.calculate_movement_metrics(gps_df)
        gps_oscillatory_results = self.gps_analyzer.detect_oscillatory_patterns(gps_movement_df)
        gps_coupling_results = self.gps_analyzer.calculate_coupling_strength(gps_movement_df, gps_oscillatory_results)
        gps_coherence_results = self.gps_analyzer.calculate_spatiotemporal_coherence(gps_movement_df)
        
        validation_results['gps_oscillatory_analysis'] = {
            'oscillatory_patterns': gps_oscillatory_results,
            'coupling_strength': gps_coupling_results,
            'coherence_metrics': gps_coherence_results
        }
        
        # 3. Technological Infrastructure Coupling Analysis
        if self.validation_config['infrastructure_analysis']:
            logger.info("Analyzing technological infrastructure coupling...")
            
            # Model infrastructure systems
            time_hours = (gps_df['timestamp'].iloc[-1] - gps_df['timestamp'].iloc[0]).total_seconds() / 3600
            sampling_rate = len(gps_df) / time_hours / 3600  # Hz
            
            gps_model = self.infrastructure_model.model_gps_satellite_system(time_hours, sampling_rate)
            cellular_model = self.infrastructure_model.model_cellular_network(time_hours, sampling_rate)
            device_model = self.infrastructure_model.model_smart_device_ensemble(time_hours, sampling_rate)
            
            infrastructure_coupling = self.infrastructure_model.calculate_biological_coupling(
                gps_model, cellular_model, device_model, bio_signal_type='circadian'
            )
            
            validation_results['infrastructure_coupling'] = {
                'gps_satellite_model': gps_model,
                'cellular_network_model': cellular_model,
                'device_ensemble_model': device_model,
                'coupling_analysis': infrastructure_coupling
            }
        
        # 4. Spatiotemporal Coupling Validation
        logger.info("Validating spatiotemporal coupling...")
        spatiotemporal_data = {'gps': gps_df, 'biometric': biometric_df, 'activity': None}
        spatiotemporal_results = self.coupling_validator.validate_biological_rhythm_coupling(
            spatiotemporal_data, bio_rhythm='circadian'
        )
        validation_results['spatiotemporal_coupling'] = spatiotemporal_results
        
        # 5. Biometric Coupling Analysis
        if biometric_df is not None and self.validation_config['biometric_analysis']:
            logger.info("Analyzing biometric coupling patterns...")
            biometric_coupling = self._analyze_biometric_coupling(gps_df, biometric_df)
            validation_results['biometric_coupling'] = biometric_coupling
        
        # 6. Meta-Oscillatory Network Analysis
        logger.info("Analyzing meta-oscillatory network properties...")
        meta_network_results = self._analyze_meta_oscillatory_networks(validation_results)
        validation_results['meta_oscillatory_networks'] = meta_network_results
        
        # 7. Overall Validation Assessment
        logger.info("Computing overall validation metrics...")
        overall_validation = self._compute_overall_validation(validation_results)
        validation_results['overall_validation'] = overall_validation
        
        logger.info("Comprehensive validation complete!")
        return validation_results
    
    def _assess_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Assess quality of input data"""
        
        quality_metrics = {}
        
        for data_type, df in data.items():
            if df is None:
                continue
                
            # Basic statistics
            n_points = len(df)
            duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 3600  # hours
            sampling_rate = n_points / duration if duration > 0 else 0
            
            # Missing data assessment
            missing_data = df.isnull().sum().sum()
            missing_percentage = missing_data / (len(df) * len(df.columns)) * 100
            
            # Temporal regularity
            time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
            temporal_regularity = 1 / (1 + np.std(time_diffs) / np.mean(time_diffs)) if np.mean(time_diffs) > 0 else 0
            
            quality_metrics[data_type] = {
                'n_points': n_points,
                'duration_hours': duration,
                'sampling_rate_hz': sampling_rate,
                'missing_data_percentage': missing_percentage,
                'temporal_regularity': temporal_regularity,
                'quality_score': (1 - missing_percentage/100) * temporal_regularity
            }
        
        return quality_metrics
    
    def _analyze_biometric_coupling(self, gps_df: pd.DataFrame, biometric_df: pd.DataFrame) -> Dict:
        """Analyze coupling between GPS patterns and biometric data"""
        
        coupling_results = {}
        
        # Align timestamps
        gps_times = (gps_df['timestamp'] - gps_df['timestamp'].iloc[0]).dt.total_seconds()
        bio_times = (biometric_df['timestamp'] - biometric_df['timestamp'].iloc[0]).dt.total_seconds()
        
        # Find common time range
        min_time = max(gps_times.min(), bio_times.min())
        max_time = min(gps_times.max(), bio_times.max())
        
        if max_time <= min_time:
            logger.warning("No overlapping time range between GPS and biometric data")
            return coupling_results
        
        # Create common time base
        common_times = np.linspace(min_time, max_time, 1000)
        
        # Interpolate GPS speed
        gps_speeds = np.sqrt(np.diff(gps_df['latitude'])**2 + np.diff(gps_df['longitude'])**2)
        gps_speed_interp = np.interp(common_times, gps_times[1:len(gps_speeds)+1], gps_speeds)
        
        # Analyze different biometric signals
        for bio_column in ['heart_rate', 'activity_level', 'step_count']:
            if bio_column not in biometric_df.columns:
                continue
                
            bio_signal = biometric_df[bio_column].values
            bio_signal_interp = np.interp(common_times, bio_times, bio_signal)
            
            # Calculate correlation
            correlation = np.corrcoef(gps_speed_interp, bio_signal_interp)[0, 1]
            significance = stats.pearsonr(gps_speed_interp, bio_signal_interp)[1]
            
            # Cross-correlation for lag analysis
            cross_corr = signal.correlate(gps_speed_interp - np.mean(gps_speed_interp),
                                        bio_signal_interp - np.mean(bio_signal_interp), mode='full')
            cross_corr_normalized = cross_corr / len(gps_speed_interp)
            
            # Find peak lag
            peak_idx = np.argmax(np.abs(cross_corr_normalized))
            lag_samples = peak_idx - len(gps_speed_interp) + 1
            lag_minutes = lag_samples * (max_time - min_time) / len(common_times) / 60
            
            coupling_results[f'gps_speed_{bio_column}'] = {
                'correlation': abs(correlation),
                'significance': significance,
                'lag_minutes': lag_minutes,
                'coupling_strength': abs(correlation) * (1 if significance < 0.05 else 0.5)
            }
        
        return coupling_results
    
    def _analyze_meta_oscillatory_networks(self, validation_results: Dict) -> Dict:
        """Analyze emergent properties of meta-oscillatory networks"""
        
        network_results = {}
        
        # Collect coupling strengths from different analyses
        coupling_strengths = []
        coupling_types = []
        
        # GPS oscillatory coupling
        gps_coupling = validation_results.get('gps_oscillatory_analysis', {}).get('coupling_strength', {})
        for signal_type, couplings in gps_coupling.items():
            for bio_type, coupling_data in couplings.items():
                if 'max_correlation' in coupling_data:
                    coupling_strengths.append(coupling_data['max_correlation'])
                    coupling_types.append(f'gps_{signal_type}_{bio_type}')
        
        # Infrastructure coupling
        infra_coupling = validation_results.get('infrastructure_coupling', {}).get('coupling_analysis', {})
        if 'meta_network' in infra_coupling:
            meta_network = infra_coupling['meta_network']
            coupling_strengths.append(meta_network.get('synchronization', 0))
            coupling_types.append('infrastructure_synchronization')
            coupling_strengths.append(meta_network.get('overall_coupling', 0))
            coupling_types.append('infrastructure_overall')
        
        # Spatiotemporal coupling
        spatiotemporal = validation_results.get('spatiotemporal_coupling', {})
        if 'validation_metrics' in spatiotemporal:
            validation_metrics = spatiotemporal['validation_metrics']
            coupling_strengths.append(validation_metrics.get('mean_correlation', 0))
            coupling_types.append('spatiotemporal_mean')
            coupling_strengths.append(validation_metrics.get('max_correlation', 0))
            coupling_types.append('spatiotemporal_max')
        
        # Biometric coupling
        bio_coupling = validation_results.get('biometric_coupling', {})
        for coupling_type, coupling_data in bio_coupling.items():
            if 'coupling_strength' in coupling_data:
                coupling_strengths.append(coupling_data['coupling_strength'])
                coupling_types.append(f'biometric_{coupling_type}')
        
        # Calculate network properties
        if coupling_strengths:
            coupling_array = np.array(coupling_strengths)
            
            network_results = {
                'n_coupling_types': len(coupling_strengths),
                'mean_coupling_strength': np.mean(coupling_array),
                'max_coupling_strength': np.max(coupling_array),
                'coupling_consistency': 1 - np.std(coupling_array) / np.mean(coupling_array) if np.mean(coupling_array) > 0 else 0,
                'network_synchronization': np.mean(coupling_array),
                'coupling_distribution': {
                    'quartile_25': np.percentile(coupling_array, 25),
                    'median': np.median(coupling_array),
                    'quartile_75': np.percentile(coupling_array, 75)
                },
                'strong_couplings': sum(1 for c in coupling_array if c >= self.validation_config['coupling_threshold']),
                'coupling_types': coupling_types
            }
            
            # Network topology metrics (simplified)
            # Create adjacency matrix based on coupling strengths
            n = len(coupling_strengths)
            if n > 1:
                adj_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(i+1, n):
                        # Connection strength based on similarity of coupling strengths
                        connection = 1 - abs(coupling_array[i] - coupling_array[j])
                        adj_matrix[i, j] = adj_matrix[j, i] = connection
                
                # Network density
                network_density = np.sum(adj_matrix > 0.5) / (n * (n - 1))
                network_results['network_density'] = network_density
                
                # Average path length (simplified)
                avg_connection = np.mean(adj_matrix[adj_matrix > 0])
                network_results['average_connection_strength'] = avg_connection
        
        return network_results
    
    def _compute_overall_validation(self, validation_results: Dict) -> Dict:
        """Compute overall validation score and classification"""
        
        # Collect validation scores from different components
        component_scores = []
        component_weights = []
        
        # GPS oscillatory analysis weight
        gps_coherence = validation_results.get('gps_oscillatory_analysis', {}).get('coherence_metrics', {})
        if 'overall_coherence' in gps_coherence:
            component_scores.append(gps_coherence['overall_coherence'])
            component_weights.append(0.25)
        
        # Infrastructure coupling weight
        infra_coupling = validation_results.get('infrastructure_coupling', {}).get('coupling_analysis', {})
        if 'meta_network' in infra_coupling:
            meta_score = infra_coupling['meta_network'].get('overall_coupling', 0)
            component_scores.append(meta_score)
            component_weights.append(0.25)
        
        # Spatiotemporal coupling weight
        spatiotemporal = validation_results.get('spatiotemporal_coupling', {})
        if 'validation_metrics' in spatiotemporal:
            spatial_score = spatiotemporal['validation_metrics'].get('validation_score', 0)
            component_scores.append(spatial_score)
            component_weights.append(0.30)
        
        # Meta-network analysis weight
        meta_network = validation_results.get('meta_oscillatory_networks', {})
        if 'network_synchronization' in meta_network:
            network_score = meta_network['network_synchronization']
            component_scores.append(network_score)
            component_weights.append(0.20)
        
        # Calculate weighted overall score
        if component_scores:
            # Normalize weights
            total_weight = sum(component_weights)
            normalized_weights = [w / total_weight for w in component_weights]
            
            overall_score = sum(score * weight for score, weight in zip(component_scores, normalized_weights))
        else:
            overall_score = 0
        
        # Determine validation classification
        if overall_score >= 0.8:
            classification = "STRONG META-OSCILLATORY COUPLING"
            confidence = "High"
            recommendation = "Strong evidence supports biological-technological meta-oscillatory coupling hypothesis"
        elif overall_score >= 0.6:
            classification = "MODERATE META-OSCILLATORY COUPLING"
            confidence = "Moderate"
            recommendation = "Moderate evidence supports coupling; additional data may strengthen validation"
        elif overall_score >= 0.4:
            classification = "WEAK META-OSCILLATORY COUPLING"
            confidence = "Low"
            recommendation = "Limited evidence for coupling; longer data collection or different analysis approaches needed"
        else:
            classification = "NO SIGNIFICANT META-OSCILLATORY COUPLING"
            confidence = "Very Low"
            recommendation = "No significant evidence for biological-technological meta-oscillatory coupling detected"
        
        return {
            'overall_score': overall_score,
            'classification': classification,
            'confidence': confidence,
            'recommendation': recommendation,
            'component_scores': dict(zip(['gps', 'infrastructure', 'spatiotemporal', 'meta_network'], component_scores)),
            'component_weights': dict(zip(['gps', 'infrastructure', 'spatiotemporal', 'meta_network'], component_weights))
        }
    
    def generate_comprehensive_report(self, validation_results: Dict) -> str:
        """Generate comprehensive validation report"""
        
        overall = validation_results['overall_validation']
        
        report = f"""
=== COMPREHENSIVE GEOSPATIAL OSCILLATORY VALIDATION REPORT ===

OVERALL VALIDATION RESULTS:
- Validation Score: {overall['overall_score']:.3f}/1.000
- Classification: {overall['classification']}
- Confidence Level: {overall['confidence']}

EXECUTIVE SUMMARY:
{overall['recommendation']}

DATA QUALITY ASSESSMENT:
"""
        
        # Data quality section
        data_quality = validation_results.get('data_quality', {})
        for data_type, quality in data_quality.items():
            report += f"\n{data_type.upper()} Data Quality:\n"
            report += f"  - Data points: {quality['n_points']:,}\n"
            report += f"  - Duration: {quality['duration_hours']:.2f} hours ({quality['duration_hours']/24:.1f} days)\n"
            report += f"  - Sampling rate: {quality['sampling_rate_hz']:.4f} Hz\n"
            report += f"  - Missing data: {quality['missing_data_percentage']:.2f}%\n"
            report += f"  - Quality score: {quality['quality_score']:.3f}\n"
        
        # GPS oscillatory analysis section
        gps_analysis = validation_results.get('gps_oscillatory_analysis', {})
        if gps_analysis:
            report += f"\nGPS OSCILLATORY ANALYSIS:\n"
            coherence = gps_analysis.get('coherence_metrics', {})
            if coherence:
                report += f"  - Overall coherence: {coherence['overall_coherence']:.3f}\n"
                report += f"  - Speed coherence: {coherence['speed_coherence']:.3f}\n"
                report += f"  - Temporal regularity: {coherence['temporal_regularity']:.3f}\n"
        
        # Infrastructure coupling section
        infra_coupling = validation_results.get('infrastructure_coupling', {})
        if infra_coupling and 'coupling_analysis' in infra_coupling:
            coupling_analysis = infra_coupling['coupling_analysis']
            report += f"\nTECHNOLOGICAL INFRASTRUCTURE COUPLING:\n"
            
            if 'gps' in coupling_analysis:
                gps_coup = coupling_analysis['gps']
                report += f"  GPS Satellite Coupling:\n"
                report += f"    - Correlation: {gps_coup['correlation']:.4f}\n"
                report += f"    - Coupling strength: {gps_coup['coupling_strength']:.4f}\n"
            
            if 'cellular' in coupling_analysis:
                cellular_coup = coupling_analysis['cellular']
                report += f"  Cellular Network Coupling:\n"
                report += f"    - Overall correlation: {cellular_coup['correlation']:.4f}\n"
                report += f"    - Coupling strength: {cellular_coup['coupling_strength']:.4f}\n"
            
            if 'device' in coupling_analysis:
                device_coup = coupling_analysis['device']
                report += f"  Smart Device Coupling:\n"
                report += f"    - Overall correlation: {device_coup['correlation']:.4f}\n"
                report += f"    - Coupling strength: {device_coup['coupling_strength']:.4f}\n"
        
        # Spatiotemporal coupling section
        spatiotemporal = validation_results.get('spatiotemporal_coupling', {})
        if spatiotemporal:
            report += f"\nSPATIOTEMPORAL COUPLING VALIDATION:\n"
            if 'validation_metrics' in spatiotemporal:
                metrics = spatiotemporal['validation_metrics']
                report += f"  - Validation score: {metrics['validation_score']:.3f}\n"
                report += f"  - Classification: {metrics['coupling_classification']}\n"
                report += f"  - Mean correlation: {metrics['mean_correlation']:.4f}\n"
                report += f"  - Max correlation: {metrics['max_correlation']:.4f}\n"
                report += f"  - Significance ratio: {metrics['significance_ratio']:.1%}\n"
        
        # Meta-oscillatory networks section
        meta_networks = validation_results.get('meta_oscillatory_networks', {})
        if meta_networks:
            report += f"\nMETA-OSCILLATORY NETWORK ANALYSIS:\n"
            report += f"  - Network synchronization: {meta_networks['network_synchronization']:.3f}\n"
            report += f"  - Coupling types analyzed: {meta_networks['n_coupling_types']}\n"
            report += f"  - Strong couplings detected: {meta_networks['strong_couplings']}\n"
            report += f"  - Mean coupling strength: {meta_networks['mean_coupling_strength']:.3f}\n"
            report += f"  - Coupling consistency: {meta_networks['coupling_consistency']:.3f}\n"
        
        # Biometric coupling section
        biometric_coupling = validation_results.get('biometric_coupling', {})
        if biometric_coupling:
            report += f"\nBIOMETRIC COUPLING ANALYSIS:\n"
            for coupling_type, coupling_data in biometric_coupling.items():
                report += f"  {coupling_type.replace('_', ' ').title()}:\n"
                report += f"    - Correlation: {coupling_data['correlation']:.4f}\n"
                report += f"    - Significance: p = {coupling_data['significance']:.4f}\n"
                report += f"    - Coupling strength: {coupling_data['coupling_strength']:.4f}\n"
        
        report += f"\n" + "="*60 + "\n"
        
        return report
    
    def plot_comprehensive_validation(self, validation_results: Dict, data: Dict[str, pd.DataFrame],
                                    save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of validation results"""
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Comprehensive Geospatial Oscillatory Validation', fontsize=16)
        
        gps_df = data['gps']
        biometric_df = data.get('biometric', None)
        
        # Plot 1: GPS trajectory with time coloring
        time_hours = (gps_df['timestamp'] - gps_df['timestamp'].iloc[0]).dt.total_seconds() / 3600
        scatter = axes[0, 0].scatter(gps_df['longitude'], gps_df['latitude'], 
                                   c=time_hours, cmap='viridis', alpha=0.6, s=15)
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].set_title('GPS Trajectory')
        plt.colorbar(scatter, ax=axes[0, 0], label='Time (hours)')
        
        # Plot 2: Movement velocity over time
        if 'speed' in gps_df.columns:
            axes[0, 1].plot(time_hours, gps_df['speed'], alpha=0.7)
            axes[0, 1].set_xlabel('Time (hours)')
            axes[0, 1].set_ylabel('Speed (m/s)')
            axes[0, 1].set_title('Movement Speed')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Biometric data (heart rate if available)
        if biometric_df is not None and 'heart_rate' in biometric_df.columns:
            bio_time_hours = (biometric_df['timestamp'] - biometric_df['timestamp'].iloc[0]).dt.total_seconds() / 3600
            axes[0, 2].plot(bio_time_hours, biometric_df['heart_rate'], alpha=0.7, color='red')
            axes[0, 2].set_xlabel('Time (hours)')
            axes[0, 2].set_ylabel('Heart Rate (BPM)')
            axes[0, 2].set_title('Heart Rate Over Time')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Oscillatory coupling strengths
        coupling_data = []
        coupling_labels = []
        
        # Collect coupling strengths from different analyses
        infra_coupling = validation_results.get('infrastructure_coupling', {}).get('coupling_analysis', {})
        if 'gps' in infra_coupling:
            coupling_data.append(infra_coupling['gps']['coupling_strength'])
            coupling_labels.append('GPS Satellite')
        if 'cellular' in infra_coupling:
            coupling_data.append(infra_coupling['cellular']['coupling_strength'])
            coupling_labels.append('Cellular Network')
        if 'device' in infra_coupling:
            coupling_data.append(infra_coupling['device']['coupling_strength'])
            coupling_labels.append('Smart Devices')
        
        if coupling_data:
            bars = axes[1, 0].bar(coupling_labels, coupling_data, alpha=0.7, color=['blue', 'orange', 'green'])
            axes[1, 0].set_ylabel('Coupling Strength')
            axes[1, 0].set_title('Infrastructure Coupling')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add values on bars
            for bar, value in zip(bars, coupling_data):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 5: Meta-network properties
        meta_networks = validation_results.get('meta_oscillatory_networks', {})
        if meta_networks:
            network_metrics = ['Mean Coupling', 'Max Coupling', 'Consistency', 'Synchronization']
            network_values = [
                meta_networks.get('mean_coupling_strength', 0),
                meta_networks.get('max_coupling_strength', 0),
                meta_networks.get('coupling_consistency', 0),
                meta_networks.get('network_synchronization', 0)
            ]
            
            axes[1, 1].bar(network_metrics, network_values, alpha=0.7, color='purple')
            axes[1, 1].set_ylabel('Metric Value')
            axes[1, 1].set_title('Meta-Network Properties')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Overall validation score
        overall = validation_results.get('overall_validation', {})
        if overall:
            component_scores = overall.get('component_scores', {})
            score_names = list(component_scores.keys())
            score_values = list(component_scores.values())
            
            if score_names and score_values:
                axes[1, 2].bar(score_names, score_values, alpha=0.7, color='red')
                axes[1, 2].axhline(y=overall['overall_score'], color='black', linestyle='--', 
                                  label=f"Overall: {overall['overall_score']:.3f}")
                axes[1, 2].set_ylabel('Validation Score')
                axes[1, 2].set_title('Component Validation Scores')
                axes[1, 2].tick_params(axis='x', rotation=45)
                axes[1, 2].legend()
        
        # Plot 7: Data quality heatmap
        data_quality = validation_results.get('data_quality', {})
        if data_quality:
            quality_matrix = []
            quality_labels = []
            for data_type, quality in data_quality.items():
                quality_row = [
                    min(quality['duration_hours'] / 48, 1),  # Normalized to 2 days max
                    1 - quality['missing_data_percentage'] / 100,
                    quality['temporal_regularity'],
                    quality['quality_score']
                ]
                quality_matrix.append(quality_row)
                quality_labels.append(data_type.title())
            
            if quality_matrix:
                im = axes[2, 0].imshow(quality_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                axes[2, 0].set_xticks(range(4))
                axes[2, 0].set_xticklabels(['Duration', 'Completeness', 'Regularity', 'Quality'])
                axes[2, 0].set_yticks(range(len(quality_labels)))
                axes[2, 0].set_yticklabels(quality_labels)
                axes[2, 0].set_title('Data Quality Heatmap')
                plt.colorbar(im, ax=axes[2, 0])
        
        # Plot 8: Circadian pattern overlay
        time_seconds = (gps_df['timestamp'] - gps_df['timestamp'].iloc[0]).dt.total_seconds()
        circadian_freq = self.frequency_database['circadian']
        circadian_signal = np.sin(2 * np.pi * circadian_freq * time_seconds)
        
        axes[2, 1].plot(time_hours, circadian_signal, 'k-', linewidth=2, label='Circadian Rhythm')
        
        # Overlay normalized movement speed if available
        if 'speed' in gps_df.columns:
            speed_normalized = (gps_df['speed'] - np.mean(gps_df['speed'])) / np.std(gps_df['speed'])
            axes[2, 1].plot(time_hours, speed_normalized, '--', alpha=0.6, label='Movement (normalized)')
        
        axes[2, 1].set_xlabel('Time (hours)')
        axes[2, 1].set_ylabel('Normalized Amplitude')
        axes[2, 1].set_title('Circadian Pattern Analysis')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Plot 9: Validation classification
        classification = overall.get('classification', 'Unknown')
        confidence = overall.get('confidence', 'Unknown')
        score = overall.get('overall_score', 0)
        
        axes[2, 2].text(0.5, 0.7, classification, ha='center', va='center', fontsize=12, 
                       fontweight='bold', wrap=True, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.5, 0.5, f'Confidence: {confidence}', ha='center', va='center', 
                       fontsize=10, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.5, 0.3, f'Score: {score:.3f}/1.000', ha='center', va='center', 
                       fontsize=10, transform=axes[2, 2].transAxes)
        
        # Create a circular progress indicator for the score
        theta = np.linspace(0, 2*np.pi, 100)
        radius = 0.15
        circle_x = 0.5 + radius * np.cos(theta)
        circle_y = 0.1 + radius * np.sin(theta)
        axes[2, 2].plot(circle_x, circle_y, 'k-', transform=axes[2, 2].transAxes)
        
        # Fill the circle based on score
        fill_theta = np.linspace(0, 2*np.pi*score, int(100*score))
        fill_x = 0.5 + radius * np.cos(fill_theta)
        fill_y = 0.1 + radius * np.sin(fill_theta)
        axes[2, 2].fill(np.concatenate([[0.5], fill_x]), np.concatenate([[0.1], fill_y]), 
                       color='green' if score >= 0.8 else 'orange' if score >= 0.6 else 'red',
                       alpha=0.7, transform=axes[2, 2].transAxes)
        
        axes[2, 2].set_xlim(0, 1)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].set_title('Validation Result')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive validation plot saved to {save_path}")
        
        plt.show()

def main():
    """Comprehensive demonstration of geospatial oscillatory validation"""
    
    logger.info("=" * 60)
    logger.info("GEOSPATIAL OSCILLATORY VALIDATION SYSTEM")
    logger.info("Biological-Technological Meta-Oscillatory Coupling")
    logger.info("=" * 60)
    
    # Initialize validator
    validator = GeospatialOscillatoryValidator()
    
    # Generate comprehensive synthetic data for demonstration
    logger.info("Generating comprehensive synthetic dataset...")
    synthetic_data = validator.generate_comprehensive_synthetic_data(
        duration_days=5, sampling_minutes=5
    )
    
    logger.info("Dataset generation complete:")
    for data_type, df in synthetic_data.items():
        logger.info(f"  - {data_type}: {len(df)} data points")
    
    # Perform comprehensive validation
    logger.info("Performing comprehensive validation analysis...")
    validation_results = validator.perform_comprehensive_validation(synthetic_data)
    
    # Generate and display report
    logger.info("Generating comprehensive validation report...")
    report = validator.generate_comprehensive_report(validation_results)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION REPORT")
    print("="*80)
    print(report)
    
    # Create comprehensive visualization
    logger.info("Creating comprehensive validation visualization...")
    validator.plot_comprehensive_validation(validation_results, synthetic_data)
    
    # Summary statistics
    overall = validation_results['overall_validation']
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY:")
    logger.info(f"Overall Score: {overall['overall_score']:.3f}/1.000")
    logger.info(f"Classification: {overall['classification']}")
    logger.info(f"Confidence: {overall['confidence']}")
    logger.info("="*60)
    
    logger.info("Geospatial oscillatory validation complete!")

if __name__ == "__main__":
    main()
