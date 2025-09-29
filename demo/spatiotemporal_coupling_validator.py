"""
Spatiotemporal Coupling Validator
Validates coupling between movement patterns and biological rhythms
Part of the Technological-Biological Meta-Oscillatory Coupling Framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatiotemporalCouplingValidator:
    """
    Validates coupling between movement patterns and biological rhythms
    using advanced signal processing and statistical analysis
    """
    
    def __init__(self):
        # Biological rhythm frequencies (Hz)
        self.biological_frequencies = {
            'circadian': 1.157e-5,          # 24 hour period
            'ultradian_90min': 1.85e-4,     # 90 minute ultradian
            'ultradian_120min': 1.39e-4,    # 120 minute ultradian
            'cardiac_rest': 1.0,            # 60 BPM resting heart rate
            'cardiac_exercise': 2.5,        # 150 BPM exercise heart rate
            'respiratory': 0.25,            # 15 breaths per minute
            'weekly': 1.653e-6,             # 7 day weekly rhythm
            'monthly': 3.86e-7,             # 30 day monthly rhythm
            'sleep_cycle': 1.67e-4,         # 90 minute sleep cycles
            'meal_rhythm': 3.472e-5         # 8 hour meal rhythm (3 meals/day)
        }
        
        # Movement pattern parameters
        self.movement_parameters = {
            'walking_speed_ms': 1.4,        # Average walking speed m/s
            'commute_distance_km': 15,      # Typical commute distance
            'home_radius_m': 100,           # Home location radius
            'work_radius_m': 200,           # Work location radius
            'activity_threshold_ms': 0.5,   # Movement threshold for activity
            'stationary_threshold_ms': 0.1  # Threshold for stationary periods
        }
        
        # Coupling validation parameters
        self.validation_params = {
            'min_data_hours': 24,           # Minimum data duration for validation
            'significance_threshold': 0.05,  # Statistical significance threshold
            'correlation_threshold': 0.3,   # Minimum correlation for coupling
            'phase_locking_threshold': 0.5, # Minimum phase locking strength
            'coherence_threshold': 0.4      # Minimum coherence for validation
        }
    
    def load_multimodal_data(self, gps_path: str, biometric_path: Optional[str] = None,
                           activity_path: Optional[str] = None) -> Dict:
        """Load multimodal data including GPS, biometric, and activity data"""
        
        # Load GPS data
        logger.info(f"Loading GPS data from {gps_path}")
        if gps_path.endswith('.json'):
            with open(gps_path, 'r') as f:
                gps_data = json.load(f)
            gps_df = pd.DataFrame(gps_data)
        elif gps_path.endswith('.csv'):
            gps_df = pd.read_csv(gps_path)
        else:
            raise ValueError("GPS data must be JSON or CSV format")
        
        gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp'])
        gps_df = gps_df.sort_values('timestamp').reset_index(drop=True)
        
        # Load biometric data if available
        biometric_df = None
        if biometric_path:
            logger.info(f"Loading biometric data from {biometric_path}")
            if biometric_path.endswith('.json'):
                with open(biometric_path, 'r') as f:
                    bio_data = json.load(f)
                biometric_df = pd.DataFrame(bio_data)
            elif biometric_path.endswith('.csv'):
                biometric_df = pd.read_csv(biometric_path)
            
            if biometric_df is not None:
                biometric_df['timestamp'] = pd.to_datetime(biometric_df['timestamp'])
                biometric_df = biometric_df.sort_values('timestamp').reset_index(drop=True)
        
        # Load activity data if available
        activity_df = None
        if activity_path:
            logger.info(f"Loading activity data from {activity_path}")
            if activity_path.endswith('.json'):
                with open(activity_path, 'r') as f:
                    activity_data = json.load(f)
                activity_df = pd.DataFrame(activity_data)
            elif activity_path.endswith('.csv'):
                activity_df = pd.read_csv(activity_path)
            
            if activity_df is not None:
                activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])
                activity_df = activity_df.sort_values('timestamp').reset_index(drop=True)
        
        return {
            'gps': gps_df,
            'biometric': biometric_df,
            'activity': activity_df
        }
    
    def extract_movement_patterns(self, gps_df: pd.DataFrame) -> Dict:
        """Extract detailed movement patterns from GPS data"""
        
        # Calculate time intervals
        gps_df['time_seconds'] = (gps_df['timestamp'] - gps_df['timestamp'].iloc[0]).dt.total_seconds()
        
        # Calculate movement metrics
        lat = gps_df['latitude'].values
        lon = gps_df['longitude'].values
        t = gps_df['time_seconds'].values
        
        # Earth radius in meters
        R = 6371000
        
        # Calculate displacements using Haversine formula
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        dlat = np.diff(lat_rad)
        dlon = np.diff(lon_rad)
        dt = np.diff(t)
        
        # Distance calculations
        a = np.sin(dlat/2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon/2)**2
        distances = 2 * R * np.arcsin(np.sqrt(a))
        
        # Velocities and accelerations
        velocities = distances / dt
        accelerations = np.diff(velocities) / dt[:-1]
        
        # Movement classification
        stationary_mask = velocities < self.movement_parameters['stationary_threshold_ms']
        active_mask = velocities > self.movement_parameters['activity_threshold_ms']
        
        # Location clustering for identifying significant places
        coordinates = np.column_stack([lat, lon])
        n_clusters = min(10, len(coordinates) // 10)  # Adaptive number of clusters
        
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates)
            cluster_centers = kmeans.cluster_centers_
            silhouette_avg = silhouette_score(coordinates, cluster_labels)
        else:
            cluster_labels = np.zeros(len(coordinates))
            cluster_centers = np.array([np.mean(coordinates, axis=0)])
            silhouette_avg = 0
        
        # Calculate dwell times at each cluster
        cluster_dwell_times = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_times = t[cluster_mask]
            if len(cluster_times) > 1:
                total_dwell = cluster_times[-1] - cluster_times[0]
                cluster_dwell_times[cluster_id] = total_dwell / 3600  # Convert to hours
            else:
                cluster_dwell_times[cluster_id] = 0
        
        # Identify home and work locations (largest dwell times)
        sorted_clusters = sorted(cluster_dwell_times.items(), key=lambda x: x[1], reverse=True)
        home_cluster = sorted_clusters[0][0] if len(sorted_clusters) > 0 else 0
        work_cluster = sorted_clusters[1][0] if len(sorted_clusters) > 1 else home_cluster
        
        # Calculate commute patterns
        home_times = t[cluster_labels == home_cluster]
        work_times = t[cluster_labels == work_cluster] if work_cluster != home_cluster else []
        
        return {
            'distances': distances,
            'velocities': velocities,
            'accelerations': accelerations,
            'time_intervals': dt,
            'stationary_periods': stationary_mask,
            'active_periods': active_mask,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'cluster_dwell_times': cluster_dwell_times,
            'silhouette_score': silhouette_avg,
            'home_cluster': home_cluster,
            'work_cluster': work_cluster,
            'home_times': home_times,
            'work_times': work_times,
            'n_significant_places': n_clusters
        }
    
    def detect_circadian_movement_coupling(self, gps_df: pd.DataFrame, 
                                         movement_patterns: Dict) -> Dict:
        """Detect coupling between movement patterns and circadian rhythms"""
        
        time_seconds = gps_df['time_seconds'].values
        time_hours = time_seconds / 3600
        
        # Generate circadian reference signal
        circadian_freq = self.biological_frequencies['circadian']
        circadian_signal = np.sin(2 * np.pi * circadian_freq * time_seconds)
        
        # Analyze different movement aspects for circadian coupling
        coupling_results = {}
        
        # 1. Velocity-circadian coupling
        velocities = movement_patterns['velocities']
        if len(velocities) > 10:
            # Interpolate velocities to match GPS timestamps
            velocity_interp = np.interp(time_seconds[1:], time_seconds[1:len(velocities)+1], velocities)
            
            # Cross-correlation analysis
            velocity_normalized = (velocity_interp - np.mean(velocity_interp)) / np.std(velocity_interp)
            circadian_normalized = (circadian_signal[1:] - np.mean(circadian_signal[1:])) / np.std(circadian_signal[1:])
            
            cross_corr = signal.correlate(velocity_normalized, circadian_normalized, mode='full')
            cross_corr_normalized = cross_corr / len(velocity_normalized)
            
            # Find peak correlation and lag
            peak_idx = np.argmax(np.abs(cross_corr_normalized))
            peak_correlation = cross_corr_normalized[peak_idx]
            lag_samples = peak_idx - len(velocity_normalized) + 1
            lag_hours = lag_samples * np.mean(movement_patterns['time_intervals']) / 3600
            
            # Phase analysis using Hilbert transform
            velocity_analytic = signal.hilbert(velocity_normalized)
            circadian_analytic = signal.hilbert(circadian_normalized)
            
            velocity_phase = np.angle(velocity_analytic)
            circadian_phase = np.angle(circadian_analytic)
            
            phase_diff = velocity_phase - circadian_phase
            phase_locking_value = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            coupling_results['velocity_circadian'] = {
                'correlation': abs(peak_correlation),
                'lag_hours': lag_hours,
                'phase_locking': phase_locking_value,
                'significance': stats.pearsonr(velocity_normalized, circadian_normalized)[1]
            }
        
        # 2. Location clustering circadian coupling
        cluster_labels = movement_patterns['cluster_labels']
        home_cluster = movement_patterns['home_cluster']
        work_cluster = movement_patterns['work_cluster']
        
        # Create location-based signals
        at_home_signal = (cluster_labels == home_cluster).astype(float)
        at_work_signal = (cluster_labels == work_cluster).astype(float) if work_cluster != home_cluster else np.zeros_like(at_home_signal)
        
        # Home presence circadian coupling
        home_correlation = np.corrcoef(at_home_signal, circadian_signal)[0, 1]
        home_significance = stats.pearsonr(at_home_signal, circadian_signal)[1]
        
        coupling_results['home_presence_circadian'] = {
            'correlation': abs(home_correlation),
            'significance': home_significance,
            'expected_phase': 'nighttime presence should correlate with circadian minimum'
        }
        
        # Work presence circadian coupling (if different from home)
        if work_cluster != home_cluster:
            work_correlation = np.corrcoef(at_work_signal, circadian_signal)[0, 1]
            work_significance = stats.pearsonr(at_work_signal, circadian_signal)[1]
            
            coupling_results['work_presence_circadian'] = {
                'correlation': abs(work_correlation),
                'significance': work_significance,
                'expected_phase': 'daytime presence should correlate with circadian maximum'
            }
        
        # 3. Activity pattern circadian coupling
        active_periods = movement_patterns['active_periods']
        if len(active_periods) > 10:
            active_signal = active_periods.astype(float)
            
            # Interpolate to match GPS timestamps
            active_interp = np.interp(time_seconds[1:], time_seconds[1:len(active_signal)+1], active_signal)
            
            activity_correlation = np.corrcoef(active_interp, circadian_signal[1:])[0, 1]
            activity_significance = stats.pearsonr(active_interp, circadian_signal[1:])[1]
            
            coupling_results['activity_circadian'] = {
                'correlation': abs(activity_correlation),
                'significance': activity_significance,
                'expected_phase': 'activity should peak during circadian daytime'
            }
        
        return coupling_results
    
    def validate_biological_rhythm_coupling(self, data: Dict, bio_rhythm: str = 'circadian') -> Dict:
        """Comprehensive validation of biological rhythm coupling with movement patterns"""
        
        if bio_rhythm not in self.biological_frequencies:
            raise ValueError(f"Unknown biological rhythm: {bio_rhythm}")
        
        gps_df = data['gps']
        biometric_df = data['biometric']
        
        # Extract movement patterns
        movement_patterns = self.extract_movement_patterns(gps_df)
        
        # Detect movement-biological rhythm coupling
        if bio_rhythm == 'circadian':
            movement_coupling = self.detect_circadian_movement_coupling(gps_df, movement_patterns)
        else:
            # Generic coupling detection for other rhythms
            movement_coupling = self._detect_generic_rhythm_coupling(gps_df, movement_patterns, bio_rhythm)
        
        validation_results = {
            'biological_rhythm': bio_rhythm,
            'rhythm_frequency': self.biological_frequencies[bio_rhythm],
            'data_duration_hours': (gps_df['timestamp'].iloc[-1] - gps_df['timestamp'].iloc[0]).total_seconds() / 3600,
            'movement_coupling': movement_coupling,
            'validation_metrics': {}
        }
        
        # Biometric validation if available
        if biometric_df is not None:
            biometric_coupling = self._validate_biometric_coupling(biometric_df, movement_patterns, bio_rhythm)
            validation_results['biometric_coupling'] = biometric_coupling
        
        # Calculate overall validation metrics
        validation_metrics = self._calculate_validation_metrics(movement_coupling, validation_results)
        validation_results['validation_metrics'] = validation_metrics
        
        return validation_results
    
    def _detect_generic_rhythm_coupling(self, gps_df: pd.DataFrame, movement_patterns: Dict, 
                                       bio_rhythm: str) -> Dict:
        """Generic coupling detection for any biological rhythm"""
        
        time_seconds = gps_df['time_seconds'].values
        bio_freq = self.biological_frequencies[bio_rhythm]
        bio_signal = np.sin(2 * np.pi * bio_freq * time_seconds)
        
        coupling_results = {}
        
        # Velocity coupling
        velocities = movement_patterns['velocities']
        if len(velocities) > 10:
            velocity_interp = np.interp(time_seconds[1:], time_seconds[1:len(velocities)+1], velocities)
            correlation = np.corrcoef(velocity_interp, bio_signal[1:])[0, 1]
            significance = stats.pearsonr(velocity_interp, bio_signal[1:])[1]
            
            coupling_results['velocity_coupling'] = {
                'correlation': abs(correlation),
                'significance': significance
            }
        
        return coupling_results
    
    def _validate_biometric_coupling(self, biometric_df: pd.DataFrame, movement_patterns: Dict,
                                   bio_rhythm: str) -> Dict:
        """Validate coupling using biometric data (heart rate, activity, etc.)"""
        
        coupling_results = {}
        
        # Heart rate coupling with movement
        if 'heart_rate' in biometric_df.columns:
            hr_values = biometric_df['heart_rate'].values
            hr_times = (biometric_df['timestamp'] - biometric_df['timestamp'].iloc[0]).dt.total_seconds().values
            
            # Correlate heart rate with velocity
            if len(movement_patterns['velocities']) > 10 and len(hr_values) > 10:
                # Find overlapping time periods
                min_time = max(hr_times.min(), movement_patterns['time_intervals'][0])
                max_time = min(hr_times.max(), movement_patterns['time_intervals'][-1])
                
                if max_time > min_time:
                    # Interpolate both signals to common time base
                    common_times = np.linspace(min_time, max_time, 100)
                    hr_interp = np.interp(common_times, hr_times, hr_values)
                    vel_interp = np.interp(common_times, movement_patterns['time_intervals'], 
                                         movement_patterns['velocities'][:len(movement_patterns['time_intervals'])])
                    
                    hr_velocity_corr = np.corrcoef(hr_interp, vel_interp)[0, 1]
                    hr_velocity_sig = stats.pearsonr(hr_interp, vel_interp)[1]
                    
                    coupling_results['heart_rate_velocity'] = {
                        'correlation': abs(hr_velocity_corr),
                        'significance': hr_velocity_sig,
                        'expected': 'positive correlation between heart rate and movement velocity'
                    }
        
        # Activity level coupling
        if 'activity_level' in biometric_df.columns:
            activity_values = biometric_df['activity_level'].values
            activity_times = (biometric_df['timestamp'] - biometric_df['timestamp'].iloc[0]).dt.total_seconds().values
            
            # Correlate with movement patterns
            if len(movement_patterns['active_periods']) > 10:
                active_signal = movement_patterns['active_periods'].astype(float)
                
                # Find overlapping periods and calculate correlation
                min_time = max(activity_times.min(), 0)
                max_time = min(activity_times.max(), len(active_signal))
                
                if max_time > min_time and max_time - min_time > 10:
                    common_times = np.linspace(min_time, max_time, int(max_time - min_time))
                    activity_interp = np.interp(common_times, activity_times, activity_values)
                    movement_interp = np.interp(common_times, range(len(active_signal)), active_signal)
                    
                    activity_movement_corr = np.corrcoef(activity_interp, movement_interp)[0, 1]
                    activity_movement_sig = stats.pearsonr(activity_interp, movement_interp)[1]
                    
                    coupling_results['activity_movement'] = {
                        'correlation': abs(activity_movement_corr),
                        'significance': activity_movement_sig,
                        'expected': 'positive correlation between biometric activity and movement'
                    }
        
        return coupling_results
    
    def _calculate_validation_metrics(self, movement_coupling: Dict, validation_results: Dict) -> Dict:
        """Calculate overall validation metrics for coupling strength"""
        
        # Extract correlation values
        correlations = []
        significances = []
        
        for coupling_type, results in movement_coupling.items():
            if 'correlation' in results:
                correlations.append(results['correlation'])
            if 'significance' in results:
                significances.append(results['significance'])
        
        # Calculate overall metrics
        if correlations:
            mean_correlation = np.mean(correlations)
            max_correlation = np.max(correlations)
            correlation_consistency = 1 - np.std(correlations) / np.mean(correlations) if np.mean(correlations) > 0 else 0
        else:
            mean_correlation = max_correlation = correlation_consistency = 0
        
        if significances:
            significant_couplings = sum(1 for p in significances if p < self.validation_params['significance_threshold'])
            significance_ratio = significant_couplings / len(significances)
        else:
            significance_ratio = 0
        
        # Validation score (0-1 scale)
        validation_score = 0
        
        # Score based on correlation strength
        if mean_correlation >= self.validation_params['correlation_threshold']:
            validation_score += 0.4 * (mean_correlation / 1.0)  # Max possible correlation is 1
        
        # Score based on statistical significance
        validation_score += 0.3 * significance_ratio
        
        # Score based on consistency across different coupling types
        validation_score += 0.2 * correlation_consistency
        
        # Score based on data duration adequacy
        min_hours = self.validation_params['min_data_hours']
        actual_hours = validation_results['data_duration_hours']
        duration_score = min(1.0, actual_hours / min_hours)
        validation_score += 0.1 * duration_score
        
        # Classification of coupling strength
        if validation_score >= 0.8:
            coupling_classification = 'Strong Coupling'
        elif validation_score >= 0.6:
            coupling_classification = 'Moderate Coupling'
        elif validation_score >= 0.4:
            coupling_classification = 'Weak Coupling'
        else:
            coupling_classification = 'No Significant Coupling'
        
        return {
            'validation_score': validation_score,
            'coupling_classification': coupling_classification,
            'mean_correlation': mean_correlation,
            'max_correlation': max_correlation,
            'correlation_consistency': correlation_consistency,
            'significance_ratio': significance_ratio,
            'n_coupling_types': len(movement_coupling),
            'data_duration_adequate': actual_hours >= min_hours
        }
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """Generate comprehensive validation report"""
        
        bio_rhythm = validation_results['biological_rhythm']
        rhythm_freq = validation_results['rhythm_frequency']
        duration = validation_results['data_duration_hours']
        metrics = validation_results['validation_metrics']
        movement_coupling = validation_results['movement_coupling']
        
        report = f"""
=== SPATIOTEMPORAL COUPLING VALIDATION REPORT ===

Biological Rhythm Analysis: {bio_rhythm.upper()}
- Rhythm frequency: {rhythm_freq:.6f} Hz
- Rhythm period: {1/rhythm_freq/3600:.2f} hours
- Data duration: {duration:.2f} hours ({duration/24:.1f} days)
- Data adequacy: {'ADEQUATE' if metrics['data_duration_adequate'] else 'INSUFFICIENT'}

Movement Pattern Coupling Analysis:
"""
        
        for coupling_type, results in movement_coupling.items():
            report += f"\n{coupling_type.replace('_', ' ').title()}:\n"
            if 'correlation' in results:
                report += f"  - Correlation strength: {results['correlation']:.4f}\n"
            if 'significance' in results:
                significance = results['significance']
                sig_level = "***" if significance < 0.001 else "**" if significance < 0.01 else "*" if significance < 0.05 else "ns"
                report += f"  - Statistical significance: p = {significance:.4f} {sig_level}\n"
            if 'lag_hours' in results:
                report += f"  - Time lag: {results['lag_hours']:.2f} hours\n"
            if 'phase_locking' in results:
                report += f"  - Phase locking: {results['phase_locking']:.4f}\n"
            if 'expected_phase' in results:
                report += f"  - Expected pattern: {results['expected_phase']}\n"
        
        if 'biometric_coupling' in validation_results:
            report += f"\nBiometric Coupling Analysis:\n"
            for coupling_type, results in validation_results['biometric_coupling'].items():
                report += f"\n{coupling_type.replace('_', ' ').title()}:\n"
                report += f"  - Correlation strength: {results['correlation']:.4f}\n"
                report += f"  - Statistical significance: p = {results['significance']:.4f}\n"
                if 'expected' in results:
                    report += f"  - Expected pattern: {results['expected']}\n"
        
        report += f"""
Validation Metrics:
- Overall validation score: {metrics['validation_score']:.3f}/1.000
- Coupling classification: {metrics['coupling_classification']}
- Mean correlation across coupling types: {metrics['mean_correlation']:.4f}
- Maximum observed correlation: {metrics['max_correlation']:.4f}
- Correlation consistency: {metrics['correlation_consistency']:.4f}
- Statistically significant couplings: {metrics['significance_ratio']:.1%}
- Number of coupling types analyzed: {metrics['n_coupling_types']}

Validation Summary:
"""
        
        if metrics['validation_score'] >= 0.8:
            report += "✓ STRONG EVIDENCE for biological rhythm coupling with movement patterns\n"
            report += "✓ Multiple independent coupling mechanisms identified\n"
            report += "✓ Statistical significance achieved across coupling types\n"
        elif metrics['validation_score'] >= 0.6:
            report += "✓ MODERATE EVIDENCE for biological rhythm coupling\n"
            report += "~ Some coupling mechanisms show significant relationships\n"
            report += "~ Additional data may strengthen validation\n"
        elif metrics['validation_score'] >= 0.4:
            report += "~ WEAK EVIDENCE for biological rhythm coupling\n"
            report += "~ Limited coupling mechanisms identified\n"
            report += "~ Longer data collection period recommended\n"
        else:
            report += "✗ NO SIGNIFICANT EVIDENCE for biological rhythm coupling\n"
            report += "✗ Observed patterns may be due to chance\n"
            report += "✗ Different analysis approach may be needed\n"
        
        return report
    
    def plot_validation_results(self, validation_results: Dict, data: Dict,
                              save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of validation results"""
        
        gps_df = data['gps']
        bio_rhythm = validation_results['biological_rhythm']
        rhythm_freq = validation_results['rhythm_frequency']
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Spatiotemporal Coupling Validation: {bio_rhythm.title()} Rhythm', fontsize=16)
        
        time_seconds = (gps_df['timestamp'] - gps_df['timestamp'].iloc[0]).dt.total_seconds()
        time_hours = time_seconds / 3600
        
        # Plot 1: GPS trajectory
        axes[0, 0].scatter(gps_df['longitude'], gps_df['latitude'], 
                          c=time_hours, cmap='viridis', alpha=0.6, s=10)
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].set_title('GPS Trajectory (colored by time)')
        
        # Plot 2: Movement velocity over time
        movement_patterns = self.extract_movement_patterns(gps_df)
        if len(movement_patterns['velocities']) > 10:
            vel_times = time_hours[1:len(movement_patterns['velocities'])+1]
            axes[0, 1].plot(vel_times, movement_patterns['velocities'], alpha=0.7)
            axes[0, 1].set_xlabel('Time (hours)')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            axes[0, 1].set_title('Movement Velocity Over Time')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Biological rhythm vs movement patterns
        bio_signal = np.sin(2 * np.pi * rhythm_freq * time_seconds)
        
        axes[1, 0].plot(time_hours, bio_signal, 'k-', linewidth=2, alpha=0.8, label=f'{bio_rhythm.title()} Rhythm')
        
        # Overlay velocity if available
        if len(movement_patterns['velocities']) > 10:
            vel_normalized = movement_patterns['velocities'] / np.max(movement_patterns['velocities'])
            vel_times = time_hours[1:len(vel_normalized)+1]
            axes[1, 0].plot(vel_times, vel_normalized, '--', alpha=0.6, label='Velocity (normalized)')
        
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Normalized Amplitude')
        axes[1, 0].set_title('Biological Rhythm vs Movement Patterns')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Location clustering
        cluster_labels = movement_patterns['cluster_labels']
        cluster_centers = movement_patterns['cluster_centers']
        
        scatter = axes[1, 1].scatter(gps_df['longitude'], gps_df['latitude'], 
                                   c=cluster_labels, cmap='tab10', alpha=0.6, s=15)
        axes[1, 1].scatter(cluster_centers[:, 1], cluster_centers[:, 0], 
                          marker='x', s=100, c='red', linewidths=2)
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        axes[1, 1].set_title(f'Location Clustering ({movement_patterns["n_significant_places"]} places)')
        
        # Plot 5: Coupling strength visualization
        movement_coupling = validation_results['movement_coupling']
        coupling_types = list(movement_coupling.keys())
        coupling_strengths = [movement_coupling[ct].get('correlation', 0) for ct in coupling_types]
        
        bars = axes[2, 0].bar(range(len(coupling_types)), coupling_strengths, alpha=0.7)
        axes[2, 0].set_xticks(range(len(coupling_types)))
        axes[2, 0].set_xticklabels([ct.replace('_', '\n').title() for ct in coupling_types], rotation=45)
        axes[2, 0].set_ylabel('Correlation Strength')
        axes[2, 0].set_title('Movement-Biological Coupling Strength')
        axes[2, 0].axhline(y=self.validation_params['correlation_threshold'], 
                          color='red', linestyle='--', alpha=0.5, label='Significance Threshold')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for bar, strength in zip(bars, coupling_strengths):
            if strength > 0:
                axes[2, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                               f'{strength:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 6: Validation metrics summary
        metrics = validation_results['validation_metrics']
        metric_names = ['Validation\nScore', 'Mean\nCorrelation', 'Max\nCorrelation', 
                       'Correlation\nConsistency', 'Significance\nRatio']
        metric_values = [metrics['validation_score'], metrics['mean_correlation'], 
                        metrics['max_correlation'], metrics['correlation_consistency'],
                        metrics['significance_ratio']]
        
        bars = axes[2, 1].bar(metric_names, metric_values, alpha=0.7, color='purple')
        axes[2, 1].set_ylabel('Metric Value')
        axes[2, 1].set_title(f'Validation Summary\n{metrics["coupling_classification"]}')
        axes[2, 1].set_ylim(0, 1)
        axes[2, 1].grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, value in zip(bars, metric_values):
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation plot saved to {save_path}")
        
        plt.show()

def main():
    """Demonstration of spatiotemporal coupling validation"""
    
    logger.info("Initializing Spatiotemporal Coupling Validator...")
    
    validator = SpatiotemporalCouplingValidator()
    
    # Generate sample data for demonstration
    np.random.seed(42)
    
    # Create 3 days of GPS data with realistic movement patterns
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    n_points = 3 * 24 * 60  # 3 days, 1 point per minute
    
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
    time_seconds = np.array([(ts - start_time).total_seconds() for ts in timestamps])
    
    # Base location (New York)
    base_lat = 40.7128
    base_lon = -74.0060
    
    # Add realistic circadian movement patterns
    circadian_freq = validator.biological_frequencies['circadian']
    
    # Home-work commute pattern
    commute_amplitude_lat = 0.02  # ~2 km commute distance
    commute_amplitude_lon = 0.02
    
    # Daily activity pattern (stronger during day, weaker at night)
    activity_pattern = 0.5 + 0.5 * np.sin(2 * np.pi * circadian_freq * time_seconds + np.pi/2)
    
    # Generate realistic GPS coordinates
    lat = base_lat + commute_amplitude_lat * np.sin(2 * np.pi * circadian_freq * time_seconds) * activity_pattern
    lat += 0.005 * np.sin(2 * np.pi * validator.biological_frequencies['ultradian_90min'] * time_seconds)
    lat += 0.002 * np.random.normal(size=n_points)
    
    lon = base_lon + commute_amplitude_lon * np.cos(2 * np.pi * circadian_freq * time_seconds) * activity_pattern
    lon += 0.008 * np.cos(2 * np.pi * validator.biological_frequencies['ultradian_90min'] * time_seconds)
    lon += 0.002 * np.random.normal(size=n_points)
    
    # Create sample GPS dataframe
    gps_df = pd.DataFrame({
        'timestamp': timestamps,
        'latitude': lat,
        'longitude': lon
    })
    
    # Create sample biometric data
    heart_rates = 70 + 20 * activity_pattern + 10 * np.random.normal(size=n_points)
    heart_rates = np.clip(heart_rates, 50, 150)  # Realistic HR range
    
    activity_levels = activity_pattern + 0.2 * np.random.normal(size=n_points)
    activity_levels = np.clip(activity_levels, 0, 1)
    
    biometric_df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': heart_rates,
        'activity_level': activity_levels
    })
    
    # Package data
    data = {
        'gps': gps_df,
        'biometric': biometric_df,
        'activity': None
    }
    
    print("=== SPATIOTEMPORAL COUPLING VALIDATION DEMONSTRATION ===\n")
    
    # Perform validation
    logger.info("Performing circadian rhythm validation...")
    validation_results = validator.validate_biological_rhythm_coupling(data, bio_rhythm='circadian')
    
    # Generate report
    report = validator.generate_validation_report(validation_results)
    print(report)
    
    # Create visualization
    validator.plot_validation_results(validation_results, data)
    
    logger.info("Spatiotemporal coupling validation complete!")

if __name__ == "__main__":
    main()
