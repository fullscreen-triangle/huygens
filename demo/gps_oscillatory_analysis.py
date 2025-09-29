"""
GPS Oscillatory Analysis Module
Analyzes GPS coordinate data for spatial-temporal oscillatory patterns
Part of the Technological-Biological Meta-Oscillatory Coupling Framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPSOscillatoryAnalyzer:
    """
    Analyzes GPS coordinate data for oscillatory patterns and coupling with biological rhythms
    """
    
    def __init__(self):
        # Fundamental oscillatory frequencies (Hz)
        self.fundamental_frequencies = {
            'circadian': 1.157e-5,  # 24 hour period
            'ultradian': 1.667e-4,   # 1.5 hour period
            'weekly': 1.653e-6,      # 7 day period
            'daily_commute': 2.314e-5,  # 12 hour period
            'cardiac_avg': 1.17,      # Average heart rate ~70 BPM
            'gps_l1': 1.57542e9,     # GPS L1 carrier frequency
            'atomic_clock': 9.192631770e9  # Cesium atomic clock
        }
        
        self.harmonic_ratios = {}
        self._calculate_harmonic_ratios()
        
    def _calculate_harmonic_ratios(self):
        """Calculate harmonic ratios between technological and biological frequencies"""
        bio_freqs = ['circadian', 'ultradian', 'weekly', 'cardiac_avg']
        tech_freqs = ['gps_l1', 'atomic_clock']
        
        for tech in tech_freqs:
            for bio in bio_freqs:
                ratio = self.fundamental_frequencies[tech] / self.fundamental_frequencies[bio]
                self.harmonic_ratios[f"{tech}_{bio}"] = ratio
                logger.info(f"Harmonic ratio {tech}-{bio}: {ratio:.2e}")
    
    def load_gps_data(self, data_path: str) -> pd.DataFrame:
        """Load GPS coordinate data from file"""
        try:
            if data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            else:
                raise ValueError("Unsupported file format. Use JSON or CSV.")
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'latitude', 'longitude']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Data must contain columns: {required_columns}")
            
            # Convert timestamps to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate time deltas in seconds
            df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
            
            logger.info(f"Loaded {len(df)} GPS data points spanning {df['time_seconds'].iloc[-1]:.2f} seconds")
            return df
            
        except Exception as e:
            logger.error(f"Error loading GPS data: {e}")
            raise
    
    def calculate_movement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate movement-related metrics from GPS coordinates"""
        # Convert to numpy arrays for calculation
        lat = df['latitude'].values
        lon = df['longitude'].values
        t = df['time_seconds'].values
        
        # Calculate approximate distances (assuming flat earth for small distances)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Earth radius in meters
        R = 6371000
        
        # Calculate displacements
        dlat = np.diff(lat_rad)
        dlon = np.diff(lon_rad)
        
        # Haversine formula approximation for small distances
        dx = R * dlon * np.cos(lat_rad[:-1])  # East-West displacement
        dy = R * dlat                         # North-South displacement
        dt = np.diff(t)
        
        # Calculate velocities
        vx = dx / dt  # East-West velocity (m/s)
        vy = dy / dt  # North-South velocity (m/s)
        speed = np.sqrt(vx**2 + vy**2)  # Total speed (m/s)
        
        # Create results dataframe
        result_df = df.iloc[1:].copy()  # Remove first point since we calculate differences
        result_df['dx'] = dx
        result_df['dy'] = dy
        result_df['dt'] = dt
        result_df['velocity_x'] = vx
        result_df['velocity_y'] = vy
        result_df['speed'] = speed
        result_df['time_interval'] = result_df['time_seconds'] - result_df['time_seconds'].iloc[0]
        
        logger.info(f"Calculated movement metrics: mean speed = {speed.mean():.2f} m/s")
        return result_df
    
    def perform_fft_analysis(self, signal_data: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform FFT analysis on signal data"""
        # Remove DC component and apply windowing
        signal_centered = signal_data - np.mean(signal_data)
        windowed_signal = signal_centered * signal.windows.hann(len(signal_centered))
        
        # Compute FFT
        fft_result = fft(windowed_signal)
        frequencies = fftfreq(len(windowed_signal), 1/sampling_rate)
        
        # Take positive frequencies only
        positive_freq_mask = frequencies > 0
        frequencies = frequencies[positive_freq_mask]
        power_spectrum = np.abs(fft_result[positive_freq_mask])**2
        
        return frequencies, power_spectrum
    
    def detect_oscillatory_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect oscillatory patterns in GPS movement data"""
        results = {}
        
        # Calculate sampling rate
        mean_dt = df['dt'].mean()
        sampling_rate = 1 / mean_dt
        logger.info(f"Mean sampling rate: {sampling_rate:.4f} Hz")
        
        # Analyze different signals
        signals_to_analyze = {
            'latitude': df['latitude'].values,
            'longitude': df['longitude'].values,
            'speed': df['speed'].values,
            'velocity_x': df['velocity_x'].values,
            'velocity_y': df['velocity_y'].values
        }
        
        for signal_name, signal_data in signals_to_analyze.items():
            if len(signal_data) < 10:  # Need minimum data points
                continue
                
            try:
                # Perform FFT analysis
                frequencies, power_spectrum = self.perform_fft_analysis(signal_data, sampling_rate)
                
                # Find peaks in power spectrum
                peak_indices, _ = signal.find_peaks(power_spectrum, 
                                                  height=np.max(power_spectrum) * 0.1,
                                                  distance=int(len(power_spectrum) * 0.05))
                
                peak_frequencies = frequencies[peak_indices]
                peak_powers = power_spectrum[peak_indices]
                
                # Calculate periods
                peak_periods = 1 / peak_frequencies
                
                # Identify biological frequency matches
                bio_matches = self._find_biological_matches(peak_frequencies)
                
                results[signal_name] = {
                    'frequencies': frequencies,
                    'power_spectrum': power_spectrum,
                    'peak_frequencies': peak_frequencies,
                    'peak_periods': peak_periods,
                    'peak_powers': peak_powers,
                    'biological_matches': bio_matches,
                    'total_power': np.sum(power_spectrum),
                    'dominant_frequency': frequencies[np.argmax(power_spectrum)],
                    'dominant_period': 1 / frequencies[np.argmax(power_spectrum)]
                }
                
                logger.info(f"{signal_name}: Found {len(peak_frequencies)} oscillatory peaks")
                
            except Exception as e:
                logger.error(f"Error analyzing {signal_name}: {e}")
                results[signal_name] = None
        
        return results
    
    def _find_biological_matches(self, frequencies: np.ndarray, tolerance: float = 0.1) -> List[Dict]:
        """Find matches between detected frequencies and known biological oscillations"""
        matches = []
        
        for freq in frequencies:
            for bio_name, bio_freq in self.fundamental_frequencies.items():
                if 'gps' not in bio_name and 'atomic' not in bio_name:  # Only biological frequencies
                    # Check direct frequency match
                    if abs(freq - bio_freq) / bio_freq < tolerance:
                        matches.append({
                            'detected_freq': freq,
                            'biological_freq': bio_freq,
                            'biological_type': bio_name,
                            'match_type': 'direct',
                            'error': abs(freq - bio_freq) / bio_freq
                        })
                    
                    # Check harmonic matches (up to 10th harmonic)
                    for n in range(2, 11):
                        harmonic_freq = bio_freq * n
                        if abs(freq - harmonic_freq) / harmonic_freq < tolerance:
                            matches.append({
                                'detected_freq': freq,
                                'biological_freq': bio_freq,
                                'biological_type': bio_name,
                                'match_type': f'{n}_harmonic',
                                'error': abs(freq - harmonic_freq) / harmonic_freq
                            })
                        
                        # Check sub-harmonic matches
                        subharmonic_freq = bio_freq / n
                        if abs(freq - subharmonic_freq) / subharmonic_freq < tolerance:
                            matches.append({
                                'detected_freq': freq,
                                'biological_freq': bio_freq,
                                'biological_type': bio_name,
                                'match_type': f'1/{n}_harmonic',
                                'error': abs(freq - subharmonic_freq) / subharmonic_freq
                            })
        
        return matches
    
    def calculate_coupling_strength(self, df: pd.DataFrame, oscillatory_results: Dict) -> Dict:
        """Calculate coupling strength between GPS patterns and biological rhythms"""
        coupling_results = {}
        
        # Extract time series for cross-correlation analysis
        time_seconds = df['time_interval'].values
        
        for signal_name, results in oscillatory_results.items():
            if results is None:
                continue
                
            coupling_results[signal_name] = {}
            
            # Generate synthetic biological oscillations for comparison
            for bio_name, bio_freq in self.fundamental_frequencies.items():
                if 'gps' in bio_name or 'atomic' in bio_name:
                    continue
                    
                # Generate synthetic biological signal
                bio_signal = np.sin(2 * np.pi * bio_freq * time_seconds)
                
                # Get GPS signal (using original data, not FFT)
                if signal_name == 'latitude':
                    gps_signal = df['latitude'].iloc[1:].values
                elif signal_name == 'longitude':
                    gps_signal = df['longitude'].iloc[1:].values
                elif signal_name == 'speed':
                    gps_signal = df['speed'].values
                elif signal_name == 'velocity_x':
                    gps_signal = df['velocity_x'].values
                elif signal_name == 'velocity_y':
                    gps_signal = df['velocity_y'].values
                
                # Normalize signals
                bio_signal_norm = (bio_signal - np.mean(bio_signal)) / np.std(bio_signal)
                gps_signal_norm = (gps_signal - np.mean(gps_signal)) / np.std(gps_signal)
                
                # Calculate cross-correlation
                correlation = signal.correlate(gps_signal_norm, bio_signal_norm, mode='full')
                correlation = correlation / len(bio_signal_norm)  # Normalize
                
                # Find maximum correlation
                max_corr = np.max(np.abs(correlation))
                max_lag_idx = np.argmax(np.abs(correlation))
                max_lag = max_lag_idx - len(bio_signal_norm) + 1
                
                coupling_results[signal_name][bio_name] = {
                    'max_correlation': max_corr,
                    'lag_samples': max_lag,
                    'lag_seconds': max_lag * df['dt'].mean(),
                    'phase_coupling': np.mean(np.cos(np.angle(fft(gps_signal_norm)) - np.angle(fft(bio_signal_norm))))
                }
        
        return coupling_results
    
    def calculate_spatiotemporal_coherence(self, df: pd.DataFrame) -> Dict:
        """Calculate spatiotemporal coherence metrics"""
        lat = df['latitude'].values
        lon = df['longitude'].values
        t = df['time_seconds'].values
        
        # Calculate spatial coherence (consistency of movement patterns)
        lat_var = np.var(lat)
        lon_var = np.var(lon)
        
        # Calculate velocity coherence
        if len(df) > 1:
            speed = df['speed'].values
            speed_coherence = 1 / (1 + np.var(speed) / np.mean(speed)**2) if np.mean(speed) > 0 else 0
        else:
            speed_coherence = 0
        
        # Calculate temporal regularity
        if len(t) > 2:
            dt_array = np.diff(t)
            temporal_regularity = 1 / (1 + np.var(dt_array) / np.mean(dt_array)**2)
        else:
            temporal_regularity = 0
        
        # Calculate location clustering coherence
        location_points = np.column_stack([lat, lon])
        if len(location_points) > 1:
            # Calculate pairwise distances
            distances = []
            for i in range(len(location_points)):
                for j in range(i + 1, len(location_points)):
                    dist = np.sqrt((location_points[i][0] - location_points[j][0])**2 + 
                                 (location_points[i][1] - location_points[j][1])**2)
                    distances.append(dist)
            
            # Clustering coherence based on distance distribution
            if distances:
                mean_dist = np.mean(distances)
                dist_var = np.var(distances)
                clustering_coherence = mean_dist / (mean_dist + dist_var) if mean_dist + dist_var > 0 else 0
            else:
                clustering_coherence = 0
        else:
            clustering_coherence = 0
        
        return {
            'spatial_variance': {'latitude': lat_var, 'longitude': lon_var},
            'speed_coherence': speed_coherence,
            'temporal_regularity': temporal_regularity,
            'clustering_coherence': clustering_coherence,
            'overall_coherence': np.mean([speed_coherence, temporal_regularity, clustering_coherence])
        }
    
    def generate_analysis_report(self, df: pd.DataFrame, oscillatory_results: Dict, 
                               coupling_results: Dict, coherence_results: Dict) -> str:
        """Generate comprehensive analysis report"""
        report = """
=== GPS OSCILLATORY ANALYSIS REPORT ===

Dataset Information:
- Total data points: {total_points}
- Time span: {time_span:.2f} seconds ({time_span_hours:.2f} hours)
- Mean sampling interval: {mean_dt:.2f} seconds
- Mean movement speed: {mean_speed:.2f} m/s

Oscillatory Pattern Detection:
""".format(
            total_points=len(df),
            time_span=df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0],
            time_span_hours=(df['time_seconds'].iloc[-1] - df['time_seconds'].iloc[0]) / 3600,
            mean_dt=df['dt'].mean() if 'dt' in df.columns else 0,
            mean_speed=df['speed'].mean() if 'speed' in df.columns else 0
        )
        
        for signal_name, results in oscillatory_results.items():
            if results is None:
                continue
                
            report += f"\n{signal_name.upper()} Signal Analysis:\n"
            report += f"  - Dominant frequency: {results['dominant_frequency']:.6f} Hz\n"
            report += f"  - Dominant period: {results['dominant_period']:.2f} seconds ({results['dominant_period']/3600:.2f} hours)\n"
            report += f"  - Peak frequencies found: {len(results['peak_frequencies'])}\n"
            
            if results['biological_matches']:
                report += f"  - Biological frequency matches: {len(results['biological_matches'])}\n"
                for match in results['biological_matches'][:3]:  # Show top 3 matches
                    report += f"    * {match['biological_type']} ({match['match_type']}): error = {match['error']:.4f}\n"
        
        report += f"\nCoupling Strength Analysis:\n"
        for signal_name, couplings in coupling_results.items():
            report += f"\n{signal_name.upper()} Biological Coupling:\n"
            for bio_name, coupling in couplings.items():
                report += f"  - {bio_name}: correlation = {coupling['max_correlation']:.4f}, lag = {coupling['lag_seconds']:.2f}s\n"
        
        report += f"\nSpatiotemporal Coherence Metrics:\n"
        report += f"  - Speed coherence: {coherence_results['speed_coherence']:.4f}\n"
        report += f"  - Temporal regularity: {coherence_results['temporal_regularity']:.4f}\n"
        report += f"  - Clustering coherence: {coherence_results['clustering_coherence']:.4f}\n"
        report += f"  - Overall coherence: {coherence_results['overall_coherence']:.4f}\n"
        
        report += f"\nTechnological-Biological Harmonic Relationships:\n"
        for ratio_name, ratio_value in list(self.harmonic_ratios.items())[:5]:
            report += f"  - {ratio_name}: {ratio_value:.2e}\n"
        
        return report
    
    def plot_oscillatory_analysis(self, df: pd.DataFrame, oscillatory_results: Dict, 
                                 save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of oscillatory analysis"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('GPS Oscillatory Pattern Analysis', fontsize=16)
        
        # Plot 1: GPS trajectory
        axes[0, 0].scatter(df['longitude'], df['latitude'], c=df['time_seconds'], 
                          cmap='viridis', alpha=0.6, s=10)
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].set_title('GPS Trajectory (colored by time)')
        
        # Plot 2: Speed over time
        if 'speed' in df.columns:
            axes[0, 1].plot(df['time_seconds']/3600, df['speed'], alpha=0.7)
            axes[0, 1].set_xlabel('Time (hours)')
            axes[0, 1].set_ylabel('Speed (m/s)')
            axes[0, 1].set_title('Movement Speed Over Time')
        
        # Plot 3: Latitude oscillations and FFT
        if 'latitude' in oscillatory_results and oscillatory_results['latitude']:
            result = oscillatory_results['latitude']
            axes[1, 0].plot(df['time_seconds']/3600, df['latitude'], alpha=0.7)
            axes[1, 0].set_xlabel('Time (hours)')
            axes[1, 0].set_ylabel('Latitude')
            axes[1, 0].set_title('Latitude Oscillations')
            
            # FFT of latitude
            axes[1, 1].loglog(result['frequencies'], result['power_spectrum'])
            if len(result['peak_frequencies']) > 0:
                axes[1, 1].scatter(result['peak_frequencies'], 
                                 result['peak_powers'], 
                                 color='red', s=30, zorder=5)
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('Power Spectral Density')
            axes[1, 1].set_title('Latitude Frequency Spectrum')
        
        # Plot 4: Speed oscillations and FFT
        if 'speed' in oscillatory_results and oscillatory_results['speed']:
            result = oscillatory_results['speed']
            
            # FFT of speed
            axes[2, 0].loglog(result['frequencies'], result['power_spectrum'])
            if len(result['peak_frequencies']) > 0:
                axes[2, 0].scatter(result['peak_frequencies'], 
                                 result['peak_powers'], 
                                 color='red', s=30, zorder=5)
            axes[2, 0].set_xlabel('Frequency (Hz)')
            axes[2, 0].set_ylabel('Power Spectral Density')
            axes[2, 0].set_title('Speed Frequency Spectrum')
        
        # Plot 5: Biological frequency comparison
        axes[2, 1].axvline(self.fundamental_frequencies['circadian'], color='blue', 
                          linestyle='--', alpha=0.7, label='Circadian (24h)')
        axes[2, 1].axvline(self.fundamental_frequencies['ultradian'], color='green', 
                          linestyle='--', alpha=0.7, label='Ultradian (1.5h)')
        axes[2, 1].axvline(self.fundamental_frequencies['cardiac_avg'], color='red', 
                          linestyle='--', alpha=0.7, label='Cardiac (~70 BPM)')
        
        # Overlay detected peaks from speed analysis
        if 'speed' in oscillatory_results and oscillatory_results['speed']:
            result = oscillatory_results['speed']
            axes[2, 1].loglog(result['frequencies'], result['power_spectrum'], alpha=0.5)
        
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].set_ylabel('Power / Biological Reference')
        axes[2, 1].set_title('Biological Frequency Comparison')
        axes[2, 1].legend()
        axes[2, 1].set_xlim(1e-6, 10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()

def main():
    """Example usage of GPS Oscillatory Analyzer"""
    analyzer = GPSOscillatoryAnalyzer()
    
    # Generate sample GPS data for demonstration
    np.random.seed(42)
    n_points = 1000
    time_hours = 24  # 24 hours of data
    
    t = np.linspace(0, time_hours * 3600, n_points)  # Time in seconds
    
    # Base location
    base_lat = 40.7128  # New York latitude
    base_lon = -74.0060  # New York longitude
    
    # Add circadian oscillation to movement
    circadian_freq = analyzer.fundamental_frequencies['circadian']
    commute_freq = analyzer.fundamental_frequencies['daily_commute']
    
    # Simulate realistic movement patterns
    lat = base_lat + 0.01 * np.sin(2 * np.pi * circadian_freq * t) + \
          0.005 * np.sin(2 * np.pi * commute_freq * t) + \
          0.002 * np.random.normal(size=n_points)
    
    lon = base_lon + 0.01 * np.cos(2 * np.pi * circadian_freq * t) + \
          0.008 * np.cos(2 * np.pi * commute_freq * t) + \
          0.002 * np.random.normal(size=n_points)
    
    # Create sample dataframe
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq=f'{time_hours*3600/n_points}s'),
        'latitude': lat,
        'longitude': lon
    })
    
    df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    
    print("=== GPS OSCILLATORY ANALYSIS DEMONSTRATION ===\n")
    
    # Perform analysis
    df_with_movement = analyzer.calculate_movement_metrics(df)
    oscillatory_results = analyzer.detect_oscillatory_patterns(df_with_movement)
    coupling_results = analyzer.calculate_coupling_strength(df_with_movement, oscillatory_results)
    coherence_results = analyzer.calculate_spatiotemporal_coherence(df_with_movement)
    
    # Generate report
    report = analyzer.generate_analysis_report(df_with_movement, oscillatory_results, 
                                             coupling_results, coherence_results)
    print(report)
    
    # Create visualization
    analyzer.plot_oscillatory_analysis(df_with_movement, oscillatory_results)

if __name__ == "__main__":
    main()
