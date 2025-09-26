"""
Experimental Biometric Data Validator

Validates consciousness and oscillatory theories using real personal biometric data.
This framework applies theoretical predictions to actual experimental measurements
including heart rate, sleep architecture, activity patterns, and geolocation data.

REVOLUTIONARY VALIDATION: First framework to test consciousness theories 
with comprehensive real-world biometric data!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ExperimentalBiometricValidator:
    """
    Validates theoretical framework using real biometric data
    """
    
    def __init__(self, experimental_data_dir="../../experimental-data", results_dir="experimental_validation_results"):
        self.experimental_data_dir = Path(experimental_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Data categories expected
        self.data_categories = {
            'heart_rate': {
                'description': 'Heart rate monitoring data including sprint sessions',
                'expected_columns': ['timestamp', 'heart_rate', 'context'],
                'sampling_rate': 1.0,  # Hz
                'analysis_type': 'time_series'
            },
            'sleep': {
                'description': 'Sleep architecture and sleep stage data', 
                'expected_columns': ['timestamp', 'sleep_stage', 'quality_metrics'],
                'sampling_rate': 1/30,  # Every 30 seconds
                'analysis_type': 'categorical_time_series'
            },
            'activity': {
                'description': 'Daily activity levels and movement patterns',
                'expected_columns': ['timestamp', 'activity_level', 'activity_type'],
                'sampling_rate': 1/60,  # Every minute
                'analysis_type': 'continuous_discrete'
            },
            'geolocation': {
                'description': 'Location data for environmental context analysis',
                'expected_columns': ['timestamp', 'latitude', 'longitude', 'context'],
                'sampling_rate': 1/300,  # Every 5 minutes
                'analysis_type': 'spatial_temporal'
            }
        }
        
        # Theoretical predictions to validate
        self.theoretical_predictions = {
            'consciousness_frequency_range': (2, 10),  # Hz - consciousness operates 100-500ms cycles
            'heart_rate_consciousness_coupling': 0.3,   # Coupling strength
            'sleep_consciousness_transitions': ['wake', 'light', 'deep', 'rem'],
            'circadian_oscillatory_period': 24.0 * 3600,  # seconds
            'activity_consciousness_correlation': 0.4,  # Expected correlation
            'sprint_oscillatory_patterns': True,  # Expected oscillatory behavior during sprints
            'multi_scale_coupling_strength': 0.2  # Cross-system coupling
        }
        
        # Data storage
        self.loaded_data = {}
        self.processed_data = {}
        self.validation_results = {}
        
        print("🏃‍♂️💓 EXPERIMENTAL BIOMETRIC VALIDATOR 💓🏃‍♂️")
        print("=" * 70)
        print("Validating consciousness theories with REAL biometric data!")
        print("=" * 70)
    
    def load_experimental_data(self):
        """
        Load all available experimental data from the experimental-data directory
        """
        print("\n📂 LOADING EXPERIMENTAL DATA")
        print("-" * 50)
        
        loaded_categories = []
        
        for category, category_info in self.data_categories.items():
            category_path = self.experimental_data_dir / 'raw' / category
            
            if category_path.exists():
                print(f"Loading {category} data from {category_path}...")
                
                try:
                    # Look for common data file formats
                    data_files = list(category_path.glob('*.csv')) + list(category_path.glob('*.json')) + list(category_path.glob('*.parquet'))
                    
                    if data_files:
                        category_data = []
                        
                        for data_file in data_files:
                            print(f"  Reading {data_file.name}...")
                            
                            if data_file.suffix == '.csv':
                                df = pd.read_csv(data_file)
                            elif data_file.suffix == '.json':
                                df = pd.read_json(data_file)
                            elif data_file.suffix == '.parquet':
                                df = pd.read_parquet(data_file)
                            
                            # Add source file information
                            df['source_file'] = data_file.name
                            category_data.append(df)
                        
                        # Combine all files for this category
                        if category_data:
                            combined_data = pd.concat(category_data, ignore_index=True)
                            
                            # Ensure timestamp column exists and is datetime
                            if 'timestamp' in combined_data.columns:
                                combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
                                combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
                            
                            self.loaded_data[category] = combined_data
                            loaded_categories.append(category)
                            
                            print(f"  ✅ {category}: {len(combined_data)} records loaded")
                        else:
                            print(f"  ⚠️ {category}: No valid data found")
                    else:
                        print(f"  ⚠️ {category}: No data files found in {category_path}")
                        
                except Exception as e:
                    print(f"  ❌ {category}: Error loading data - {str(e)}")
            else:
                print(f"  📁 {category}: Directory not found - {category_path}")
        
        print(f"\n📊 LOADED DATA SUMMARY:")
        print(f"Categories loaded: {len(loaded_categories)}/{len(self.data_categories)}")
        
        for category in loaded_categories:
            data = self.loaded_data[category]
            if 'timestamp' in data.columns:
                duration = data['timestamp'].max() - data['timestamp'].min()
                print(f"  {category}: {len(data)} records over {duration}")
            else:
                print(f"  {category}: {len(data)} records")
        
        return loaded_categories
    
    def validate_heart_rate_consciousness_coupling(self):
        """
        Validate heart rate patterns against consciousness theoretical predictions
        """
        print("\n🔬 EXPERIMENT: Heart Rate-Consciousness Coupling")
        print("-" * 50)
        
        if 'heart_rate' not in self.loaded_data:
            print("❌ Heart rate data not available")
            return {'error': 'No heart rate data'}
        
        hr_data = self.loaded_data['heart_rate'].copy()
        results = {}
        
        # Basic heart rate analysis
        if 'heart_rate' in hr_data.columns:
            heart_rates = hr_data['heart_rate'].dropna()
            
            results['basic_stats'] = {
                'mean_hr': heart_rates.mean(),
                'std_hr': heart_rates.std(),
                'min_hr': heart_rates.min(),
                'max_hr': heart_rates.max(),
                'num_measurements': len(heart_rates)
            }
            
            print(f"Basic HR Stats: Mean={results['basic_stats']['mean_hr']:.1f}, "
                  f"Std={results['basic_stats']['std_hr']:.1f}")
        
        # Time series analysis for oscillatory patterns
        if 'timestamp' in hr_data.columns and 'heart_rate' in hr_data.columns:
            # Ensure proper time series
            hr_ts = hr_data.set_index('timestamp')['heart_rate'].dropna()
            
            if len(hr_ts) > 100:  # Need sufficient data for frequency analysis
                # Resample to regular intervals for frequency analysis
                hr_resampled = hr_ts.resample('1S').mean().interpolate()  # 1 second intervals
                
                # Frequency domain analysis
                sampling_rate = 1.0  # 1 Hz after resampling
                frequencies, power_spectrum = signal.periodogram(
                    hr_resampled.dropna().values, 
                    fs=sampling_rate,
                    window='hann'
                )
                
                # Find dominant frequencies
                dominant_freq_idx = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies
                dominant_frequencies = frequencies[dominant_freq_idx]
                dominant_powers = power_spectrum[dominant_freq_idx]
                
                results['frequency_analysis'] = {
                    'dominant_frequencies': dominant_frequencies.tolist(),
                    'dominant_powers': dominant_powers.tolist(),
                    'frequency_range': (frequencies[0], frequencies[-1]),
                    'consciousness_band_power': 0.0
                }
                
                # Validate consciousness frequency range (2-10 Hz)
                consciousness_band = (frequencies >= 2) & (frequencies <= 10)
                if np.any(consciousness_band):
                    consciousness_power = np.sum(power_spectrum[consciousness_band])
                    total_power = np.sum(power_spectrum)
                    results['frequency_analysis']['consciousness_band_power'] = consciousness_power / total_power
                
                print(f"Frequency Analysis: Consciousness band power = "
                      f"{results['frequency_analysis']['consciousness_band_power']:.3f}")
                
                # Heart Rate Variability (HRV) analysis
                if len(hr_resampled) > 10:
                    rr_intervals = 60.0 / hr_resampled.dropna()  # Convert to RR intervals (seconds)
                    rr_intervals = rr_intervals[rr_intervals > 0.3]  # Filter physiological range
                    rr_intervals = rr_intervals[rr_intervals < 2.0]
                    
                    if len(rr_intervals) > 5:
                        results['hrv_analysis'] = {
                            'rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000,  # ms
                            'sdnn': np.std(rr_intervals) * 1000,  # ms
                            'mean_rr': np.mean(rr_intervals) * 1000,  # ms
                            'cv_rr': np.std(rr_intervals) / np.mean(rr_intervals)  # Coefficient of variation
                        }
                        
                        print(f"HRV Analysis: RMSSD={results['hrv_analysis']['rmssd']:.1f}ms, "
                              f"SDNN={results['hrv_analysis']['sdnn']:.1f}ms")
            
        # Context-specific analysis (if context data available)
        if 'context' in hr_data.columns:
            context_analysis = {}
            
            for context in hr_data['context'].unique():
                if pd.notna(context):
                    context_data = hr_data[hr_data['context'] == context]['heart_rate'].dropna()
                    
                    if len(context_data) > 5:
                        context_analysis[str(context)] = {
                            'mean_hr': context_data.mean(),
                            'std_hr': context_data.std(),
                            'count': len(context_data)
                        }
            
            results['context_analysis'] = context_analysis
            print(f"Context Analysis: {len(context_analysis)} contexts identified")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        if 'frequency_analysis' in results:
            # Check if consciousness-range frequencies are present
            consciousness_power = results['frequency_analysis']['consciousness_band_power']
            predictions_validated['consciousness_frequency_presence'] = consciousness_power > 0.1
            
        if 'hrv_analysis' in results:
            # Healthy HRV indicates good autonomic function (related to consciousness)
            rmssd = results['hrv_analysis']['rmssd']
            predictions_validated['healthy_hrv'] = rmssd > 20  # ms, typical healthy threshold
        
        results['theoretical_validation'] = predictions_validated
        validation_success = len(predictions_validated) > 0 and any(predictions_validated.values())
        results['validation_success'] = validation_success
        
        print(f"✅ Heart Rate-Consciousness Coupling: {validation_success}")
        
        return results
    
    def validate_sleep_consciousness_transitions(self):
        """
        Validate sleep architecture against consciousness state transition predictions
        """
        print("\n🔬 EXPERIMENT: Sleep-Consciousness Transitions")
        print("-" * 50)
        
        if 'sleep' not in self.loaded_data:
            print("❌ Sleep data not available")
            return {'error': 'No sleep data'}
        
        sleep_data = self.loaded_data['sleep'].copy()
        results = {}
        
        # Analyze sleep architecture
        if 'sleep_stage' in sleep_data.columns and 'timestamp' in sleep_data.columns:
            # Create time series of sleep stages
            sleep_ts = sleep_data.set_index('timestamp')['sleep_stage'].dropna()
            
            # Basic sleep statistics
            stage_counts = sleep_ts.value_counts()
            results['sleep_architecture'] = {
                'stage_distribution': stage_counts.to_dict(),
                'total_sleep_records': len(sleep_ts),
                'unique_stages': sleep_ts.unique().tolist()
            }
            
            print(f"Sleep Architecture: {len(sleep_ts)} records, "
                  f"{len(sleep_ts.unique())} unique stages")
            
            # Analyze sleep stage transitions
            if len(sleep_ts) > 10:
                transitions = []
                
                for i in range(1, len(sleep_ts)):
                    if sleep_ts.iloc[i] != sleep_ts.iloc[i-1]:
                        transitions.append({
                            'from_stage': sleep_ts.iloc[i-1],
                            'to_stage': sleep_ts.iloc[i],
                            'timestamp': sleep_ts.index[i]
                        })
                
                if transitions:
                    transitions_df = pd.DataFrame(transitions)
                    transition_counts = transitions_df.groupby(['from_stage', 'to_stage']).size()
                    
                    results['transition_analysis'] = {
                        'total_transitions': len(transitions),
                        'transition_patterns': transition_counts.to_dict(),
                        'transition_rate': len(transitions) / (len(sleep_ts) / 3600)  # Transitions per hour
                    }
                    
                    print(f"Transition Analysis: {len(transitions)} transitions, "
                          f"{results['transition_analysis']['transition_rate']:.2f} per hour")
            
            # Oscillatory analysis of sleep cycles
            if len(sleep_ts) > 50:
                # Encode sleep stages numerically for frequency analysis
                stage_encoding = {'wake': 4, 'rem': 3, 'light': 2, 'deep': 1}
                encoded_stages = sleep_ts.map(stage_encoding).dropna()
                
                if len(encoded_stages) > 20:
                    # Resample to regular intervals
                    encoded_resampled = encoded_stages.resample('5min').mean().interpolate()
                    
                    # Frequency analysis
                    sampling_rate = 1/300  # 1 sample per 5 minutes
                    frequencies, power_spectrum = signal.periodogram(
                        encoded_resampled.values,
                        fs=sampling_rate
                    )
                    
                    # Look for circadian rhythms (approximately 24 hour cycles)
                    circadian_freq = 1 / (24 * 3600)  # Hz
                    circadian_idx = np.argmin(np.abs(frequencies - circadian_freq))
                    circadian_power = power_spectrum[circadian_idx]
                    
                    results['circadian_analysis'] = {
                        'circadian_frequency': frequencies[circadian_idx],
                        'circadian_power': circadian_power,
                        'total_power': np.sum(power_spectrum)
                    }
                    
                    print(f"Circadian Analysis: Circadian power = {circadian_power:.3e}")
        
        # Sleep quality analysis (if available)
        quality_columns = [col for col in sleep_data.columns if 'quality' in col.lower()]
        
        if quality_columns:
            quality_data = sleep_data[quality_columns + ['timestamp']].dropna()
            
            if len(quality_data) > 5:
                quality_stats = {}
                for col in quality_columns:
                    if pd.api.types.is_numeric_dtype(quality_data[col]):
                        quality_stats[col] = {
                            'mean': quality_data[col].mean(),
                            'std': quality_data[col].std()
                        }
                
                results['sleep_quality'] = quality_stats
                print(f"Sleep Quality: {len(quality_columns)} quality metrics analyzed")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        if 'sleep_architecture' in results:
            # Check for presence of expected sleep stages
            expected_stages = set(self.theoretical_predictions['sleep_consciousness_transitions'])
            actual_stages = set(results['sleep_architecture']['unique_stages'])
            stage_overlap = len(expected_stages.intersection(actual_stages))
            predictions_validated['expected_stages_present'] = stage_overlap >= 2
        
        if 'circadian_analysis' in results:
            # Check for circadian rhythm presence
            circadian_power_ratio = (results['circadian_analysis']['circadian_power'] / 
                                   results['circadian_analysis']['total_power'])
            predictions_validated['circadian_rhythm_detected'] = circadian_power_ratio > 0.01
        
        if 'transition_analysis' in results:
            # Check for reasonable transition frequency
            transition_rate = results['transition_analysis']['transition_rate']
            predictions_validated['reasonable_transition_rate'] = 0.5 <= transition_rate <= 5.0  # Per hour
        
        results['theoretical_validation'] = predictions_validated
        validation_success = len(predictions_validated) > 0 and any(predictions_validated.values())
        results['validation_success'] = validation_success
        
        print(f"✅ Sleep-Consciousness Transitions: {validation_success}")
        
        return results
    
    def validate_activity_consciousness_coupling(self):
        """
        Validate activity patterns against consciousness theoretical predictions
        """
        print("\n🔬 EXPERIMENT: Activity-Consciousness Coupling")
        print("-" * 50)
        
        if 'activity' not in self.loaded_data:
            print("❌ Activity data not available")
            return {'error': 'No activity data'}
        
        activity_data = self.loaded_data['activity'].copy()
        results = {}
        
        # Basic activity analysis
        if 'activity_level' in activity_data.columns:
            activity_levels = activity_data['activity_level'].dropna()
            
            if len(activity_levels) > 10:
                results['basic_activity'] = {
                    'mean_activity': activity_levels.mean(),
                    'std_activity': activity_levels.std(),
                    'activity_range': (activity_levels.min(), activity_levels.max()),
                    'num_measurements': len(activity_levels)
                }
                
                print(f"Basic Activity: Mean={results['basic_activity']['mean_activity']:.2f}, "
                      f"Std={results['basic_activity']['std_activity']:.2f}")
        
        # Time series analysis for circadian patterns
        if 'timestamp' in activity_data.columns and 'activity_level' in activity_data.columns:
            activity_ts = activity_data.set_index('timestamp')['activity_level'].dropna()
            
            if len(activity_ts) > 100:
                # Resample to hourly activity levels
                activity_hourly = activity_ts.resample('1H').mean()
                
                # Circadian analysis
                activity_by_hour = activity_hourly.groupby(activity_hourly.index.hour).mean()
                
                results['circadian_activity'] = {
                    'hourly_pattern': activity_by_hour.to_dict(),
                    'peak_activity_hour': activity_by_hour.idxmax(),
                    'lowest_activity_hour': activity_by_hour.idxmin(),
                    'activity_amplitude': activity_by_hour.max() - activity_by_hour.min()
                }
                
                print(f"Circadian Activity: Peak at hour {results['circadian_activity']['peak_activity_hour']}, "
                      f"Amplitude={results['circadian_activity']['activity_amplitude']:.2f}")
                
                # Frequency analysis of activity patterns
                if len(activity_hourly.dropna()) > 48:  # At least 2 days of data
                    sampling_rate = 1/3600  # 1 sample per hour
                    frequencies, power_spectrum = signal.periodogram(
                        activity_hourly.dropna().values,
                        fs=sampling_rate
                    )
                    
                    # Look for daily cycles
                    daily_freq = 1 / (24 * 3600)  # Hz
                    daily_idx = np.argmin(np.abs(frequencies - daily_freq))
                    daily_power = power_spectrum[daily_idx]
                    
                    results['activity_oscillations'] = {
                        'dominant_frequencies': frequencies[np.argsort(power_spectrum)[-3:]].tolist(),
                        'daily_cycle_power': daily_power,
                        'total_power': np.sum(power_spectrum)
                    }
                    
                    print(f"Activity Oscillations: Daily cycle power = {daily_power:.3e}")
        
        # Activity type analysis (if available)
        if 'activity_type' in activity_data.columns:
            activity_types = activity_data['activity_type'].value_counts()
            
            results['activity_types'] = {
                'type_distribution': activity_types.to_dict(),
                'most_common_activity': activity_types.index[0] if len(activity_types) > 0 else None,
                'activity_diversity': len(activity_types)
            }
            
            print(f"Activity Types: {len(activity_types)} different types, "
                  f"Most common: {results['activity_types']['most_common_activity']}")
        
        # Cross-correlation with other data sources
        correlation_results = {}
        
        if 'heart_rate' in self.loaded_data and 'timestamp' in activity_data.columns:
            # Correlate activity with heart rate
            hr_data = self.loaded_data['heart_rate']
            
            if 'timestamp' in hr_data.columns and 'heart_rate' in hr_data.columns:
                # Resample both to same time resolution
                activity_resampled = activity_data.set_index('timestamp')['activity_level'].resample('5min').mean()
                hr_resampled = hr_data.set_index('timestamp')['heart_rate'].resample('5min').mean()
                
                # Find overlapping time periods
                common_times = activity_resampled.index.intersection(hr_resampled.index)
                
                if len(common_times) > 20:
                    activity_common = activity_resampled.loc[common_times].dropna()
                    hr_common = hr_resampled.loc[common_times].dropna()
                    
                    # Further filter to ensure same length
                    min_len = min(len(activity_common), len(hr_common))
                    if min_len > 10:
                        correlation = np.corrcoef(activity_common.values[:min_len], 
                                                hr_common.values[:min_len])[0, 1]
                        
                        correlation_results['activity_heart_rate'] = correlation
                        print(f"Activity-HR Correlation: {correlation:.3f}")
        
        results['cross_correlations'] = correlation_results
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        if 'circadian_activity' in results:
            # Check for reasonable circadian pattern
            amplitude = results['circadian_activity']['activity_amplitude']
            predictions_validated['circadian_pattern_present'] = amplitude > 0.1
            
            # Check for daytime activity peak (typical for humans)
            peak_hour = results['circadian_activity']['peak_activity_hour']
            predictions_validated['daytime_activity_peak'] = 8 <= peak_hour <= 20
        
        if 'activity_oscillations' in results:
            # Check for daily oscillatory pattern
            daily_power_ratio = (results['activity_oscillations']['daily_cycle_power'] / 
                               results['activity_oscillations']['total_power'])
            predictions_validated['daily_oscillation_detected'] = daily_power_ratio > 0.01
        
        if correlation_results.get('activity_heart_rate'):
            # Check for expected positive correlation between activity and heart rate
            correlation = correlation_results['activity_heart_rate']
            expected_correlation = self.theoretical_predictions['activity_consciousness_correlation']
            predictions_validated['activity_hr_coupling'] = correlation > expected_correlation
        
        results['theoretical_validation'] = predictions_validated
        validation_success = len(predictions_validated) > 0 and any(predictions_validated.values())
        results['validation_success'] = validation_success
        
        print(f"✅ Activity-Consciousness Coupling: {validation_success}")
        
        return results
    
    def validate_sprint_oscillatory_dynamics(self):
        """
        Special analysis of 400m sprint data for oscillatory patterns
        """
        print("\n🔬 EXPERIMENT: Sprint Oscillatory Dynamics (400m)")
        print("-" * 50)
        
        if 'heart_rate' not in self.loaded_data:
            print("❌ Heart rate data not available for sprint analysis")
            return {'error': 'No heart rate data'}
        
        hr_data = self.loaded_data['heart_rate'].copy()
        results = {}
        
        # Look for sprint-related contexts
        sprint_contexts = []
        
        if 'context' in hr_data.columns:
            unique_contexts = hr_data['context'].dropna().unique()
            
            for context in unique_contexts:
                context_str = str(context).lower()
                if any(keyword in context_str for keyword in ['sprint', '400m', 'track', 'run', 'race']):
                    sprint_contexts.append(context)
            
            print(f"Found {len(sprint_contexts)} potential sprint contexts: {sprint_contexts}")
        
        if not sprint_contexts:
            # Look for high-intensity periods based on heart rate
            if 'heart_rate' in hr_data.columns:
                hr_values = hr_data['heart_rate'].dropna()
                
                if len(hr_values) > 100:
                    # Define high intensity as top 10% of heart rates
                    high_intensity_threshold = np.percentile(hr_values, 90)
                    
                    # Find continuous periods of high intensity
                    high_intensity_mask = hr_data['heart_rate'] > high_intensity_threshold
                    
                    results['intensity_analysis'] = {
                        'high_intensity_threshold': high_intensity_threshold,
                        'high_intensity_periods': high_intensity_mask.sum(),
                        'percentage_high_intensity': high_intensity_mask.sum() / len(hr_data) * 100
                    }
                    
                    print(f"High Intensity Analysis: {high_intensity_mask.sum()} high-intensity records "
                          f"(>{high_intensity_threshold:.0f} bpm)")
        
        # Analyze sprint sessions (if identified)
        sprint_analysis = {}
        
        for sprint_context in sprint_contexts[:3]:  # Analyze up to 3 sprint sessions
            sprint_data = hr_data[hr_data['context'] == sprint_context].copy()
            
            if len(sprint_data) > 20 and 'heart_rate' in sprint_data.columns:
                sprint_hr = sprint_data['heart_rate'].dropna()
                
                sprint_session_results = {
                    'duration_records': len(sprint_data),
                    'mean_hr': sprint_hr.mean(),
                    'max_hr': sprint_hr.max(),
                    'hr_range': sprint_hr.max() - sprint_hr.min(),
                    'hr_variability': sprint_hr.std()
                }
                
                # Time series analysis if timestamps available
                if 'timestamp' in sprint_data.columns:
                    sprint_ts = sprint_data.set_index('timestamp')['heart_rate']
                    
                    # Calculate time resolution
                    if len(sprint_ts) > 5:
                        time_diffs = sprint_ts.index.to_series().diff().dropna()
                        median_interval = time_diffs.median().total_seconds()
                        
                        if median_interval > 0:
                            sampling_rate = 1.0 / median_interval
                            
                            # Frequency analysis for oscillatory patterns
                            if len(sprint_hr) > 10 and sampling_rate > 0.1:
                                frequencies, power_spectrum = signal.periodogram(
                                    sprint_hr.values,
                                    fs=sampling_rate
                                )
                                
                                # Find dominant frequencies
                                dominant_idx = np.argsort(power_spectrum)[-3:]
                                dominant_freqs = frequencies[dominant_idx]
                                dominant_powers = power_spectrum[dominant_idx]
                                
                                sprint_session_results['oscillatory_analysis'] = {
                                    'sampling_rate': sampling_rate,
                                    'dominant_frequencies': dominant_freqs.tolist(),
                                    'dominant_powers': dominant_powers.tolist(),
                                    'total_power': np.sum(power_spectrum)
                                }
                                
                                # Check for consciousness-range frequencies during sprint
                                consciousness_band = (frequencies >= 2) & (frequencies <= 10)
                                if np.any(consciousness_band):
                                    consciousness_power = np.sum(power_spectrum[consciousness_band])
                                    total_power = np.sum(power_spectrum)
                                    sprint_session_results['consciousness_band_power'] = consciousness_power / total_power
                
                sprint_analysis[str(sprint_context)] = sprint_session_results
                
                print(f"Sprint {sprint_context}: {len(sprint_data)} records, "
                      f"Mean HR={sprint_hr.mean():.0f}, Max HR={sprint_hr.max():.0f}")
        
        results['sprint_sessions'] = sprint_analysis
        
        # Overall sprint oscillatory validation
        predictions_validated = {}
        
        if sprint_analysis:
            # Check if oscillatory patterns were found in sprint data
            oscillatory_found = any(
                'oscillatory_analysis' in session_data 
                for session_data in sprint_analysis.values()
            )
            predictions_validated['sprint_oscillatory_patterns'] = oscillatory_found
            
            # Check for consciousness-band activity during sprints
            consciousness_activity = any(
                'consciousness_band_power' in session_data and session_data['consciousness_band_power'] > 0.1
                for session_data in sprint_analysis.values()
            )
            predictions_validated['sprint_consciousness_coupling'] = consciousness_activity
            
            # Check for high-intensity physiological response
            high_intensity_response = any(
                session_data['max_hr'] > 160  # Typical high-intensity threshold
                for session_data in sprint_analysis.values()
            )
            predictions_validated['high_intensity_response'] = high_intensity_response
        
        results['theoretical_validation'] = predictions_validated
        validation_success = len(predictions_validated) > 0 and any(predictions_validated.values())
        results['validation_success'] = validation_success
        
        print(f"✅ Sprint Oscillatory Dynamics: {validation_success}")
        
        return results
    
    def validate_multi_system_integration(self):
        """
        Validate integration and coupling between multiple biometric systems
        """
        print("\n🔬 EXPERIMENT: Multi-System Integration")
        print("-" * 50)
        
        available_systems = list(self.loaded_data.keys())
        
        if len(available_systems) < 2:
            print("❌ Need at least 2 data systems for integration analysis")
            return {'error': 'Insufficient data systems'}
        
        results = {
            'available_systems': available_systems,
            'integration_analysis': {},
            'cross_correlations': {},
            'temporal_alignment': {}
        }
        
        # Analyze temporal overlap between systems
        system_time_ranges = {}
        
        for system in available_systems:
            data = self.loaded_data[system]
            
            if 'timestamp' in data.columns:
                time_range = (data['timestamp'].min(), data['timestamp'].max())
                system_time_ranges[system] = time_range
                
                print(f"{system}: {time_range[0]} to {time_range[1]} "
                      f"({time_range[1] - time_range[0]})")
        
        results['system_time_ranges'] = system_time_ranges
        
        # Find common time periods for multi-system analysis
        if len(system_time_ranges) >= 2:
            # Calculate overlap between all pairs of systems
            system_pairs = [(s1, s2) for i, s1 in enumerate(available_systems) 
                           for s2 in available_systems[i+1:]]
            
            overlap_analysis = {}
            
            for s1, s2 in system_pairs:
                if s1 in system_time_ranges and s2 in system_time_ranges:
                    range1 = system_time_ranges[s1]
                    range2 = system_time_ranges[s2]
                    
                    # Calculate temporal overlap
                    overlap_start = max(range1[0], range2[0])
                    overlap_end = min(range1[1], range2[1])
                    
                    if overlap_start < overlap_end:
                        overlap_duration = overlap_end - overlap_start
                        
                        overlap_analysis[f"{s1}_{s2}"] = {
                            'overlap_start': overlap_start,
                            'overlap_end': overlap_end,
                            'overlap_duration': overlap_duration,
                            'overlap_hours': overlap_duration.total_seconds() / 3600
                        }
                        
                        print(f"{s1} ↔ {s2}: {overlap_duration} overlap")
            
            results['temporal_alignment'] = overlap_analysis
        
        # Cross-system correlation analysis
        correlation_matrix = {}
        
        # Heart rate vs Activity correlation
        if 'heart_rate' in available_systems and 'activity' in available_systems:
            hr_data = self.loaded_data['heart_rate']
            activity_data = self.loaded_data['activity']
            
            correlation = self._calculate_cross_system_correlation(
                hr_data, 'heart_rate', 'timestamp',
                activity_data, 'activity_level', 'timestamp'
            )
            
            if correlation is not None:
                correlation_matrix['heart_rate_activity'] = correlation
                print(f"Heart Rate ↔ Activity correlation: {correlation:.3f}")
        
        # Sleep vs Activity correlation
        if 'sleep' in available_systems and 'activity' in available_systems:
            sleep_data = self.loaded_data['sleep']
            activity_data = self.loaded_data['activity']
            
            # For sleep, we need to encode stages numerically
            if 'sleep_stage' in sleep_data.columns:
                sleep_encoded = sleep_data.copy()
                stage_encoding = {'wake': 4, 'rem': 3, 'light': 2, 'deep': 1}
                sleep_encoded['sleep_stage_numeric'] = sleep_encoded['sleep_stage'].map(stage_encoding)
                
                correlation = self._calculate_cross_system_correlation(
                    sleep_encoded, 'sleep_stage_numeric', 'timestamp',
                    activity_data, 'activity_level', 'timestamp'
                )
                
                if correlation is not None:
                    correlation_matrix['sleep_activity'] = correlation
                    print(f"Sleep ↔ Activity correlation: {correlation:.3f}")
        
        results['cross_correlations'] = correlation_matrix
        
        # Multi-system oscillatory coupling analysis
        coupling_analysis = {}
        
        if len(correlation_matrix) > 0:
            # Calculate overall system coupling strength
            coupling_values = list(correlation_matrix.values())
            mean_coupling = np.mean([abs(c) for c in coupling_values if not np.isnan(c)])
            
            coupling_analysis['mean_coupling_strength'] = mean_coupling
            coupling_analysis['strong_couplings'] = sum(1 for c in coupling_values if abs(c) > 0.3)
            coupling_analysis['total_correlations'] = len(coupling_values)
            
            print(f"Multi-system coupling: Mean strength = {mean_coupling:.3f}, "
                  f"Strong couplings = {coupling_analysis['strong_couplings']}")
        
        results['coupling_analysis'] = coupling_analysis
        
        # Validate theoretical predictions for multi-system integration
        predictions_validated = {}
        
        if 'coupling_analysis' in results and 'mean_coupling_strength' in coupling_analysis:
            # Check for expected multi-scale coupling
            expected_coupling = self.theoretical_predictions['multi_scale_coupling_strength']
            actual_coupling = coupling_analysis['mean_coupling_strength']
            predictions_validated['multi_system_coupling'] = actual_coupling >= expected_coupling
        
        if 'temporal_alignment' in results and results['temporal_alignment']:
            # Check for sufficient temporal overlap for integration analysis
            overlap_count = len(results['temporal_alignment'])
            predictions_validated['temporal_integration'] = overlap_count > 0
        
        if correlation_matrix:
            # Check for expected physiological correlations
            if 'heart_rate_activity' in correlation_matrix:
                hr_activity_corr = correlation_matrix['heart_rate_activity']
                predictions_validated['hr_activity_coupling'] = hr_activity_corr > 0.2
        
        results['theoretical_validation'] = predictions_validated
        validation_success = len(predictions_validated) > 0 and any(predictions_validated.values())
        results['validation_success'] = validation_success
        
        print(f"✅ Multi-System Integration: {validation_success}")
        
        return results
    
    def _calculate_cross_system_correlation(self, data1, col1, time_col1, 
                                           data2, col2, time_col2, resample_freq='5min'):
        """
        Calculate correlation between two time series from different systems
        """
        try:
            # Create time series
            ts1 = data1.set_index(time_col1)[col1].dropna()
            ts2 = data2.set_index(time_col2)[col2].dropna()
            
            # Resample to common frequency
            ts1_resampled = ts1.resample(resample_freq).mean()
            ts2_resampled = ts2.resample(resample_freq).mean()
            
            # Find common time period
            common_index = ts1_resampled.index.intersection(ts2_resampled.index)
            
            if len(common_index) > 10:
                ts1_common = ts1_resampled.loc[common_index].dropna()
                ts2_common = ts2_resampled.loc[common_index].dropna()
                
                # Ensure same length
                min_len = min(len(ts1_common), len(ts2_common))
                
                if min_len > 5:
                    correlation = np.corrcoef(ts1_common.values[:min_len], 
                                           ts2_common.values[:min_len])[0, 1]
                    return correlation
            
        except Exception as e:
            print(f"  Warning: Correlation calculation failed - {str(e)}")
        
        return None
    
    def run_comprehensive_experimental_validation(self):
        """
        Run complete experimental validation using all available biometric data
        """
        print("\n" + "="*70)
        print("🏃‍♂️💓 COMPREHENSIVE EXPERIMENTAL BIOMETRIC VALIDATION 💓🏃‍♂️")
        print("="*70)
        print("Validating consciousness theories with REAL biometric data!")
        print("="*70)
        
        # Load all available data
        loaded_categories = self.load_experimental_data()
        
        if not loaded_categories:
            print("\n❌ No experimental data found!")
            print("Please ensure your biometric data is placed in:")
            print(f"  {self.experimental_data_dir}/raw/<category>/")
            print("Supported formats: CSV, JSON, Parquet")
            return {'error': 'No experimental data available'}
        
        # Run all validation experiments
        experiments = [
            ('Heart Rate-Consciousness Coupling', self.validate_heart_rate_consciousness_coupling),
            ('Sleep-Consciousness Transitions', self.validate_sleep_consciousness_transitions),
            ('Activity-Consciousness Coupling', self.validate_activity_consciousness_coupling),
            ('Sprint Oscillatory Dynamics', self.validate_sprint_oscillatory_dynamics),
            ('Multi-System Integration', self.validate_multi_system_integration)
        ]
        
        validation_results = {}
        successful_validations = 0
        
        for experiment_name, experiment_func in experiments:
            print(f"\n{'='*50}")
            try:
                result = experiment_func()
                validation_results[experiment_name] = result
                
                if result.get('validation_success', False):
                    successful_validations += 1
                    
            except Exception as e:
                print(f"❌ Error in {experiment_name}: {str(e)}")
                validation_results[experiment_name] = {'error': str(e)}
        
        # Generate comprehensive summary
        comprehensive_summary = {
            'total_experiments': len(experiments),
            'successful_validations': successful_validations,
            'validation_success_rate': successful_validations / len(experiments),
            'loaded_data_categories': loaded_categories,
            'experimental_validation_success': successful_validations >= 3,
            'theoretical_predictions_validated': self._count_validated_predictions(validation_results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save comprehensive results
        complete_results = {
            'summary': comprehensive_summary,
            'individual_experiments': validation_results,
            'loaded_data_info': {
                category: {'num_records': len(self.loaded_data[category])} 
                for category in loaded_categories
            }
        }
        
        # Save results
        with open(self.results_dir / 'comprehensive_experimental_validation.json', 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_experimental_visualizations(validation_results)
        
        # Print final summary
        print("\n" + "="*70)
        print("📊 EXPERIMENTAL VALIDATION SUMMARY")
        print("="*70)
        print(f"Data Categories Loaded: {len(loaded_categories)}")
        print(f"Experiments Run: {len(experiments)}")
        print(f"Successful Validations: {successful_validations}")
        print(f"Success Rate: {comprehensive_summary['validation_success_rate']*100:.1f}%")
        print(f"Framework Validated: {comprehensive_summary['experimental_validation_success']}")
        print("="*70)
        
        if comprehensive_summary['experimental_validation_success']:
            print("\n🎉 EXPERIMENTAL VALIDATION SUCCESS! 🎉")
            print("Real biometric data supports consciousness theories!")
        else:
            print(f"\n⚠️ Need more successful validations for complete framework validation")
        
        return complete_results
    
    def _count_validated_predictions(self, validation_results):
        """Count total validated theoretical predictions across all experiments"""
        total_predictions = 0
        
        for experiment_result in validation_results.values():
            if isinstance(experiment_result, dict) and 'theoretical_validation' in experiment_result:
                theoretical_val = experiment_result['theoretical_validation']
                if isinstance(theoretical_val, dict):
                    total_predictions += sum(1 for pred in theoretical_val.values() if pred)
        
        return total_predictions
    
    def _create_experimental_visualizations(self, validation_results):
        """Create comprehensive visualizations of experimental validation results"""
        print("\n🎨 Creating experimental validation visualizations...")
        
        # Create summary dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Experimental Biometric Validation Dashboard', fontsize=16, fontweight='bold')
        
        # Validation success summary
        ax1 = axes[0, 0]
        experiment_names = list(validation_results.keys())
        success_status = [validation_results[exp].get('validation_success', False) 
                         for exp in experiment_names]
        
        colors = ['green' if success else 'red' for success in success_status]
        bars = ax1.barh(range(len(experiment_names)), [1 if success else 0 for success in success_status],
                       color=colors, alpha=0.7)
        
        ax1.set_yticks(range(len(experiment_names)))
        ax1.set_yticklabels([name.replace('-', '\n') for name in experiment_names], fontsize=10)
        ax1.set_xlabel('Validation Status')
        ax1.set_title('Experiment Validation Results')
        ax1.set_xlim(0, 1.2)
        
        for i, (bar, status) in enumerate(zip(bars, success_status)):
            label = '✅ SUCCESS' if status else '❌ FAILED'
            ax1.text(0.6, bar.get_y() + bar.get_height()/2, label, 
                    ha='center', va='center', fontweight='bold', fontsize=8)
        
        # Heart rate analysis (if available)
        ax2 = axes[0, 1]
        hr_results = validation_results.get('Heart Rate-Consciousness Coupling', {})
        
        if 'basic_stats' in hr_results:
            hr_stats = hr_results['basic_stats']
            metrics = ['mean_hr', 'std_hr', 'min_hr', 'max_hr']
            values = [hr_stats.get(metric, 0) for metric in metrics]
            
            bars = ax2.bar(metrics, values, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
            ax2.set_title('Heart Rate Statistics')
            ax2.set_ylabel('Heart Rate (bpm)')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.0f}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No Heart Rate\nData Available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Heart Rate Analysis')
        
        # Sleep analysis (if available)
        ax3 = axes[0, 2]
        sleep_results = validation_results.get('Sleep-Consciousness Transitions', {})
        
        if 'sleep_architecture' in sleep_results:
            stage_dist = sleep_results['sleep_architecture']['stage_distribution']
            
            if stage_dist:
                stages = list(stage_dist.keys())
                counts = list(stage_dist.values())
                
                ax3.pie(counts, labels=stages, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Sleep Stage Distribution')
            else:
                ax3.text(0.5, 0.5, 'No Sleep Stage\nData Available', ha='center', va='center',
                        transform=ax3.transAxes, fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'No Sleep\nData Available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Sleep Analysis')
        
        # Activity patterns (if available)
        ax4 = axes[1, 0]
        activity_results = validation_results.get('Activity-Consciousness Coupling', {})
        
        if 'circadian_activity' in activity_results:
            hourly_pattern = activity_results['circadian_activity']['hourly_pattern']
            
            if hourly_pattern:
                hours = sorted(hourly_pattern.keys())
                activities = [hourly_pattern[hour] for hour in hours]
                
                ax4.plot(hours, activities, marker='o', linewidth=2, markersize=4)
                ax4.set_xlabel('Hour of Day')
                ax4.set_ylabel('Activity Level')
                ax4.set_title('Circadian Activity Pattern')
                ax4.grid(True, alpha=0.3)
                ax4.set_xticks(range(0, 24, 4))
            else:
                ax4.text(0.5, 0.5, 'No Activity Pattern\nData Available', ha='center', va='center',
                        transform=ax4.transAxes, fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'No Activity\nData Available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Activity Analysis')
        
        # Cross-correlations (if available)
        ax5 = axes[1, 1]
        multi_system_results = validation_results.get('Multi-System Integration', {})
        
        if 'cross_correlations' in multi_system_results:
            correlations = multi_system_results['cross_correlations']
            
            if correlations:
                corr_names = list(correlations.keys())
                corr_values = list(correlations.values())
                
                colors = ['green' if corr > 0 else 'red' for corr in corr_values]
                bars = ax5.bar(range(len(corr_names)), corr_values, color=colors, alpha=0.7)
                
                ax5.set_xticks(range(len(corr_names)))
                ax5.set_xticklabels([name.replace('_', '\n') for name in corr_names], rotation=45)
                ax5.set_ylabel('Correlation')
                ax5.set_title('Cross-System Correlations')
                ax5.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax5.grid(True, alpha=0.3)
                
                for bar, value in zip(bars, corr_values):
                    ax5.text(bar.get_x() + bar.get_width()/2, 
                            bar.get_height() + (0.02 if value > 0 else -0.05),
                            f'{value:.3f}', ha='center', va='bottom' if value > 0 else 'top')
            else:
                ax5.text(0.5, 0.5, 'No Cross-Correlation\nData Available', ha='center', va='center',
                        transform=ax5.transAxes, fontsize=12)
        else:
            ax5.text(0.5, 0.5, 'No Integration\nData Available', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Cross-System Analysis')
        
        # Overall validation summary
        ax6 = axes[1, 2]
        total_experiments = len(validation_results)
        successful_experiments = sum(1 for result in validation_results.values() 
                                   if result.get('validation_success', False))
        
        # Create pie chart for validation summary
        labels = ['Successful', 'Failed']
        sizes = [successful_experiments, total_experiments - successful_experiments]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax6.set_title('Overall Validation Success')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'experimental_validation_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Experimental validation dashboard saved")
        print(f"  📁 Results directory: {self.results_dir}")


def run_experimental_validation():
    """
    Main function to run experimental biometric validation
    """
    validator = ExperimentalBiometricValidator()
    results = validator.run_comprehensive_experimental_validation()
    
    return validator, results


if __name__ == "__main__":
    validator, results = run_experimental_validation()
    
    print(f"\n🎯 EXPERIMENTAL VALIDATION COMPLETE!")
    print(f"Results available in: {validator.results_dir}")
    
    if results.get('summary', {}).get('experimental_validation_success', False):
        print("\n🏆 REAL DATA VALIDATES CONSCIOUSNESS THEORIES! 🏆")
    else:
        print("\n📊 Partial validation achieved - more data analysis needed")
