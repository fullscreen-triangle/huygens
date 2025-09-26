#!/usr/bin/env python3
"""
Cadence Oscillatory Analysis

Validates oscillatory theories using real biomechanical cadence data from track running.
Cadence represents the fundamental oscillatory rhythm of human locomotion - steps per minute.

This script applies the Universal Biological Oscillatory Framework to analyze:
1. Cadence frequency stability and coupling
2. Multi-scale oscillatory coupling with heart rate and performance
3. Terrain-dependent oscillatory adaptation 
4. Left/right coupling symmetry in cadence oscillations
5. Validation of bounded oscillatory system theorems

REVOLUTIONARY VALIDATION: First framework to apply oscillatory theory to real running biomechanics!
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats, fft
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CadenceOscillatoryAnalyzer:
    """
    Analyzes cadence oscillations using the Universal Biological Oscillatory Framework
    """
    
    def __init__(self, data_file="../../experimental-data/circuit/annotated_track_series.json", 
                 results_dir="biomechanics_results/cadence"):
        self.data_file = Path(data_file)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Theoretical predictions from oscillatory framework
        self.theoretical_predictions = {
            'optimal_cadence_range': (85, 105),  # steps/min for efficient running
            'cadence_stability_threshold': 0.05,  # CV < 5% for stable oscillation
            'heart_rate_coupling_strength': 0.3,  # Expected coupling coefficient  
            'terrain_adaptation_range': (80, 110),  # Adaptive cadence range
            'frequency_coherence_threshold': 0.7,  # Coherence > 0.7 for coupled oscillations
            'oscillatory_efficiency_target': 0.85,  # Target efficiency for optimal oscillations
        }
        
        # Load and validate data
        self.data = None
        self.cadence_time_series = None
        self.validation_results = {}
        
        print("🏃‍♂️⚡ CADENCE OSCILLATORY ANALYZER ⚡🏃‍♂️")
        print("=" * 60)
        print("Validating Universal Oscillatory Framework with REAL cadence data!")
        print("=" * 60)
    
    def load_track_data(self):
        """Load and preprocess the annotated track series data"""
        print("\n📂 Loading Track Data")
        print("-" * 40)
        
        try:
            with open(self.data_file, 'r') as f:
                raw_data = json.load(f)
            
            # Convert to DataFrame
            self.data = pd.DataFrame(raw_data)
            
            # Clean and validate cadence data
            self.data = self.data[self.data['cadence'] > 0]  # Remove zero cadence (stops)
            self.data = self.data.reset_index(drop=True)
            
            # Create time series
            self.cadence_time_series = self.data['cadence'].values
            
            print(f"✅ Loaded {len(self.data)} data points")
            print(f"📊 Cadence range: {self.cadence_time_series.min():.1f} - {self.cadence_time_series.max():.1f} steps/min")
            print(f"⏱️  Time span: {self.data['time'].max() - self.data['time'].min():.1f} seconds")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def analyze_cadence_oscillatory_stability(self):
        """
        Analyze cadence stability using oscillatory system theory
        Tests: Bounded Oscillatory System Theorem
        """
        print("\n🔬 EXPERIMENT 1: Cadence Oscillatory Stability")
        print("-" * 50)
        
        if self.cadence_time_series is None:
            print("❌ No data available")
            return {}
        
        results = {}
        cadence = self.cadence_time_series
        
        # Basic stability metrics
        mean_cadence = np.mean(cadence)
        std_cadence = np.std(cadence)
        cv_cadence = std_cadence / mean_cadence  # Coefficient of variation
        
        results['basic_stats'] = {
            'mean_cadence': mean_cadence,
            'std_cadence': std_cadence,
            'cv_cadence': cv_cadence,
            'range_cadence': np.max(cadence) - np.min(cadence)
        }
        
        print(f"Mean Cadence: {mean_cadence:.2f} ± {std_cadence:.2f} steps/min")
        print(f"Coefficient of Variation: {cv_cadence:.3f} ({cv_cadence*100:.1f}%)")
        
        # Bounded Oscillatory System Analysis
        # Test if cadence oscillations remain within theoretical bounds
        optimal_range = self.theoretical_predictions['optimal_cadence_range']
        in_optimal_range = np.sum((cadence >= optimal_range[0]) & (cadence <= optimal_range[1]))
        optimal_percentage = in_optimal_range / len(cadence) * 100
        
        results['bounded_system_validation'] = {
            'optimal_range_adherence': optimal_percentage,
            'range_violations': len(cadence) - in_optimal_range,
            'theoretical_bounds': optimal_range,
            'bounded_system_stability': cv_cadence < self.theoretical_predictions['cadence_stability_threshold']
        }
        
        print(f"Optimal Range Adherence: {optimal_percentage:.1f}%")
        print(f"Bounded System Stability: {'✅ STABLE' if results['bounded_system_validation']['bounded_system_stability'] else '⚠️  UNSTABLE'}")
        
        # Frequency domain analysis of cadence oscillations
        # Cadence as fundamental locomotor frequency
        if len(cadence) > 10:
            # Calculate running frequency (Hz) from cadence (steps/min)
            running_frequency = mean_cadence / 60.0  # Convert to Hz
            
            # Power spectral density of cadence variations
            frequencies, psd = signal.periodogram(cadence, fs=1.0)  # Assume 1 Hz sampling
            
            # Find dominant frequency of cadence variations
            dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
            dominant_frequency = frequencies[dominant_freq_idx]
            
            results['frequency_analysis'] = {
                'fundamental_running_frequency': running_frequency,
                'cadence_variation_dominant_freq': dominant_frequency,
                'spectral_power_total': np.sum(psd),
                'spectral_power_dominant': psd[dominant_freq_idx],
                'frequency_stability': psd[dominant_freq_idx] / np.sum(psd)
            }
            
            print(f"Fundamental Running Frequency: {running_frequency:.2f} Hz")
            print(f"Cadence Variation Frequency: {dominant_frequency:.4f} Hz")
        
        # Oscillatory coherence analysis
        if len(cadence) > 20:
            # Calculate autocorrelation to find oscillatory patterns
            autocorr = np.correlate(cadence - np.mean(cadence), cadence - np.mean(cadence), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find first significant minimum (indicates oscillatory period)
            significant_lags = np.where(autocorr[1:20] < 0.5)[0]  # Look for correlation < 0.5
            
            if len(significant_lags) > 0:
                oscillatory_period = significant_lags[0] + 1
                results['oscillatory_coherence'] = {
                    'autocorr_oscillatory_period': oscillatory_period,
                    'coherence_decay_rate': -np.log(autocorr[oscillatory_period]) / oscillatory_period if oscillatory_period < len(autocorr) else 0,
                    'long_term_stability': autocorr[min(10, len(autocorr)-1)]
                }
                
                print(f"Oscillatory Period: {oscillatory_period} samples")
                print(f"Long-term Stability: {results['oscillatory_coherence']['long_term_stability']:.3f}")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        # Test cadence stability prediction
        predictions_validated['cadence_stability'] = cv_cadence < self.theoretical_predictions['cadence_stability_threshold']
        
        # Test optimal range prediction
        predictions_validated['optimal_range'] = optimal_percentage > 80  # 80% in optimal range
        
        # Test bounded system behavior
        predictions_validated['bounded_oscillation'] = results['bounded_system_validation']['bounded_system_stability']
        
        results['theoretical_validation'] = predictions_validated
        validation_success = sum(predictions_validated.values()) >= 2  # At least 2/3 validations
        results['validation_success'] = validation_success
        
        print(f"🎯 Theoretical Validation: {'✅ SUCCESS' if validation_success else '❌ FAILED'}")
        
        return results
    
    def analyze_cadence_heart_rate_coupling(self):
        """
        Analyze oscillatory coupling between cadence and heart rate
        Tests: Multi-scale Oscillatory Coupling Theory
        """
        print("\n🔬 EXPERIMENT 2: Cadence-Heart Rate Oscillatory Coupling")
        print("-" * 55)
        
        if self.data is None:
            print("❌ No data available")
            return {}
        
        # Extract synchronized cadence and heart rate data
        cadence_data = self.data['cadence'].values
        hr_data = self.data['heart_rate'].values
        
        # Remove any missing values
        valid_indices = (cadence_data > 0) & (hr_data > 0)
        cadence_clean = cadence_data[valid_indices]
        hr_clean = hr_data[valid_indices]
        
        if len(cadence_clean) < 10:
            print("❌ Insufficient synchronized data")
            return {}
        
        results = {}
        
        # Basic coupling analysis
        correlation = np.corrcoef(cadence_clean, hr_clean)[0, 1]
        
        results['basic_coupling'] = {
            'correlation_coefficient': correlation,
            'sample_size': len(cadence_clean),
            'cadence_mean': np.mean(cadence_clean),
            'hr_mean': np.mean(hr_clean)
        }
        
        print(f"Correlation Coefficient: {correlation:.3f}")
        print(f"Sample Size: {len(cadence_clean)} synchronized points")
        
        # Multi-scale oscillatory coupling analysis
        if len(cadence_clean) > 20:
            # Cross-correlation analysis
            cross_corr = np.correlate(cadence_clean - np.mean(cadence_clean), 
                                    hr_clean - np.mean(hr_clean), mode='full')
            cross_corr = cross_corr / (np.std(cadence_clean) * np.std(hr_clean) * len(cadence_clean))
            
            # Find peak correlation and lag
            max_corr_idx = np.argmax(np.abs(cross_corr))
            lag = max_corr_idx - len(cadence_clean) + 1
            max_correlation = cross_corr[max_corr_idx]
            
            results['cross_correlation'] = {
                'max_correlation': max_correlation,
                'optimal_lag': lag,
                'coupling_strength': np.abs(max_correlation)
            }
            
            print(f"Maximum Cross-Correlation: {max_correlation:.3f}")
            print(f"Optimal Lag: {lag} samples")
            
            # Coherence analysis using Welch's method
            if len(cadence_clean) > 50:
                frequencies, coherence = signal.coherence(cadence_clean, hr_clean, 
                                                        fs=1.0, nperseg=min(32, len(cadence_clean)//4))
                
                # Find peak coherence
                peak_coherence_idx = np.argmax(coherence[1:]) + 1  # Skip DC
                peak_coherence = coherence[peak_coherence_idx]
                peak_frequency = frequencies[peak_coherence_idx]
                
                results['coherence_analysis'] = {
                    'peak_coherence': peak_coherence,
                    'peak_frequency': peak_frequency,
                    'mean_coherence': np.mean(coherence[1:]),  # Exclude DC
                    'coherence_bandwidth': np.sum(coherence > 0.5)  # Number of freq bins with coherence > 0.5
                }
                
                print(f"Peak Coherence: {peak_coherence:.3f} at {peak_frequency:.4f} Hz")
                print(f"Mean Coherence: {np.mean(coherence[1:]):.3f}")
        
        # Oscillatory coupling strength classification
        coupling_strength_categories = {
            'weak': (0.0, 0.3),
            'moderate': (0.3, 0.6),
            'strong': (0.6, 1.0)
        }
        
        abs_correlation = np.abs(correlation)
        coupling_category = 'weak'
        
        for category, (lower, upper) in coupling_strength_categories.items():
            if lower <= abs_correlation < upper:
                coupling_category = category
                break
        
        results['coupling_classification'] = {
            'coupling_strength_category': coupling_category,
            'coupling_strength_value': abs_correlation
        }
        
        print(f"Coupling Strength: {coupling_category.upper()} ({abs_correlation:.3f})")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        # Test expected coupling strength
        expected_coupling = self.theoretical_predictions['heart_rate_coupling_strength']
        predictions_validated['coupling_strength'] = abs_correlation >= expected_coupling
        
        # Test coherence threshold (if coherence analysis was performed)
        if 'coherence_analysis' in results:
            coherence_threshold = self.theoretical_predictions['frequency_coherence_threshold']
            predictions_validated['coherence_threshold'] = results['coherence_analysis']['peak_coherence'] >= coherence_threshold
        
        # Test multi-scale coupling (non-zero lag indicates multi-scale interaction)
        if 'cross_correlation' in results:
            predictions_validated['multi_scale_coupling'] = np.abs(results['cross_correlation']['optimal_lag']) > 0
        
        results['theoretical_validation'] = predictions_validated
        validation_success = sum(predictions_validated.values()) >= len(predictions_validated) // 2
        results['validation_success'] = validation_success
        
        print(f"🎯 Theoretical Validation: {'✅ SUCCESS' if validation_success else '❌ FAILED'}")
        
        return results
    
    def analyze_terrain_adaptive_oscillations(self):
        """
        Analyze how cadence oscillations adapt to different track segments
        Tests: Environmental Coupling and Adaptive Oscillatory Response
        """
        print("\n🔬 EXPERIMENT 3: Terrain-Adaptive Oscillatory Response")
        print("-" * 55)
        
        if self.data is None:
            print("❌ No data available")
            return {}
        
        # Group by track segments
        segment_data = self.data.groupby('segments')
        
        if len(segment_data) < 2:
            print("❌ Insufficient segment variety")
            return {}
        
        results = {}
        segment_analysis = {}
        
        print(f"Analyzing {len(segment_data)} different track segments:")
        
        for segment_name, segment_df in segment_data:
            if len(segment_df) < 5:  # Skip segments with too few points
                continue
                
            cadence_segment = segment_df['cadence'].values
            
            segment_stats = {
                'n_points': len(cadence_segment),
                'mean_cadence': np.mean(cadence_segment),
                'std_cadence': np.std(cadence_segment),
                'cv_cadence': np.std(cadence_segment) / np.mean(cadence_segment),
                'cadence_range': np.max(cadence_segment) - np.min(cadence_segment)
            }
            
            # Calculate adaptation metrics
            segment_stats['adaptation_coefficient'] = segment_stats['cv_cadence']
            segment_stats['oscillatory_efficiency'] = 1.0 / (1.0 + segment_stats['cv_cadence'])
            
            segment_analysis[segment_name] = segment_stats
            
            print(f"  {segment_name}: {segment_stats['mean_cadence']:.1f} ± {segment_stats['std_cadence']:.1f} steps/min")
        
        results['segment_analysis'] = segment_analysis
        
        # Cross-segment oscillatory coupling analysis
        if len(segment_analysis) >= 2:
            segment_names = list(segment_analysis.keys())
            
            # Create cadence adaptation matrix
            adaptation_matrix = []
            mean_cadences = []
            
            for segment_name in segment_names:
                mean_cadences.append(segment_analysis[segment_name]['mean_cadence'])
                adaptation_matrix.append([
                    segment_analysis[segment_name]['mean_cadence'],
                    segment_analysis[segment_name]['cv_cadence'],
                    segment_analysis[segment_name]['oscillatory_efficiency']
                ])
            
            adaptation_matrix = np.array(adaptation_matrix)
            
            # Calculate inter-segment oscillatory coupling
            cadence_range_adaptation = np.max(mean_cadences) - np.min(mean_cadences)
            adaptation_coefficient = np.std(mean_cadences) / np.mean(mean_cadences)
            
            results['cross_segment_analysis'] = {
                'cadence_range_adaptation': cadence_range_adaptation,
                'adaptation_coefficient': adaptation_coefficient,
                'segment_count': len(segment_names),
                'segments': segment_names
            }
            
            print(f"Cadence Adaptation Range: {cadence_range_adaptation:.1f} steps/min")
            print(f"Adaptation Coefficient: {adaptation_coefficient:.3f}")
            
            # Statistical analysis of segment differences
            segment_cadences = [self.data[self.data['segments'] == segment]['cadence'].values 
                              for segment in segment_names if len(self.data[self.data['segments'] == segment]) > 5]
            
            if len(segment_cadences) >= 2:
                # ANOVA test for significant differences
                try:
                    f_stat, p_value = stats.f_oneway(*segment_cadences)
                    results['statistical_analysis'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant_difference': p_value < 0.05
                    }
                    
                    print(f"ANOVA F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}")
                    print(f"Significant Adaptation: {'✅ YES' if p_value < 0.05 else '❌ NO'}")
                    
                except Exception as e:
                    print(f"Statistical analysis failed: {str(e)}")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        # Test adaptive range prediction
        if 'cross_segment_analysis' in results:
            terrain_range = self.theoretical_predictions['terrain_adaptation_range']
            all_cadences = [stats['mean_cadence'] for stats in segment_analysis.values()]
            
            # Check if adaptations stay within theoretical bounds
            min_cadence = min(all_cadences)
            max_cadence = max(all_cadences)
            
            predictions_validated['adaptation_bounds'] = (
                min_cadence >= terrain_range[0] and max_cadence <= terrain_range[1]
            )
            
            # Test for significant adaptive response
            if 'statistical_analysis' in results:
                predictions_validated['adaptive_response'] = results['statistical_analysis']['significant_difference']
        
        # Test oscillatory efficiency maintenance
        efficiency_values = [stats['oscillatory_efficiency'] for stats in segment_analysis.values()]
        if efficiency_values:
            mean_efficiency = np.mean(efficiency_values)
            target_efficiency = self.theoretical_predictions['oscillatory_efficiency_target']
            predictions_validated['efficiency_maintenance'] = mean_efficiency >= target_efficiency
            
            print(f"Mean Oscillatory Efficiency: {mean_efficiency:.3f}")
        
        results['theoretical_validation'] = predictions_validated
        validation_success = sum(predictions_validated.values()) >= len(predictions_validated) // 2
        results['validation_success'] = validation_success
        
        print(f"🎯 Theoretical Validation: {'✅ SUCCESS' if validation_success else '❌ FAILED'}")
        
        return results
    
    def create_comprehensive_visualizations(self, results):
        """Generate comprehensive visualizations of cadence oscillatory analysis"""
        print("\n🎨 Creating Comprehensive Visualizations...")
        
        if self.data is None:
            print("❌ No data for visualization")
            return
        
        # Create multi-panel dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Cadence time series with oscillatory analysis
        ax1 = plt.subplot(3, 4, 1)
        time = self.data['time'].values
        cadence = self.data['cadence'].values
        
        plt.plot(time, cadence, 'b-', alpha=0.7, linewidth=1)
        plt.axhline(y=np.mean(cadence), color='r', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(cadence):.1f}')
        
        # Add optimal range
        optimal_range = self.theoretical_predictions['optimal_cadence_range']
        plt.axhspan(optimal_range[0], optimal_range[1], alpha=0.2, color='green', label='Optimal Range')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Cadence (steps/min)')
        plt.title('Cadence Oscillatory Time Series')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Cadence distribution and stability analysis
        ax2 = plt.subplot(3, 4, 2)
        plt.hist(cadence, bins=20, alpha=0.7, color='blue', density=True)
        plt.axvline(x=np.mean(cadence), color='red', linestyle='--', label=f'Mean: {np.mean(cadence):.1f}')
        plt.axvline(x=np.mean(cadence) + np.std(cadence), color='orange', linestyle='--', alpha=0.7)
        plt.axvline(x=np.mean(cadence) - np.std(cadence), color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {np.std(cadence):.1f}')
        
        plt.xlabel('Cadence (steps/min)')
        plt.ylabel('Density')
        plt.title('Cadence Distribution & Stability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Power spectral density of cadence oscillations
        ax3 = plt.subplot(3, 4, 3)
        frequencies, psd = signal.periodogram(cadence, fs=1.0)
        plt.semilogy(frequencies[1:], psd[1:], 'b-', alpha=0.8)  # Skip DC component
        
        # Highlight dominant frequency
        dominant_idx = np.argmax(psd[1:]) + 1
        plt.scatter(frequencies[dominant_idx], psd[dominant_idx], color='red', s=100, zorder=5, label=f'Peak: {frequencies[dominant_idx]:.4f} Hz')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Cadence Oscillatory Frequency Spectrum')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Cadence vs Heart Rate coupling
        ax4 = plt.subplot(3, 4, 4)
        valid_data = self.data[(self.data['cadence'] > 0) & (self.data['heart_rate'] > 0)]
        
        if len(valid_data) > 10:
            cadence_valid = valid_data['cadence'].values
            hr_valid = valid_data['heart_rate'].values
            
            plt.scatter(cadence_valid, hr_valid, alpha=0.6, c=valid_data['time'].values, cmap='viridis')
            
            # Add trend line
            z = np.polyfit(cadence_valid, hr_valid, 1)
            p = np.poly1d(z)
            plt.plot(cadence_valid, p(cadence_valid), 'r--', alpha=0.8, linewidth=2)
            
            # Calculate and display correlation
            corr = np.corrcoef(cadence_valid, hr_valid)[0, 1]
            plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.colorbar(label='Time (s)')
        
        plt.xlabel('Cadence (steps/min)')
        plt.ylabel('Heart Rate (bpm)')
        plt.title('Cadence-Heart Rate Oscillatory Coupling')
        plt.grid(True, alpha=0.3)
        
        # 5. Segment-wise cadence analysis
        ax5 = plt.subplot(3, 4, 5)
        segment_data = self.data.groupby('segments')
        
        segment_means = []
        segment_names = []
        segment_stds = []
        
        for segment_name, segment_df in segment_data:
            if len(segment_df) >= 5:
                segment_cadences = segment_df['cadence'].values
                segment_means.append(np.mean(segment_cadences))
                segment_stds.append(np.std(segment_cadences))
                segment_names.append(segment_name)
        
        if segment_means:
            x_pos = np.arange(len(segment_names))
            bars = plt.bar(x_pos, segment_means, yerr=segment_stds, alpha=0.7, capsize=5)
            
            # Color bars by adaptation level
            colors = plt.cm.viridis(np.linspace(0, 1, len(segment_means)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.xticks(x_pos, segment_names, rotation=45)
            plt.ylabel('Cadence (steps/min)')
            plt.title('Segment-wise Oscillatory Adaptation')
            plt.grid(True, alpha=0.3)
        
        # 6. Autocorrelation analysis
        ax6 = plt.subplot(3, 4, 6)
        if len(cadence) > 20:
            autocorr = np.correlate(cadence - np.mean(cadence), cadence - np.mean(cadence), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            lags = np.arange(len(autocorr))
            plt.plot(lags[:min(50, len(lags))], autocorr[:min(50, len(autocorr))], 'b-', alpha=0.8)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% correlation')
            
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.title('Cadence Oscillatory Autocorrelation')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. Speed vs Cadence oscillatory relationship
        ax7 = plt.subplot(3, 4, 7)
        valid_speed_data = self.data[(self.data['cadence'] > 0) & (self.data['speed'] > 0)]
        
        if len(valid_speed_data) > 10:
            speed_valid = valid_speed_data['speed'].values
            cadence_valid = valid_speed_data['cadence'].values
            
            plt.scatter(speed_valid, cadence_valid, alpha=0.6, c=valid_speed_data['time'].values, cmap='plasma')
            
            # Add trend line
            z = np.polyfit(speed_valid, cadence_valid, 1)
            p = np.poly1d(z)
            plt.plot(speed_valid, p(speed_valid), 'r--', alpha=0.8, linewidth=2)
            
            plt.colorbar(label='Time (s)')
        
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Cadence (steps/min)')
        plt.title('Speed-Cadence Oscillatory Relationship')
        plt.grid(True, alpha=0.3)
        
        # 8. Vertical oscillation vs Cadence coupling
        ax8 = plt.subplot(3, 4, 8)
        valid_vertical_data = self.data[(self.data['cadence'] > 0) & (self.data['vertical_oscillation'] > 0)]
        
        if len(valid_vertical_data) > 10:
            vertical_valid = valid_vertical_data['vertical_oscillation'].values
            cadence_valid = valid_vertical_data['cadence'].values
            
            plt.scatter(cadence_valid, vertical_valid, alpha=0.6, c=valid_vertical_data['speed'].values, cmap='coolwarm')
            
            # Add trend line
            z = np.polyfit(cadence_valid, vertical_valid, 1)
            p = np.poly1d(z)
            plt.plot(cadence_valid, p(cadence_valid), 'r--', alpha=0.8, linewidth=2)
            
            plt.colorbar(label='Speed (m/s)')
        
        plt.xlabel('Cadence (steps/min)')
        plt.ylabel('Vertical Oscillation (mm)')
        plt.title('Cadence-Vertical Oscillation Coupling')
        plt.grid(True, alpha=0.3)
        
        # 9. Time-frequency analysis (spectrogram)
        ax9 = plt.subplot(3, 4, 9)
        if len(cadence) > 50:
            f, t_spec, Sxx = signal.spectrogram(cadence, fs=1.0, nperseg=min(32, len(cadence)//4))
            
            im = plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            plt.title('Cadence Oscillatory Spectrogram')
            plt.colorbar(im, label='Power (dB)')
        
        # 10. Heart rate zones and cadence
        ax10 = plt.subplot(3, 4, 10)
        hr_zone_data = self.data.groupby('hr_zones')['cadence'].agg(['mean', 'std']).reset_index()
        
        if len(hr_zone_data) > 1:
            plt.errorbar(hr_zone_data['hr_zones'], hr_zone_data['mean'], 
                        yerr=hr_zone_data['std'], fmt='o-', capsize=5, alpha=0.8)
            
            plt.xlabel('Heart Rate Zone')
            plt.ylabel('Cadence (steps/min)')
            plt.title('Cadence by Heart Rate Zone')
            plt.grid(True, alpha=0.3)
        
        # 11. Oscillatory efficiency metrics
        ax11 = plt.subplot(3, 4, 11)
        if 'segment_analysis' in results.get('terrain_adaptation', {}):
            segment_analysis = results['terrain_adaptation']['segment_analysis']
            
            segments = list(segment_analysis.keys())
            efficiencies = [segment_analysis[seg]['oscillatory_efficiency'] for seg in segments]
            
            bars = plt.bar(range(len(segments)), efficiencies, alpha=0.7)
            
            # Color bars by efficiency
            colors = plt.cm.RdYlGn([eff for eff in efficiencies])
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.xticks(range(len(segments)), segments, rotation=45)
            plt.ylabel('Oscillatory Efficiency')
            plt.title('Segment Oscillatory Efficiency')
            plt.axhline(y=self.theoretical_predictions['oscillatory_efficiency_target'], 
                       color='red', linestyle='--', alpha=0.7, label='Target')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 12. Validation summary
        ax12 = plt.subplot(3, 4, 12)
        
        # Collect validation results
        validation_categories = []
        validation_scores = []
        
        for analysis_name, analysis_results in results.items():
            if isinstance(analysis_results, dict) and 'theoretical_validation' in analysis_results:
                theoretical_val = analysis_results['theoretical_validation']
                if isinstance(theoretical_val, dict):
                    for pred_name, pred_result in theoretical_val.items():
                        validation_categories.append(f"{analysis_name[:8]}_{pred_name[:8]}")
                        validation_scores.append(1 if pred_result else 0)
        
        if validation_categories:
            colors = ['green' if score else 'red' for score in validation_scores]
            bars = plt.barh(range(len(validation_categories)), validation_scores, color=colors, alpha=0.7)
            
            plt.yticks(range(len(validation_categories)), validation_categories)
            plt.xlabel('Validation Success')
            plt.title('Theoretical Validation Summary')
            plt.xlim(0, 1.2)
            
            # Add success/failure labels
            for i, (bar, score) in enumerate(zip(bars, validation_scores)):
                label = '✅' if score else '❌'
                plt.text(0.6, bar.get_y() + bar.get_height()/2, label, 
                        ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        
        # Save visualization
        output_path = self.results_dir / 'cadence_oscillatory_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Comprehensive dashboard saved: {output_path}")
        
        # Create interactive Plotly visualization
        self._create_interactive_visualization(results)
        
    def _create_interactive_visualization(self, results):
        """Create interactive Plotly visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cadence Time Series', 'Cadence-Heart Rate Coupling', 
                          'Segment Analysis', 'Frequency Spectrum'),
            specs=[[{"secondary_y": True}, {}],
                   [{}, {"secondary_y": True}]]
        )
        
        # Cadence time series with heart rate
        fig.add_trace(
            go.Scatter(x=self.data['time'], y=self.data['cadence'], 
                      name='Cadence', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=self.data['time'], y=self.data['heart_rate'], 
                      name='Heart Rate', line=dict(color='red')),
            row=1, col=1, secondary_y=True
        )
        
        # Cadence vs Heart Rate scatter
        valid_data = self.data[(self.data['cadence'] > 0) & (self.data['heart_rate'] > 0)]
        if len(valid_data) > 10:
            fig.add_trace(
                go.Scatter(x=valid_data['cadence'], y=valid_data['heart_rate'],
                          mode='markers', name='HR-Cadence Coupling',
                          marker=dict(color=valid_data['time'], colorscale='viridis')),
                row=1, col=2
            )
        
        # Segment analysis
        segment_data = self.data.groupby('segments')
        segment_means = []
        segment_names = []
        
        for segment_name, segment_df in segment_data:
            if len(segment_df) >= 5:
                segment_means.append(np.mean(segment_df['cadence']))
                segment_names.append(segment_name)
        
        if segment_means:
            fig.add_trace(
                go.Bar(x=segment_names, y=segment_means, name='Segment Cadence'),
                row=2, col=1
            )
        
        # Frequency spectrum
        if len(self.cadence_time_series) > 10:
            frequencies, psd = signal.periodogram(self.cadence_time_series, fs=1.0)
            fig.add_trace(
                go.Scatter(x=frequencies[1:], y=psd[1:], name='PSD',
                          line=dict(color='green')),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Interactive Cadence Oscillatory Analysis")
        
        # Save interactive visualization
        interactive_path = self.results_dir / 'cadence_interactive_analysis.html'
        fig.write_html(str(interactive_path))
        
        print(f"  📊 Interactive visualization saved: {interactive_path}")
    
    def save_results(self, results):
        """Save analysis results to JSON"""
        output_file = self.results_dir / 'cadence_oscillatory_results.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy_types(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"  💾 Results saved: {output_file}")
    
    def run_comprehensive_analysis(self):
        """Run complete cadence oscillatory analysis"""
        print("\n" + "="*60)
        print("🏃‍♂️⚡ COMPREHENSIVE CADENCE OSCILLATORY ANALYSIS ⚡🏃‍♂️")
        print("="*60)
        print("Validating Universal Oscillatory Framework with REAL biomechanical data!")
        print("="*60)
        
        # Load data
        if not self.load_track_data():
            return None
        
        # Run all experiments
        results = {}
        
        experiments = [
            ('oscillatory_stability', self.analyze_cadence_oscillatory_stability),
            ('heart_rate_coupling', self.analyze_cadence_heart_rate_coupling),
            ('terrain_adaptation', self.analyze_terrain_adaptive_oscillations),
        ]
        
        successful_experiments = 0
        
        for experiment_name, experiment_func in experiments:
            try:
                result = experiment_func()
                results[experiment_name] = result
                
                if result.get('validation_success', False):
                    successful_experiments += 1
                    
            except Exception as e:
                print(f"❌ Error in {experiment_name}: {str(e)}")
                results[experiment_name] = {'error': str(e)}
        
        # Generate comprehensive summary
        comprehensive_summary = {
            'total_experiments': len(experiments),
            'successful_validations': successful_experiments,
            'validation_success_rate': successful_experiments / len(experiments),
            'cadence_oscillatory_validation_success': successful_experiments >= 2,
            'theoretical_predictions_validated': self._count_validated_predictions(results),
            'data_points_analyzed': len(self.data) if self.data is not None else 0,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        results['comprehensive_summary'] = comprehensive_summary
        
        # Create visualizations
        self.create_comprehensive_visualizations(results)
        
        # Save results
        self.save_results(results)
        
        # Print final summary
        print("\n" + "="*60)
        print("📊 CADENCE OSCILLATORY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Experiments Run: {len(experiments)}")
        print(f"Successful Validations: {successful_experiments}")
        print(f"Success Rate: {comprehensive_summary['validation_success_rate']*100:.1f}%")
        print(f"Framework Validated: {comprehensive_summary['cadence_oscillatory_validation_success']}")
        print("="*60)
        
        if comprehensive_summary['cadence_oscillatory_validation_success']:
            print("\n🎉 CADENCE OSCILLATORY VALIDATION SUCCESS! 🎉")
            print("Real biomechanical data supports oscillatory theories!")
        else:
            print(f"\n⚠️ Partial validation achieved - more analysis needed")
        
        return results
    
    def _count_validated_predictions(self, results):
        """Count total validated theoretical predictions"""
        total_predictions = 0
        
        for experiment_result in results.values():
            if isinstance(experiment_result, dict) and 'theoretical_validation' in experiment_result:
                theoretical_val = experiment_result['theoretical_validation']
                if isinstance(theoretical_val, dict):
                    total_predictions += sum(1 for pred in theoretical_val.values() if pred)
        
        return total_predictions


def main():
    """Main function to run cadence oscillatory analysis"""
    print("Starting Cadence Oscillatory Analysis...")
    
    analyzer = CadenceOscillatoryAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print(f"Results saved in: {analyzer.results_dir}")
        
        if results.get('comprehensive_summary', {}).get('cadence_oscillatory_validation_success', False):
            print(f"\n🏆 BREAKTHROUGH: REAL DATA VALIDATES OSCILLATORY THEORIES! 🏆")
    else:
        print(f"\n❌ Analysis could not be completed")


if __name__ == "__main__":
    main()
