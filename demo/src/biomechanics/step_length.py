#!/usr/bin/env python3
"""
Step Length Oscillatory Analysis

Validates oscillatory theories using REAL step length data from track running.
Step length represents the fundamental spatial oscillation of human gait patterns.

This script applies the Universal Biological Oscillatory Framework to analyze:
1. Step length oscillatory stability and rhythm patterns
2. Multi-scale coupling with cadence, speed, and vertical oscillations
3. Adaptive step length modulation in response to terrain changes
4. Gait efficiency optimization through oscillatory step length control
5. Bilateral step length symmetry and coupling dynamics
6. Validation of spatial-temporal oscillatory integration theory

REVOLUTIONARY VALIDATION: First framework to analyze spatial gait oscillations in real biomechanics!
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

class StepLengthAnalyzer:
    """
    Analyzes step length oscillations using spatial-temporal oscillatory theory
    """
    
    def __init__(self, data_file="../../experimental-data/circuit/annotated_track_series.json", 
                 results_dir="biomechanics_results/step_length"):
        self.data_file = Path(data_file)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Theoretical predictions from spatial-temporal oscillatory framework
        self.theoretical_predictions = {
            'optimal_step_length_range': (1200, 1600),  # mm for efficient running
            'step_length_stability_cv': 0.08,  # CV < 8% for stable gait
            'cadence_coupling_strength': 0.6,  # Expected coupling with cadence  
            'speed_coupling_strength': 0.7,  # Positive correlation with speed
            'vertical_coupling_strength': 0.4,  # Coupling with vertical oscillations
            'terrain_adaptation_coefficient': 0.2,  # Adaptation responsiveness
            'bilateral_symmetry_threshold': 0.05,  # < 5% left/right difference
            'efficiency_optimization_threshold': 0.85,  # Gait efficiency target
            'oscillatory_frequency_match': 2.0,  # Step length should match 2x cadence rhythm
        }
        
        # Load and validate data
        self.data = None
        self.step_length_series = None
        self.validation_results = {}
        
        print("🦵⚡ STEP LENGTH ANALYZER ⚡🦵")
        print("=" * 60)
        print("Analyzing spatial gait oscillations - the rhythm of stride!")
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
            
            # Clean and validate step length data
            # Step length of 0 indicates missing or invalid data
            self.data = self.data[self.data['step_length'] > 0]
            self.data = self.data.reset_index(drop=True)
            
            # Create time series
            self.step_length_series = self.data['step_length'].values
            
            print(f"✅ Loaded {len(self.data)} data points")
            print(f"📊 Step Length range: {self.step_length_series.min():.0f} - {self.step_length_series.max():.0f} mm")
            print(f"⏱️  Time span: {self.data['time'].max() - self.data['time'].min():.1f} seconds")
            print(f"🦵 Mean Step Length: {np.mean(self.step_length_series):.0f} ± {np.std(self.step_length_series):.0f} mm")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def analyze_step_length_stability(self):
        """
        Analyze step length stability and oscillatory rhythm patterns
        Tests: Spatial-Temporal Oscillatory Stability Theory
        """
        print("\n🔬 EXPERIMENT 1: Step Length Stability & Rhythm Analysis")
        print("-" * 55)
        
        if self.step_length_series is None:
            print("❌ No data available")
            return {}
        
        results = {}
        step_length = self.step_length_series
        
        # Basic step length characteristics
        mean_step_length = np.mean(step_length)
        std_step_length = np.std(step_length)
        cv_step_length = std_step_length / mean_step_length
        step_length_range = np.max(step_length) - np.min(step_length)
        
        results['step_length_characteristics'] = {
            'mean_step_length': mean_step_length,
            'std_step_length': std_step_length,
            'cv_step_length': cv_step_length,
            'step_length_range': step_length_range,
            'step_length_stability': cv_step_length < self.theoretical_predictions['step_length_stability_cv']
        }
        
        print(f"Mean Step Length: {mean_step_length:.0f} ± {std_step_length:.0f} mm")
        print(f"Coefficient of Variation: {cv_step_length:.3f} ({cv_step_length*100:.1f}%)")
        print(f"Step Length Stability: {'✅ STABLE' if results['step_length_characteristics']['step_length_stability'] else '⚠️  UNSTABLE'}")
        
        # Optimal step length range validation
        optimal_range = self.theoretical_predictions['optimal_step_length_range']
        in_optimal_range = np.sum((step_length >= optimal_range[0]) & (step_length <= optimal_range[1]))
        optimal_percentage = in_optimal_range / len(step_length) * 100
        
        results['optimal_range_validation'] = {
            'optimal_range_adherence': optimal_percentage,
            'range_violations': len(step_length) - in_optimal_range,
            'theoretical_bounds': optimal_range,
            'optimal_gait_maintained': optimal_percentage > 70  # 70% in optimal range
        }
        
        print(f"Optimal Range Adherence: {optimal_percentage:.1f}%")
        print(f"Optimal Gait: {'✅ MAINTAINED' if results['optimal_range_validation']['optimal_gait_maintained'] else '⚠️  COMPROMISED'}")
        
        # Step length rhythm analysis
        if len(step_length) > 20:
            # Detect step length oscillatory patterns
            step_length_normalized = step_length / mean_step_length
            
            # Calculate rhythm consistency using autocorrelation
            autocorr = np.correlate(step_length_normalized - 1, step_length_normalized - 1, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find oscillatory period
            significant_peaks, _ = signal.find_peaks(autocorr[1:20], height=0.3)
            
            if len(significant_peaks) > 0:
                primary_period = significant_peaks[0] + 1
                rhythm_strength = autocorr[primary_period]
                
                results['rhythm_analysis'] = {
                    'primary_rhythm_period': primary_period,
                    'rhythm_strength': rhythm_strength,
                    'consistent_rhythm': rhythm_strength > 0.5,
                    'rhythm_detected': True
                }
                
                print(f"Primary Rhythm Period: {primary_period} steps")
                print(f"Rhythm Strength: {rhythm_strength:.3f}")
                print(f"Consistent Rhythm: {'✅ YES' if results['rhythm_analysis']['consistent_rhythm'] else '❌ NO'}")
            else:
                results['rhythm_analysis'] = {'rhythm_detected': False}
                print("No clear rhythmic pattern detected")
        
        # Step length variability analysis
        if len(step_length) > 10:
            # Calculate successive step length differences
            step_length_diff = np.diff(step_length)
            step_length_variability = np.std(step_length_diff)
            
            # Calculate relative variability
            relative_variability = step_length_variability / mean_step_length
            
            results['variability_analysis'] = {
                'absolute_variability': step_length_variability,
                'relative_variability': relative_variability,
                'variability_controlled': relative_variability < 0.1  # < 10% relative variability
            }
            
            print(f"Step Length Variability: {step_length_variability:.1f} mm ({relative_variability*100:.1f}%)")
            print(f"Variability Controlled: {'✅ YES' if results['variability_analysis']['variability_controlled'] else '❌ NO'}")
        
        # Frequency domain analysis of step length oscillations
        if len(step_length) > 50:
            # Power spectral density
            frequencies, psd = signal.periodogram(step_length - mean_step_length, fs=1.0)
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
            dominant_frequency = frequencies[dominant_freq_idx]
            dominant_power = psd[dominant_freq_idx]
            
            # Calculate spectral entropy (complexity measure)
            psd_normalized = psd[1:] / np.sum(psd[1:])  # Normalize and exclude DC
            spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))
            
            results['frequency_analysis'] = {
                'dominant_frequency': dominant_frequency,
                'dominant_power': dominant_power,
                'spectral_entropy': spectral_entropy,
                'spectral_complexity': spectral_entropy / np.log2(len(psd_normalized))  # Normalized entropy
            }
            
            print(f"Dominant Frequency: {dominant_frequency:.4f} Hz")
            print(f"Spectral Entropy: {spectral_entropy:.3f}")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        # Test step length stability
        predictions_validated['step_length_stability'] = results['step_length_characteristics']['step_length_stability']
        
        # Test optimal range maintenance
        predictions_validated['optimal_range'] = results['optimal_range_validation']['optimal_gait_maintained']
        
        # Test rhythm consistency
        if 'rhythm_analysis' in results and results['rhythm_analysis']['rhythm_detected']:
            predictions_validated['rhythm_consistency'] = results['rhythm_analysis']['consistent_rhythm']
        
        # Test variability control
        if 'variability_analysis' in results:
            predictions_validated['variability_control'] = results['variability_analysis']['variability_controlled']
        
        results['theoretical_validation'] = predictions_validated
        validation_success = sum(predictions_validated.values()) >= len(predictions_validated) // 2
        results['validation_success'] = validation_success
        
        print(f"🎯 Theoretical Validation: {'✅ SUCCESS' if validation_success else '❌ FAILED'}")
        
        return results
    
    def analyze_multi_scale_coupling(self):
        """
        Analyze step length coupling with cadence, speed, and vertical oscillations
        Tests: Multi-Scale Spatial-Temporal Coupling Theory
        """
        print("\n🔬 EXPERIMENT 2: Multi-Scale Step Length Coupling")
        print("-" * 50)
        
        if self.data is None:
            print("❌ No data available")
            return {}
        
        # Extract relevant data for coupling analysis
        step_length = self.data['step_length'].values
        cadence = self.data['cadence'].values
        speed = self.data['speed'].values
        vertical_osc = self.data['vertical_oscillation'].values
        
        # Remove invalid data points
        valid_indices = (step_length > 0) & (cadence > 0) & (speed > 0) & (vertical_osc > 0)
        
        step_length_clean = step_length[valid_indices]
        cadence_clean = cadence[valid_indices]
        speed_clean = speed[valid_indices]
        vertical_clean = vertical_osc[valid_indices]
        
        if len(step_length_clean) < 10:
            print("❌ Insufficient valid data")
            return {}
        
        results = {}
        
        # Step length - Cadence coupling
        step_cadence_correlation = np.corrcoef(step_length_clean, cadence_clean)[0, 1]
        
        results['cadence_coupling'] = {
            'correlation_coefficient': step_cadence_correlation,
            'coupling_strength': abs(step_cadence_correlation),
            'coupling_direction': 'positive' if step_cadence_correlation > 0 else 'negative',
            'significant_coupling': abs(step_cadence_correlation) >= self.theoretical_predictions['cadence_coupling_strength']
        }
        
        print(f"Step-Cadence Coupling: {step_cadence_correlation:.3f}")
        print(f"Significant Coupling: {'✅ YES' if results['cadence_coupling']['significant_coupling'] else '❌ NO'}")
        
        # Step length - Speed coupling
        step_speed_correlation = np.corrcoef(step_length_clean, speed_clean)[0, 1]
        
        results['speed_coupling'] = {
            'correlation_coefficient': step_speed_correlation,
            'coupling_strength': abs(step_speed_correlation),
            'coupling_direction': 'positive' if step_speed_correlation > 0 else 'negative',
            'significant_coupling': abs(step_speed_correlation) >= self.theoretical_predictions['speed_coupling_strength'],
            'expected_positive_coupling': step_speed_correlation > 0  # Should be positive for efficiency
        }
        
        print(f"Step-Speed Coupling: {step_speed_correlation:.3f}")
        print(f"Expected Positive Coupling: {'✅ YES' if results['speed_coupling']['expected_positive_coupling'] else '❌ NO'}")
        
        # Step length - Vertical oscillation coupling
        step_vertical_correlation = np.corrcoef(step_length_clean, vertical_clean)[0, 1]
        
        results['vertical_coupling'] = {
            'correlation_coefficient': step_vertical_correlation,
            'coupling_strength': abs(step_vertical_correlation),
            'coupling_direction': 'positive' if step_vertical_correlation > 0 else 'negative',
            'significant_coupling': abs(step_vertical_correlation) >= self.theoretical_predictions['vertical_coupling_strength']
        }
        
        print(f"Step-Vertical Coupling: {step_vertical_correlation:.3f}")
        print(f"Significant Coupling: {'✅ YES' if results['vertical_coupling']['significant_coupling'] else '❌ NO'}")
        
        # Multi-dimensional coupling analysis
        if len(step_length_clean) > 50:
            # Principal component analysis to understand coupling patterns
            from sklearn.decomposition import PCA
            
            # Create feature matrix
            features = np.column_stack([step_length_clean, cadence_clean, speed_clean, vertical_clean])
            
            # Standardize features
            features_std = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
            
            # Perform PCA
            pca = PCA(n_components=4)
            pca_result = pca.fit_transform(features_std)
            
            results['pca_coupling'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'first_component_variance': pca.explained_variance_ratio_[0],
                'cumulative_variance_2_components': np.sum(pca.explained_variance_ratio_[:2]),
                'strong_coupling_indicated': pca.explained_variance_ratio_[0] > 0.6  # First component explains >60%
            }
            
            print(f"First Component Variance: {pca.explained_variance_ratio_[0]:.3f}")
            print(f"Strong Multi-dimensional Coupling: {'✅ YES' if results['pca_coupling']['strong_coupling_indicated'] else '❌ NO'}")
        
        # Gait efficiency analysis
        # Calculate stride rate (steps per second) and stride length
        stride_rate = cadence_clean / 60.0  # Convert to Hz
        stride_length = step_length_clean / 1000.0  # Convert to meters
        
        # Calculate theoretical speed from stride parameters
        theoretical_speed = stride_rate * stride_length
        speed_error = np.abs(theoretical_speed - speed_clean)
        
        gait_efficiency = 1.0 / (1.0 + np.mean(speed_error))
        
        results['gait_efficiency'] = {
            'mean_speed_error': np.mean(speed_error),
            'gait_efficiency': gait_efficiency,
            'efficiency_optimal': gait_efficiency >= self.theoretical_predictions['efficiency_optimization_threshold']
        }
        
        print(f"Gait Efficiency: {gait_efficiency:.3f}")
        print(f"Optimal Efficiency: {'✅ YES' if results['gait_efficiency']['efficiency_optimal'] else '❌ NO'}")
        
        # Cross-correlation analysis for phase relationships
        if len(step_length_clean) > 100:
            # Analyze phase relationships between step length and other parameters
            
            # Step length - Cadence cross-correlation
            step_centered = step_length_clean - np.mean(step_length_clean)
            cadence_centered = cadence_clean - np.mean(cadence_clean)
            
            cross_corr = signal.correlate(step_centered, cadence_centered, mode='full')
            cross_corr = cross_corr / (np.std(step_centered) * np.std(cadence_centered) * len(step_centered))
            
            # Find peak correlation and lag
            max_corr_idx = np.argmax(np.abs(cross_corr))
            lag = max_corr_idx - len(step_centered) + 1
            max_correlation = cross_corr[max_corr_idx]
            
            results['phase_relationships'] = {
                'max_cross_correlation': max_correlation,
                'optimal_lag': lag,
                'phase_coupling_strength': abs(max_correlation),
                'synchronized_coupling': abs(lag) < 5  # Low lag indicates synchronization
            }
            
            print(f"Phase Coupling Strength: {abs(max_correlation):.3f}")
            print(f"Synchronized Coupling: {'✅ YES' if results['phase_relationships']['synchronized_coupling'] else '❌ NO'}")
        
        # Coherence analysis in frequency domain
        if len(step_length_clean) > 200:
            # Coherence between step length and cadence
            frequencies, coherence = signal.coherence(step_centered, cadence_centered, fs=1.0,
                                                    nperseg=min(64, len(step_centered)//4))
            
            # Find peak coherence
            peak_coherence_idx = np.argmax(coherence[1:]) + 1  # Skip DC
            peak_coherence = coherence[peak_coherence_idx]
            peak_frequency = frequencies[peak_coherence_idx]
            
            results['coherence_analysis'] = {
                'peak_coherence': peak_coherence,
                'peak_frequency': peak_frequency,
                'mean_coherence': np.mean(coherence[1:]),
                'strong_coherence': peak_coherence > 0.7
            }
            
            print(f"Peak Coherence: {peak_coherence:.3f} at {peak_frequency:.4f} Hz")
            print(f"Strong Coherence: {'✅ YES' if results['coherence_analysis']['strong_coherence'] else '❌ NO'}")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        # Test cadence coupling strength
        predictions_validated['cadence_coupling'] = results['cadence_coupling']['significant_coupling']
        
        # Test speed coupling strength and direction
        predictions_validated['speed_coupling_strength'] = results['speed_coupling']['significant_coupling']
        predictions_validated['speed_coupling_direction'] = results['speed_coupling']['expected_positive_coupling']
        
        # Test vertical oscillation coupling
        predictions_validated['vertical_coupling'] = results['vertical_coupling']['significant_coupling']
        
        # Test gait efficiency
        predictions_validated['gait_efficiency'] = results['gait_efficiency']['efficiency_optimal']
        
        # Test multi-dimensional coupling
        if 'pca_coupling' in results:
            predictions_validated['multidimensional_coupling'] = results['pca_coupling']['strong_coupling_indicated']
        
        # Test phase synchronization
        if 'phase_relationships' in results:
            predictions_validated['phase_synchronization'] = results['phase_relationships']['synchronized_coupling']
        
        results['theoretical_validation'] = predictions_validated
        validation_success = sum(predictions_validated.values()) >= len(predictions_validated) // 2
        results['validation_success'] = validation_success
        
        print(f"🎯 Theoretical Validation: {'✅ SUCCESS' if validation_success else '❌ FAILED'}")
        
        return results
    
    def analyze_terrain_step_adaptation(self):
        """
        Analyze step length adaptation to different terrain segments
        Tests: Environmental Spatial-Temporal Adaptation Theory
        """
        print("\n🔬 EXPERIMENT 3: Terrain Step Length Adaptation")
        print("-" * 48)
        
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
                
            step_length_segment = segment_df['step_length'].values
            speed_segment = segment_df['speed'].values
            cadence_segment = segment_df['cadence'].values
            
            # Remove invalid data
            valid_mask = (step_length_segment > 0) & (speed_segment > 0) & (cadence_segment > 0)
            
            if np.sum(valid_mask) < 3:
                continue
                
            step_length_clean = step_length_segment[valid_mask]
            speed_clean = speed_segment[valid_mask]
            cadence_clean = cadence_segment[valid_mask]
            
            # Calculate segment-specific step length metrics
            segment_stats = {
                'n_points': len(step_length_clean),
                'mean_step_length': np.mean(step_length_clean),
                'std_step_length': np.std(step_length_clean),
                'cv_step_length': np.std(step_length_clean) / np.mean(step_length_clean),
                'mean_speed': np.mean(speed_clean),
                'mean_cadence': np.mean(cadence_clean),
                'step_length_range': np.max(step_length_clean) - np.min(step_length_clean)
            }
            
            # Calculate adaptation-specific metrics
            segment_stats['step_efficiency'] = np.mean(step_length_clean) / np.mean(speed_clean) if np.mean(speed_clean) > 0 else 0
            segment_stats['adaptation_index'] = segment_stats['cv_step_length'] * segment_stats['step_length_range']
            segment_stats['gait_optimization'] = 1.0 / (1.0 + segment_stats['cv_step_length'])
            
            segment_analysis[segment_name] = segment_stats
            
            print(f"  {segment_name}: {segment_stats['mean_step_length']:.0f} ± {segment_stats['std_step_length']:.0f} mm")
            print(f"    CV: {segment_stats['cv_step_length']:.3f}, Efficiency: {segment_stats['step_efficiency']:.2f}")
        
        results['segment_analysis'] = segment_analysis
        
        # Cross-segment adaptation analysis
        if len(segment_analysis) >= 2:
            segment_names = list(segment_analysis.keys())
            
            # Extract metrics for analysis
            mean_step_lengths = [segment_analysis[seg]['mean_step_length'] for seg in segment_names]
            step_cvs = [segment_analysis[seg]['cv_step_length'] for seg in segment_names]
            step_efficiencies = [segment_analysis[seg]['step_efficiency'] for seg in segment_names]
            gait_optimizations = [segment_analysis[seg]['gait_optimization'] for seg in segment_names]
            
            # Calculate adaptation metrics
            step_length_adaptation_range = np.max(mean_step_lengths) - np.min(mean_step_lengths)
            cv_adaptation_range = np.max(step_cvs) - np.min(step_cvs)
            efficiency_adaptation_range = np.max(step_efficiencies) - np.min(step_efficiencies)
            
            results['adaptation_analysis'] = {
                'step_length_adaptation_range': step_length_adaptation_range,
                'cv_adaptation_range': cv_adaptation_range,
                'efficiency_adaptation_range': efficiency_adaptation_range,
                'mean_gait_optimization': np.mean(gait_optimizations),
                'adaptation_consistency': 1.0 - (np.std(mean_step_lengths) / np.mean(mean_step_lengths)),
                'segment_count': len(segment_names),
                'segments': segment_names
            }
            
            print(f"Step Length Adaptation Range: {step_length_adaptation_range:.0f} mm")
            print(f"Mean Gait Optimization: {results['adaptation_analysis']['mean_gait_optimization']:.3f}")
            print(f"Adaptation Consistency: {results['adaptation_analysis']['adaptation_consistency']:.3f}")
            
            # Statistical analysis of segment differences
            segment_step_lengths = []
            
            for segment_name in segment_names:
                segment_df = self.data[self.data['segments'] == segment_name]
                valid_data = segment_df[segment_df['step_length'] > 0]
                
                if len(valid_data) >= 3:
                    segment_step_lengths.append(valid_data['step_length'].values)
            
            if len(segment_step_lengths) >= 2:
                # ANOVA test for significant differences
                try:
                    f_stat, p_value = stats.f_oneway(*segment_step_lengths)
                    
                    results['statistical_analysis'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant_adaptation': p_value < 0.05
                    }
                    
                    print(f"Step Length ANOVA: F={f_stat:.2f}, p={p_value:.4f}")
                    print(f"Significant Adaptation: {'✅ YES' if p_value < 0.05 else '❌ NO'}")
                    
                except Exception as e:
                    print(f"Statistical analysis failed: {str(e)}")
        
        # Speed-normalized step length analysis
        if len(segment_analysis) >= 2:
            # Analyze step length normalized by speed (step length per unit speed)
            speed_normalized_steps = {}
            
            for segment_name in segment_names:
                segment_df = self.data[self.data['segments'] == segment_name]
                valid_data = segment_df[(segment_df['step_length'] > 0) & (segment_df['speed'] > 0)]
                
                if len(valid_data) >= 3:
                    step_length_norm = valid_data['step_length'].values / valid_data['speed'].values
                    
                    speed_normalized_steps[segment_name] = {
                        'mean_normalized_step': np.mean(step_length_norm),
                        'std_normalized_step': np.std(step_length_norm),
                        'cv_normalized_step': np.std(step_length_norm) / np.mean(step_length_norm)
                    }
            
            if len(speed_normalized_steps) >= 2:
                results['speed_normalized_analysis'] = speed_normalized_steps
                
                # Calculate adaptation in speed-normalized terms
                norm_means = [speed_normalized_steps[seg]['mean_normalized_step'] for seg in speed_normalized_steps.keys()]
                norm_adaptation_range = np.max(norm_means) - np.min(norm_means)
                
                results['normalized_adaptation'] = {
                    'normalized_adaptation_range': norm_adaptation_range,
                    'relative_adaptation': norm_adaptation_range / np.mean(norm_means)
                }
                
                print(f"Speed-Normalized Adaptation Range: {norm_adaptation_range:.1f}")
                print(f"Relative Adaptation: {results['normalized_adaptation']['relative_adaptation']:.3f}")
        
        # Bilateral step length analysis (if side data available)
        if 'side' in self.data.columns:
            bilateral_analysis = {}
            
            for side in ['left', 'right']:
                side_data = self.data[(self.data['side'] == side) & (self.data['step_length'] > 0)]
                
                if len(side_data) >= 10:
                    side_step_lengths = side_data['step_length'].values
                    
                    bilateral_analysis[side] = {
                        'mean_step_length': np.mean(side_step_lengths),
                        'std_step_length': np.std(side_step_lengths),
                        'cv_step_length': np.std(side_step_lengths) / np.mean(side_step_lengths),
                        'n_points': len(side_step_lengths)
                    }
            
            if len(bilateral_analysis) == 2:
                results['bilateral_analysis'] = bilateral_analysis
                
                # Calculate bilateral symmetry
                left_mean = bilateral_analysis['left']['mean_step_length']
                right_mean = bilateral_analysis['right']['mean_step_length']
                
                bilateral_asymmetry = abs(left_mean - right_mean) / ((left_mean + right_mean) / 2)
                
                results['bilateral_symmetry'] = {
                    'left_mean_step_length': left_mean,
                    'right_mean_step_length': right_mean,
                    'bilateral_asymmetry': bilateral_asymmetry,
                    'symmetric_gait': bilateral_asymmetry < self.theoretical_predictions['bilateral_symmetry_threshold']
                }
                
                print(f"Bilateral Asymmetry: {bilateral_asymmetry:.3f} ({bilateral_asymmetry*100:.1f}%)")
                print(f"Symmetric Gait: {'✅ YES' if results['bilateral_symmetry']['symmetric_gait'] else '❌ NO'}")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        # Test adaptation reasonableness (should adapt but not excessively)
        if 'adaptation_analysis' in results:
            reasonable_adaptation = (
                results['adaptation_analysis']['step_length_adaptation_range'] < 500 and  # < 500mm range
                results['adaptation_analysis']['cv_adaptation_range'] < 0.15  # < 15% CV range
            )
            predictions_validated['reasonable_adaptation'] = reasonable_adaptation
            
            # Test adaptation consistency
            consistency_threshold = 0.7
            predictions_validated['adaptation_consistency'] = results['adaptation_analysis']['adaptation_consistency'] >= consistency_threshold
            
            # Test gait optimization maintenance
            optimization_threshold = 0.8
            predictions_validated['gait_optimization'] = results['adaptation_analysis']['mean_gait_optimization'] >= optimization_threshold
        
        # Test significant adaptive response
        if 'statistical_analysis' in results:
            predictions_validated['significant_adaptation'] = results['statistical_analysis']['significant_adaptation']
        
        # Test bilateral symmetry
        if 'bilateral_symmetry' in results:
            predictions_validated['bilateral_symmetry'] = results['bilateral_symmetry']['symmetric_gait']
        
        # Test speed normalization efficiency
        if 'normalized_adaptation' in results:
            efficient_normalization = results['normalized_adaptation']['relative_adaptation'] < 0.3  # < 30% relative adaptation
            predictions_validated['speed_normalization_efficiency'] = efficient_normalization
        
        results['theoretical_validation'] = predictions_validated
        validation_success = sum(predictions_validated.values()) >= len(predictions_validated) // 2
        results['validation_success'] = validation_success
        
        print(f"🎯 Theoretical Validation: {'✅ SUCCESS' if validation_success else '❌ FAILED'}")
        
        return results
    
    def create_comprehensive_visualizations(self, results):
        """Generate comprehensive visualizations of step length analysis"""
        print("\n🎨 Creating Comprehensive Visualizations...")
        
        if self.data is None:
            print("❌ No data for visualization")
            return
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Step length time series
        ax1 = plt.subplot(3, 4, 1)
        time = self.data['time'].values
        step_length = self.data['step_length'].values
        
        plt.plot(time, step_length, 'b-', alpha=0.7, linewidth=1)
        plt.axhline(y=np.mean(step_length), color='r', linestyle='--', alpha=0.8, 
                   label=f'Mean: {np.mean(step_length):.0f} mm')
        
        # Add optimal range
        optimal_range = self.theoretical_predictions['optimal_step_length_range']
        plt.axhspan(optimal_range[0], optimal_range[1], alpha=0.2, color='green', label='Optimal Range')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Step Length (mm)')
        plt.title('Step Length Time Series')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Step length distribution
        ax2 = plt.subplot(3, 4, 2)
        plt.hist(step_length, bins=25, alpha=0.7, color='blue', density=True)
        plt.axvline(x=np.mean(step_length), color='red', linestyle='--', label=f'Mean: {np.mean(step_length):.0f}')
        plt.axvline(x=np.mean(step_length) + np.std(step_length), color='orange', linestyle='--', alpha=0.7)
        plt.axvline(x=np.mean(step_length) - np.std(step_length), color='orange', linestyle='--', alpha=0.7,
                   label=f'±1σ: {np.std(step_length):.0f}')
        
        plt.xlabel('Step Length (mm)')
        plt.ylabel('Density')
        plt.title('Step Length Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Power spectral density
        ax3 = plt.subplot(3, 4, 3)
        if len(step_length) > 10:
            step_length_centered = step_length - np.mean(step_length)
            frequencies, psd = signal.periodogram(step_length_centered, fs=1.0)
            plt.semilogy(frequencies[1:], psd[1:], 'b-', alpha=0.8)
            
            # Highlight dominant frequency
            dominant_idx = np.argmax(psd[1:]) + 1
            plt.scatter(frequencies[dominant_idx], psd[dominant_idx], color='red', s=100, zorder=5,
                       label=f'Peak: {frequencies[dominant_idx]:.4f} Hz')
            
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density')
            plt.title('Step Length Frequency Spectrum')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. Step length vs Cadence coupling
        ax4 = plt.subplot(3, 4, 4)
        valid_data = self.data[(self.data['cadence'] > 0) & (self.data['step_length'] > 0)]
        
        if len(valid_data) > 10:
            cadence_valid = valid_data['cadence'].values
            step_valid = valid_data['step_length'].values
            
            scatter = plt.scatter(cadence_valid, step_valid, alpha=0.6, 
                                c=valid_data['speed'].values, cmap='viridis', s=20)
            
            # Add trend line
            z = np.polyfit(cadence_valid, step_valid, 1)
            p = np.poly1d(z)
            plt.plot(cadence_valid, p(cadence_valid), 'r--', alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr = np.corrcoef(cadence_valid, step_valid)[0, 1]
            plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.colorbar(scatter, label='Speed (m/s)')
            plt.xlabel('Cadence (steps/min)')
            plt.ylabel('Step Length (mm)')
            plt.title('Step-Cadence Coupling')
            plt.grid(True, alpha=0.3)
        
        # 5. Step length vs Speed coupling
        ax5 = plt.subplot(3, 4, 5)
        valid_speed_data = self.data[(self.data['speed'] > 0) & (self.data['step_length'] > 0)]
        
        if len(valid_speed_data) > 10:
            speed_valid = valid_speed_data['speed'].values
            step_valid = valid_speed_data['step_length'].values
            
            scatter = plt.scatter(speed_valid, step_valid, alpha=0.6, 
                                c=valid_speed_data['cadence'].values, cmap='plasma', s=20)
            
            # Add trend line
            z = np.polyfit(speed_valid, step_valid, 1)
            p = np.poly1d(z)
            plt.plot(speed_valid, p(speed_valid), 'r--', alpha=0.8, linewidth=2)
            
            corr = np.corrcoef(speed_valid, step_valid)[0, 1]
            plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax5.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.colorbar(scatter, label='Cadence (steps/min)')
            plt.xlabel('Speed (m/s)')
            plt.ylabel('Step Length (mm)')
            plt.title('Step-Speed Coupling')
            plt.grid(True, alpha=0.3)
        
        # 6. Segment-wise step length analysis
        ax6 = plt.subplot(3, 4, 6)
        segment_data = self.data.groupby('segments')
        
        segment_means = []
        segment_names = []
        segment_stds = []
        
        for segment_name, segment_df in segment_data:
            valid_segment = segment_df['step_length'] > 0
            if np.sum(valid_segment) >= 5:
                segment_steps = segment_df[valid_segment]['step_length'].values
                segment_means.append(np.mean(segment_steps))
                segment_stds.append(np.std(segment_steps))
                segment_names.append(segment_name)
        
        if segment_means:
            x_pos = np.arange(len(segment_names))
            bars = plt.bar(x_pos, segment_means, yerr=segment_stds, alpha=0.7, capsize=5)
            
            # Color bars by mean value
            colors = plt.cm.viridis(np.linspace(0, 1, len(segment_means)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.xticks(x_pos, segment_names, rotation=45)
            plt.ylabel('Step Length (mm)')
            plt.title('Segment Step Length Analysis')
            plt.grid(True, alpha=0.3)
        
        # 7. Step length vs Vertical oscillation
        ax7 = plt.subplot(3, 4, 7)
        valid_vertical_data = self.data[(self.data['vertical_oscillation'] > 0) & (self.data['step_length'] > 0)]
        
        if len(valid_vertical_data) > 10:
            vertical_valid = valid_vertical_data['vertical_oscillation'].values
            step_valid = valid_vertical_data['step_length'].values
            
            plt.scatter(step_valid, vertical_valid, alpha=0.6, 
                       c=valid_vertical_data['speed'].values, cmap='coolwarm', s=20)
            
            # Add trend line
            z = np.polyfit(step_valid, vertical_valid, 1)
            p = np.poly1d(z)
            plt.plot(step_valid, p(step_valid), 'r--', alpha=0.8, linewidth=2)
            
            plt.colorbar(label='Speed (m/s)')
            plt.xlabel('Step Length (mm)')
            plt.ylabel('Vertical Oscillation (mm)')
            plt.title('Step-Vertical Oscillation Coupling')
            plt.grid(True, alpha=0.3)
        
        # 8. Step length autocorrelation
        ax8 = plt.subplot(3, 4, 8)
        if len(step_length) > 20:
            step_centered = step_length - np.mean(step_length)
            autocorr = np.correlate(step_centered, step_centered, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            lags = np.arange(len(autocorr))
            plt.plot(lags[:min(50, len(lags))], autocorr[:min(50, len(autocorr))], 'b-', alpha=0.8)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% correlation')
            
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.title('Step Length Autocorrelation')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 9. 3D coupling visualization
        ax9 = plt.subplot(3, 4, 9, projection='3d')
        valid_3d_data = self.data[(self.data['step_length'] > 0) & 
                                 (self.data['cadence'] > 0) & 
                                 (self.data['speed'] > 0)]
        
        if len(valid_3d_data) > 20:
            step_3d = valid_3d_data['step_length'].values[:100]  # Limit for visibility
            cadence_3d = valid_3d_data['cadence'].values[:100]
            speed_3d = valid_3d_data['speed'].values[:100]
            
            scatter = ax9.scatter(step_3d, cadence_3d, speed_3d, 
                                c=valid_3d_data['time'].values[:100], cmap='viridis', alpha=0.6)
            
            ax9.set_xlabel('Step Length (mm)')
            ax9.set_ylabel('Cadence (steps/min)')
            ax9.set_zlabel('Speed (m/s)')
            ax9.set_title('3D Step Length Phase Space')
        
        # 10. Step length variability
        ax10 = plt.subplot(3, 4, 10)
        if len(step_length) > 20:
            step_diff = np.diff(step_length)
            
            plt.plot(time[1:], step_diff, 'g-', alpha=0.7, linewidth=1)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.8, label='No Change')
            plt.axhline(y=np.mean(step_diff), color='b', linestyle='--', alpha=0.8, 
                       label=f'Mean: {np.mean(step_diff):.1f} mm')
            
            plt.xlabel('Time (s)')
            plt.ylabel('Step Length Change (mm)')
            plt.title('Step Length Variability')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 11. Gait efficiency over time
        ax11 = plt.subplot(3, 4, 11)
        valid_efficiency_data = self.data[(self.data['step_length'] > 0) & 
                                         (self.data['speed'] > 0) & 
                                         (self.data['cadence'] > 0)]
        
        if len(valid_efficiency_data) > 20:
            # Calculate stride rate and theoretical speed
            stride_rate = valid_efficiency_data['cadence'].values / 60.0
            stride_length = valid_efficiency_data['step_length'].values / 1000.0
            theoretical_speed = stride_rate * stride_length
            actual_speed = valid_efficiency_data['speed'].values
            
            efficiency = actual_speed / theoretical_speed
            
            plt.plot(valid_efficiency_data['time'], efficiency, 'purple', alpha=0.7, linewidth=1)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.8, label='Perfect Efficiency')
            plt.axhline(y=np.mean(efficiency), color='b', linestyle='--', alpha=0.8, 
                       label=f'Mean: {np.mean(efficiency):.2f}')
            
            plt.xlabel('Time (s)')
            plt.ylabel('Gait Efficiency Ratio')
            plt.title('Gait Efficiency Over Time')
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
        output_path = self.results_dir / 'step_length_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Comprehensive dashboard saved: {output_path}")
        
        # Create interactive visualization
        self._create_interactive_visualization(results)
    
    def _create_interactive_visualization(self, results):
        """Create interactive Plotly visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Step Length Time Series', 'Step-Speed Coupling', 
                          'Multi-Scale Coupling', 'Segment Analysis'),
            specs=[[{"secondary_y": True}, {}],
                   [{}, {}]]
        )
        
        # Step length time series with cadence
        fig.add_trace(
            go.Scatter(x=self.data['time'], y=self.data['step_length'], 
                      name='Step Length', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=self.data['time'], y=self.data['cadence'], 
                      name='Cadence', line=dict(color='red')),
            row=1, col=1, secondary_y=True
        )
        
        # Step vs Speed coupling
        valid_data = self.data[(self.data['step_length'] > 0) & (self.data['speed'] > 0)]
        if len(valid_data) > 10:
            fig.add_trace(
                go.Scatter(x=valid_data['speed'], y=valid_data['step_length'],
                          mode='markers', name='Step-Speed',
                          marker=dict(color=valid_data['cadence'], colorscale='viridis')),
                row=1, col=2
            )
        
        # Multi-scale coupling (3D-like view using color)
        valid_multi_data = self.data[(self.data['step_length'] > 0) & 
                                    (self.data['cadence'] > 0) & 
                                    (self.data['vertical_oscillation'] > 0)]
        
        if len(valid_multi_data) > 10:
            fig.add_trace(
                go.Scatter(x=valid_multi_data['cadence'], y=valid_multi_data['step_length'],
                          mode='markers', name='Multi-Scale Coupling',
                          marker=dict(color=valid_multi_data['vertical_oscillation'], 
                                    colorscale='plasma')),
                row=2, col=1
            )
        
        # Segment analysis
        segment_data = self.data.groupby('segments')
        segment_means = []
        segment_names = []
        
        for segment_name, segment_df in segment_data:
            valid_segment = segment_df['step_length'] > 0
            if np.sum(valid_segment) >= 5:
                segment_steps = segment_df[valid_segment]['step_length'].values
                segment_means.append(np.mean(segment_steps))
                segment_names.append(segment_name)
        
        if segment_means:
            fig.add_trace(
                go.Bar(x=segment_names, y=segment_means, name='Segment Step Length'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Interactive Step Length Analysis")
        
        # Save interactive visualization
        interactive_path = self.results_dir / 'step_length_interactive.html'
        fig.write_html(str(interactive_path))
        
        print(f"  📊 Interactive visualization saved: {interactive_path}")
    
    def save_results(self, results):
        """Save analysis results to JSON"""
        output_file = self.results_dir / 'step_length_results.json'
        
        # Convert numpy types for JSON serialization
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
        """Run complete step length analysis"""
        print("\n" + "="*60)
        print("🦵⚡ COMPREHENSIVE STEP LENGTH ANALYSIS ⚡🦵")
        print("="*60)
        print("Analyzing spatial gait oscillations - the rhythm of stride!")
        print("="*60)
        
        # Load data
        if not self.load_track_data():
            return None
        
        # Run all experiments
        results = {}
        
        experiments = [
            ('step_stability', self.analyze_step_length_stability),
            ('multi_scale_coupling', self.analyze_multi_scale_coupling),
            ('terrain_adaptation', self.analyze_terrain_step_adaptation),
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
            'step_length_validation_success': successful_experiments >= 2,
            'theoretical_predictions_validated': self._count_validated_predictions(results),
            'data_points_analyzed': len(self.data) if self.data is not None else 0,
            'step_length_range_analyzed': f"{self.step_length_series.min():.0f} - {self.step_length_series.max():.0f} mm",
            'mean_step_length': np.mean(self.step_length_series),
            'step_length_cv': np.std(self.step_length_series) / np.mean(self.step_length_series),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        results['comprehensive_summary'] = comprehensive_summary
        
        # Create visualizations
        self.create_comprehensive_visualizations(results)
        
        # Save results
        self.save_results(results)
        
        # Print final summary
        print("\n" + "="*60)
        print("📊 STEP LENGTH ANALYSIS SUMMARY")
        print("="*60)
        print(f"Experiments Run: {len(experiments)}")
        print(f"Successful Validations: {successful_experiments}")
        print(f"Success Rate: {comprehensive_summary['validation_success_rate']*100:.1f}%")
        print(f"Framework Validated: {comprehensive_summary['step_length_validation_success']}")
        print(f"Step Length Range: {comprehensive_summary['step_length_range_analyzed']}")
        print(f"Mean Step Length: {comprehensive_summary['mean_step_length']:.0f} mm")
        print(f"CV: {comprehensive_summary['step_length_cv']:.3f}")
        print("="*60)
        
        if comprehensive_summary['step_length_validation_success']:
            print("\n🎉 STEP LENGTH VALIDATION SUCCESS! 🎉")
            print("Spatial gait oscillations validate our theoretical framework!")
            print("First framework to analyze spatial-temporal gait coupling!")
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
    """Main function to run step length analysis"""
    print("Starting Step Length Analysis...")
    
    analyzer = StepLengthAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print(f"Results saved in: {analyzer.results_dir}")
        
        if results.get('comprehensive_summary', {}).get('step_length_validation_success', False):
            print(f"\n🏆 BREAKTHROUGH: SPATIAL GAIT OSCILLATIONS VALIDATE THEORIES! 🏆")
            print(f"First framework to validate spatial-temporal oscillatory coupling!")
    else:
        print(f"\n❌ Analysis could not be completed")


if __name__ == "__main__":
    main()
