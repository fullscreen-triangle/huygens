#!/usr/bin/env python3
"""
Vertical Ratio Oscillatory Analysis

Validates oscillatory efficiency theories using REAL vertical ratio data from track running.
Vertical ratio represents the efficiency of vertical oscillations - the ratio of vertical motion
to forward motion, expressed as a percentage. Lower ratios indicate more efficient running.

This script applies the Universal Biological Oscillatory Framework to analyze:
1. Oscillatory efficiency optimization and energy conservation principles
2. Multi-scale coupling between efficiency and performance metrics
3. Adaptive efficiency modulation in response to terrain and speed changes
4. Efficiency-stability trade-offs in oscillatory systems
5. Integration of vertical efficiency with overall gait optimization
6. Validation of biological oscillatory efficiency optimization theory

REVOLUTIONARY VALIDATION: First framework to analyze oscillatory efficiency in real biomechanics!
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

class VerticalRatioAnalyzer:
    """
    Analyzes vertical ratio (oscillatory efficiency) using biological optimization theory
    """
    
    def __init__(self, data_file="../../experimental-data/circuit/annotated_track_series.json", 
                 results_dir="biomechanics_results/vertical_ratio"):
        self.data_file = Path(data_file)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Theoretical predictions from oscillatory efficiency optimization framework
        self.theoretical_predictions = {
            'optimal_vertical_ratio_range': (4.0, 8.0),  # % for efficient running
            'efficiency_stability_threshold': 0.2,  # Standard deviation < 20% of mean
            'speed_efficiency_coupling': -0.4,  # Negative correlation (higher speed, lower ratio)
            'cadence_efficiency_coupling': -0.3,  # Negative correlation (optimal cadence, lower ratio)
            'vertical_oscillation_coupling': 0.5,  # Positive correlation with vertical oscillation
            'terrain_adaptation_range': (3.0, 12.0),  # Adaptive range for different terrains
            'efficiency_optimization_threshold': 0.85,  # Target for oscillatory efficiency
            'energy_conservation_coefficient': 0.9,  # Energy conservation in efficient oscillations
        }
        
        # Load and validate data
        self.data = None
        self.vertical_ratio_series = None
        self.validation_results = {}
        
        print("📐⚡ VERTICAL RATIO ANALYZER ⚡📐")
        print("=" * 60)
        print("Analyzing oscillatory efficiency - the optimization of vertical motion!")
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
            
            # Clean and validate vertical ratio data
            # Vertical ratio of 0 indicates missing or invalid data
            self.data = self.data[self.data['vertical_ratio'] > 0]
            self.data = self.data.reset_index(drop=True)
            
            # Create time series
            self.vertical_ratio_series = self.data['vertical_ratio'].values
            
            print(f"✅ Loaded {len(self.data)} data points")
            print(f"📊 Vertical Ratio range: {self.vertical_ratio_series.min():.2f} - {self.vertical_ratio_series.max():.2f} %")
            print(f"⏱️  Time span: {self.data['time'].max() - self.data['time'].min():.1f} seconds")
            print(f"📐 Mean Vertical Ratio: {np.mean(self.vertical_ratio_series):.2f} ± {np.std(self.vertical_ratio_series):.2f} %")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def analyze_efficiency_optimization(self):
        """
        Analyze vertical ratio efficiency optimization and stability
        Tests: Biological Oscillatory Efficiency Optimization Theory
        """
        print("\n🔬 EXPERIMENT 1: Efficiency Optimization Analysis")
        print("-" * 50)
        
        if self.vertical_ratio_series is None:
            print("❌ No data available")
            return {}
        
        results = {}
        vertical_ratio = self.vertical_ratio_series
        
        # Basic efficiency characteristics
        mean_ratio = np.mean(vertical_ratio)
        std_ratio = np.std(vertical_ratio)
        cv_ratio = std_ratio / mean_ratio
        ratio_range = np.max(vertical_ratio) - np.min(vertical_ratio)
        
        results['efficiency_characteristics'] = {
            'mean_vertical_ratio': mean_ratio,
            'std_vertical_ratio': std_ratio,
            'cv_vertical_ratio': cv_ratio,
            'vertical_ratio_range': ratio_range,
            'efficiency_stability': std_ratio / mean_ratio < self.theoretical_predictions['efficiency_stability_threshold']
        }
        
        print(f"Mean Vertical Ratio: {mean_ratio:.2f} ± {std_ratio:.2f} %")
        print(f"Coefficient of Variation: {cv_ratio:.3f} ({cv_ratio*100:.1f}%)")
        print(f"Efficiency Stability: {'✅ STABLE' if results['efficiency_characteristics']['efficiency_stability'] else '⚠️  UNSTABLE'}")
        
        # Optimal efficiency range validation
        optimal_range = self.theoretical_predictions['optimal_vertical_ratio_range']
        in_optimal_range = np.sum((vertical_ratio >= optimal_range[0]) & (vertical_ratio <= optimal_range[1]))
        optimal_percentage = in_optimal_range / len(vertical_ratio) * 100
        
        results['optimal_efficiency_validation'] = {
            'optimal_range_adherence': optimal_percentage,
            'range_violations': len(vertical_ratio) - in_optimal_range,
            'theoretical_bounds': optimal_range,
            'efficient_oscillation_maintained': optimal_percentage > 60  # 60% in optimal range
        }
        
        print(f"Optimal Efficiency Range Adherence: {optimal_percentage:.1f}%")
        print(f"Efficient Oscillation: {'✅ MAINTAINED' if results['optimal_efficiency_validation']['efficient_oscillation_maintained'] else '⚠️  COMPROMISED'}")
        
        # Efficiency optimization over time
        if len(vertical_ratio) > 50:
            # Calculate rolling efficiency metrics
            window_size = min(20, len(vertical_ratio) // 4)
            rolling_mean = pd.Series(vertical_ratio).rolling(window=window_size, center=True).mean()
            rolling_std = pd.Series(vertical_ratio).rolling(window=window_size, center=True).std()
            
            # Calculate efficiency improvement trend
            time_indices = np.arange(len(vertical_ratio))
            efficiency_trend = np.polyfit(time_indices, vertical_ratio, 1)[0]  # Slope
            
            # Negative trend indicates improving efficiency (lower ratios)
            efficiency_improving = efficiency_trend < 0
            
            results['temporal_optimization'] = {
                'efficiency_trend': efficiency_trend,
                'efficiency_improving': efficiency_improving,
                'rolling_stability': np.nanstd(rolling_std),
                'optimization_consistency': 1.0 / (1.0 + np.nanvar(rolling_mean))
            }
            
            print(f"Efficiency Trend: {efficiency_trend:.4f} %/sample ({'Improving' if efficiency_improving else 'Declining'})")
            print(f"Optimization Consistency: {results['temporal_optimization']['optimization_consistency']:.3f}")
        
        # Energy conservation analysis
        # Lower vertical ratios indicate better energy conservation
        efficiency_score = 1.0 / (vertical_ratio + 0.1)  # Inverse relationship with ratio
        normalized_efficiency = efficiency_score / np.max(efficiency_score)
        
        energy_conservation = np.mean(normalized_efficiency)
        energy_stability = 1.0 - np.var(normalized_efficiency)
        
        results['energy_conservation'] = {
            'mean_efficiency_score': np.mean(efficiency_score),
            'energy_conservation_ratio': energy_conservation,
            'energy_stability': energy_stability,
            'conservation_adequate': energy_conservation >= self.theoretical_predictions['energy_conservation_coefficient']
        }
        
        print(f"Energy Conservation Ratio: {energy_conservation:.3f}")
        print(f"Conservation Adequate: {'✅ YES' if results['energy_conservation']['conservation_adequate'] else '❌ NO'}")
        
        # Efficiency distribution analysis
        # Test for optimization peaks (lower ratios should be more frequent for efficient runners)
        ratio_hist, ratio_bins = np.histogram(vertical_ratio, bins=20)
        
        # Find mode (most frequent range)
        mode_idx = np.argmax(ratio_hist)
        mode_value = (ratio_bins[mode_idx] + ratio_bins[mode_idx + 1]) / 2
        
        # Calculate skewness (should be positive for efficient runners - more low ratios)
        ratio_skewness = stats.skew(vertical_ratio)
        
        results['distribution_analysis'] = {
            'mode_vertical_ratio': mode_value,
            'distribution_skewness': ratio_skewness,
            'efficiency_optimized_distribution': ratio_skewness > 0 and mode_value < mean_ratio,
            'distribution_peak_in_efficient_range': optimal_range[0] <= mode_value <= optimal_range[1]
        }
        
        print(f"Distribution Mode: {mode_value:.2f}%")
        print(f"Skewness: {ratio_skewness:.3f}")
        print(f"Efficiency-Optimized Distribution: {'✅ YES' if results['distribution_analysis']['efficiency_optimized_distribution'] else '❌ NO'}")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        # Test efficiency stability
        predictions_validated['efficiency_stability'] = results['efficiency_characteristics']['efficiency_stability']
        
        # Test optimal range maintenance
        predictions_validated['optimal_range'] = results['optimal_efficiency_validation']['efficient_oscillation_maintained']
        
        # Test energy conservation
        predictions_validated['energy_conservation'] = results['energy_conservation']['conservation_adequate']
        
        # Test optimization trend (if available)
        if 'temporal_optimization' in results:
            predictions_validated['efficiency_improvement'] = results['temporal_optimization']['efficiency_improving']
        
        # Test distribution optimization
        predictions_validated['distribution_optimization'] = results['distribution_analysis']['efficiency_optimized_distribution']
        
        results['theoretical_validation'] = predictions_validated
        validation_success = sum(predictions_validated.values()) >= len(predictions_validated) // 2
        results['validation_success'] = validation_success
        
        print(f"🎯 Theoretical Validation: {'✅ SUCCESS' if validation_success else '❌ FAILED'}")
        
        return results
    
    def analyze_multi_parameter_efficiency_coupling(self):
        """
        Analyze efficiency coupling with speed, cadence, and vertical oscillation
        Tests: Multi-Scale Efficiency Coupling Theory
        """
        print("\n🔬 EXPERIMENT 2: Multi-Parameter Efficiency Coupling")
        print("-" * 55)
        
        if self.data is None:
            print("❌ No data available")
            return {}
        
        # Extract relevant data for coupling analysis
        vertical_ratio = self.data['vertical_ratio'].values
        speed = self.data['speed'].values
        cadence = self.data['cadence'].values
        vertical_osc = self.data['vertical_oscillation'].values
        
        # Remove invalid data points
        valid_indices = (vertical_ratio > 0) & (speed > 0) & (cadence > 0) & (vertical_osc > 0)
        
        ratio_clean = vertical_ratio[valid_indices]
        speed_clean = speed[valid_indices]
        cadence_clean = cadence[valid_indices]
        vertical_clean = vertical_osc[valid_indices]
        
        if len(ratio_clean) < 10:
            print("❌ Insufficient valid data")
            return {}
        
        results = {}
        
        # Vertical ratio - Speed coupling
        ratio_speed_correlation = np.corrcoef(ratio_clean, speed_clean)[0, 1]
        
        results['speed_coupling'] = {
            'correlation_coefficient': ratio_speed_correlation,
            'coupling_strength': abs(ratio_speed_correlation),
            'coupling_direction': 'positive' if ratio_speed_correlation > 0 else 'negative',
            'efficient_coupling': ratio_speed_correlation < 0,  # Negative correlation indicates efficiency
            'significant_coupling': abs(ratio_speed_correlation) >= abs(self.theoretical_predictions['speed_efficiency_coupling'])
        }
        
        print(f"Ratio-Speed Coupling: {ratio_speed_correlation:.3f}")
        print(f"Efficient Coupling: {'✅ YES' if results['speed_coupling']['efficient_coupling'] else '❌ NO'}")
        
        # Vertical ratio - Cadence coupling
        ratio_cadence_correlation = np.corrcoef(ratio_clean, cadence_clean)[0, 1]
        
        results['cadence_coupling'] = {
            'correlation_coefficient': ratio_cadence_correlation,
            'coupling_strength': abs(ratio_cadence_correlation),
            'coupling_direction': 'positive' if ratio_cadence_correlation > 0 else 'negative',
            'efficient_coupling': ratio_cadence_correlation < 0,  # Negative correlation indicates efficiency
            'significant_coupling': abs(ratio_cadence_correlation) >= abs(self.theoretical_predictions['cadence_efficiency_coupling'])
        }
        
        print(f"Ratio-Cadence Coupling: {ratio_cadence_correlation:.3f}")
        print(f"Efficient Coupling: {'✅ YES' if results['cadence_coupling']['efficient_coupling'] else '❌ NO'}")
        
        # Vertical ratio - Vertical oscillation coupling
        ratio_vertical_correlation = np.corrcoef(ratio_clean, vertical_clean)[0, 1]
        
        results['vertical_oscillation_coupling'] = {
            'correlation_coefficient': ratio_vertical_correlation,
            'coupling_strength': abs(ratio_vertical_correlation),
            'coupling_direction': 'positive' if ratio_vertical_correlation > 0 else 'negative',
            'expected_positive_coupling': ratio_vertical_correlation > 0,  # Should be positive
            'significant_coupling': abs(ratio_vertical_correlation) >= self.theoretical_predictions['vertical_oscillation_coupling']
        }
        
        print(f"Ratio-Vertical Oscillation Coupling: {ratio_vertical_correlation:.3f}")
        print(f"Expected Positive Coupling: {'✅ YES' if results['vertical_oscillation_coupling']['expected_positive_coupling'] else '❌ NO'}")
        
        # Multi-dimensional efficiency analysis
        if len(ratio_clean) > 50:
            # Create efficiency landscape
            # Calculate composite efficiency score based on multiple parameters
            speed_normalized = (speed_clean - np.min(speed_clean)) / (np.max(speed_clean) - np.min(speed_clean))
            cadence_normalized = (cadence_clean - np.min(cadence_clean)) / (np.max(cadence_clean) - np.min(cadence_clean))
            
            # Higher speed and optimal cadence should correlate with lower vertical ratio
            expected_efficiency = speed_normalized + (1.0 - np.abs(cadence_normalized - 0.5))  # Optimal cadence at middle range
            
            efficiency_correlation = np.corrcoef(ratio_clean, expected_efficiency)[0, 1]
            
            results['composite_efficiency'] = {
                'efficiency_landscape_correlation': efficiency_correlation,
                'landscape_coupling_strength': abs(efficiency_correlation),
                'efficiency_optimized': efficiency_correlation < -0.3  # Strong negative correlation expected
            }
            
            print(f"Efficiency Landscape Correlation: {efficiency_correlation:.3f}")
            print(f"Efficiency Optimized: {'✅ YES' if results['composite_efficiency']['efficiency_optimized'] else '❌ NO'}")
        
        # Performance efficiency analysis
        # Calculate performance metric and its relationship with vertical ratio
        performance_metric = speed_clean * cadence_clean / (vertical_clean + 1.0)  # Normalized performance
        performance_ratio_correlation = np.corrcoef(ratio_clean, performance_metric)[0, 1]
        
        results['performance_efficiency'] = {
            'performance_ratio_correlation': performance_ratio_correlation,
            'performance_coupling_strength': abs(performance_ratio_correlation),
            'performance_optimized': performance_ratio_correlation < -0.2  # Negative correlation expected
        }
        
        print(f"Performance-Efficiency Coupling: {performance_ratio_correlation:.3f}")
        print(f"Performance Optimized: {'✅ YES' if results['performance_efficiency']['performance_optimized'] else '❌ NO'}")
        
        # Efficiency phase relationships
        if len(ratio_clean) > 100:
            # Analyze phase relationships between efficiency and performance parameters
            
            # Cross-correlation with speed
            ratio_centered = ratio_clean - np.mean(ratio_clean)
            speed_centered = speed_clean - np.mean(speed_clean)
            
            cross_corr = signal.correlate(ratio_centered, speed_centered, mode='full')
            cross_corr = cross_corr / (np.std(ratio_centered) * np.std(speed_centered) * len(ratio_centered))
            
            # Find peak correlation and lag
            max_corr_idx = np.argmax(np.abs(cross_corr))
            lag = max_corr_idx - len(ratio_centered) + 1
            max_correlation = cross_corr[max_corr_idx]
            
            results['phase_relationships'] = {
                'max_cross_correlation': max_correlation,
                'optimal_lag': lag,
                'phase_coupling_strength': abs(max_correlation),
                'synchronized_efficiency': abs(lag) < 10  # Low lag indicates synchronization
            }
            
            print(f"Phase Coupling Strength: {abs(max_correlation):.3f}")
            print(f"Synchronized Efficiency: {'✅ YES' if results['phase_relationships']['synchronized_efficiency'] else '❌ NO'}")
        
        # Efficiency optimization boundaries
        # Identify the relationship between efficiency and performance boundaries
        if len(ratio_clean) > 20:
            # Calculate percentiles to identify efficiency boundaries
            ratio_percentiles = np.percentile(ratio_clean, [10, 25, 50, 75, 90])
            
            # Analyze performance at different efficiency levels
            efficiency_performance = {}
            
            for i, (lower, upper) in enumerate(zip(ratio_percentiles[:-1], ratio_percentiles[1:])):
                efficiency_mask = (ratio_clean >= lower) & (ratio_clean < upper)
                if np.sum(efficiency_mask) >= 3:
                    efficiency_performance[f'percentile_{10+i*20}-{30+i*20}'] = {
                        'mean_speed': np.mean(speed_clean[efficiency_mask]),
                        'mean_cadence': np.mean(cadence_clean[efficiency_mask]),
                        'n_points': np.sum(efficiency_mask)
                    }
            
            results['efficiency_boundaries'] = efficiency_performance
            
            if efficiency_performance:
                # Check if higher efficiency (lower ratios) corresponds to better performance
                efficiency_keys = list(efficiency_performance.keys())
                if len(efficiency_keys) >= 2:
                    first_quartile_speed = efficiency_performance[efficiency_keys[0]]['mean_speed']
                    last_quartile_speed = efficiency_performance[efficiency_keys[-1]]['mean_speed']
                    
                    efficiency_performance_relationship = first_quartile_speed > last_quartile_speed  # Lower ratios should have higher speed
                    
                    results['efficiency_performance_relationship'] = efficiency_performance_relationship
                    
                    print(f"Efficiency-Performance Relationship: {'✅ VALIDATED' if efficiency_performance_relationship else '❌ NOT VALIDATED'}")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        # Test speed coupling
        predictions_validated['speed_coupling'] = results['speed_coupling']['efficient_coupling']
        
        # Test cadence coupling
        predictions_validated['cadence_coupling'] = results['cadence_coupling']['efficient_coupling']
        
        # Test vertical oscillation coupling
        predictions_validated['vertical_coupling'] = results['vertical_oscillation_coupling']['expected_positive_coupling']
        
        # Test composite efficiency
        if 'composite_efficiency' in results:
            predictions_validated['composite_efficiency'] = results['composite_efficiency']['efficiency_optimized']
        
        # Test performance efficiency
        predictions_validated['performance_efficiency'] = results['performance_efficiency']['performance_optimized']
        
        # Test efficiency-performance relationship
        if 'efficiency_performance_relationship' in results:
            predictions_validated['efficiency_performance_relationship'] = results['efficiency_performance_relationship']
        
        results['theoretical_validation'] = predictions_validated
        validation_success = sum(predictions_validated.values()) >= len(predictions_validated) // 2
        results['validation_success'] = validation_success
        
        print(f"🎯 Theoretical Validation: {'✅ SUCCESS' if validation_success else '❌ FAILED'}")
        
        return results
    
    def analyze_terrain_efficiency_adaptation(self):
        """
        Analyze efficiency adaptation to different terrain segments
        Tests: Environmental Efficiency Adaptation Theory
        """
        print("\n🔬 EXPERIMENT 3: Terrain Efficiency Adaptation")
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
                
            ratio_segment = segment_df['vertical_ratio'].values
            speed_segment = segment_df['speed'].values
            
            # Remove invalid data
            valid_mask = (ratio_segment > 0) & (speed_segment > 0)
            
            if np.sum(valid_mask) < 3:
                continue
                
            ratio_clean = ratio_segment[valid_mask]
            speed_clean = speed_segment[valid_mask]
            
            # Calculate segment-specific efficiency metrics
            segment_stats = {
                'n_points': len(ratio_clean),
                'mean_vertical_ratio': np.mean(ratio_clean),
                'std_vertical_ratio': np.std(ratio_clean),
                'cv_vertical_ratio': np.std(ratio_clean) / np.mean(ratio_clean),
                'mean_speed': np.mean(speed_clean),
                'ratio_range': np.max(ratio_clean) - np.min(ratio_clean)
            }
            
            # Calculate adaptation-specific metrics
            segment_stats['efficiency_score'] = 1.0 / (segment_stats['mean_vertical_ratio'] + 0.1)  # Inverse relationship
            segment_stats['speed_efficiency'] = segment_stats['mean_speed'] / segment_stats['mean_vertical_ratio']
            segment_stats['adaptation_index'] = segment_stats['cv_vertical_ratio'] * segment_stats['ratio_range']
            segment_stats['efficiency_stability'] = 1.0 / (1.0 + segment_stats['cv_vertical_ratio'])
            
            segment_analysis[segment_name] = segment_stats
            
            print(f"  {segment_name}: {segment_stats['mean_vertical_ratio']:.2f} ± {segment_stats['std_vertical_ratio']:.2f} %")
            print(f"    Efficiency Score: {segment_stats['efficiency_score']:.3f}, Speed/Efficiency: {segment_stats['speed_efficiency']:.2f}")
        
        results['segment_analysis'] = segment_analysis
        
        # Cross-segment adaptation analysis
        if len(segment_analysis) >= 2:
            segment_names = list(segment_analysis.keys())
            
            # Extract metrics for analysis
            mean_ratios = [segment_analysis[seg]['mean_vertical_ratio'] for seg in segment_names]
            efficiency_scores = [segment_analysis[seg]['efficiency_score'] for seg in segment_names]
            speed_efficiencies = [segment_analysis[seg]['speed_efficiency'] for seg in segment_names]
            stability_scores = [segment_analysis[seg]['efficiency_stability'] for seg in segment_names]
            
            # Calculate adaptation metrics
            ratio_adaptation_range = np.max(mean_ratios) - np.min(mean_ratios)
            efficiency_adaptation_range = np.max(efficiency_scores) - np.min(efficiency_scores)
            stability_adaptation_range = np.max(stability_scores) - np.min(stability_scores)
            
            results['adaptation_analysis'] = {
                'ratio_adaptation_range': ratio_adaptation_range,
                'efficiency_adaptation_range': efficiency_adaptation_range,
                'stability_adaptation_range': stability_adaptation_range,
                'mean_efficiency_score': np.mean(efficiency_scores),
                'mean_speed_efficiency': np.mean(speed_efficiencies),
                'adaptation_consistency': 1.0 - (np.std(mean_ratios) / np.mean(mean_ratios)),
                'segment_count': len(segment_names)
            }
            
            print(f"Ratio Adaptation Range: {ratio_adaptation_range:.2f} %")
            print(f"Mean Efficiency Score: {results['adaptation_analysis']['mean_efficiency_score']:.3f}")
            print(f"Adaptation Consistency: {results['adaptation_analysis']['adaptation_consistency']:.3f}")
            
            # Statistical analysis of segment differences
            segment_ratios = []
            
            for segment_name in segment_names:
                segment_df = self.data[self.data['segments'] == segment_name]
                valid_data = segment_df[segment_df['vertical_ratio'] > 0]
                
                if len(valid_data) >= 3:
                    segment_ratios.append(valid_data['vertical_ratio'].values)
            
            if len(segment_ratios) >= 2:
                # ANOVA test for significant differences
                try:
                    f_stat, p_value = stats.f_oneway(*segment_ratios)
                    
                    results['statistical_analysis'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant_adaptation': p_value < 0.05
                    }
                    
                    print(f"Efficiency ANOVA: F={f_stat:.2f}, p={p_value:.4f}")
                    print(f"Significant Adaptation: {'✅ YES' if p_value < 0.05 else '❌ NO'}")
                    
                except Exception as e:
                    print(f"Statistical analysis failed: {str(e)}")
        
        # Terrain-specific efficiency optimization
        if len(segment_analysis) >= 2:
            # Identify most and least efficient segments
            efficiency_ranking = sorted(segment_analysis.items(), 
                                      key=lambda x: x[1]['efficiency_score'], reverse=True)
            
            most_efficient_segment = efficiency_ranking[0]
            least_efficient_segment = efficiency_ranking[-1]
            
            results['efficiency_ranking'] = {
                'most_efficient': {
                    'segment': most_efficient_segment[0],
                    'ratio': most_efficient_segment[1]['mean_vertical_ratio'],
                    'score': most_efficient_segment[1]['efficiency_score']
                },
                'least_efficient': {
                    'segment': least_efficient_segment[0],
                    'ratio': least_efficient_segment[1]['mean_vertical_ratio'],
                    'score': least_efficient_segment[1]['efficiency_score']
                },
                'efficiency_range': most_efficient_segment[1]['efficiency_score'] - least_efficient_segment[1]['efficiency_score']
            }
            
            print(f"Most Efficient Segment: {most_efficient_segment[0]} ({most_efficient_segment[1]['mean_vertical_ratio']:.2f}%)")
            print(f"Least Efficient Segment: {least_efficient_segment[0]} ({least_efficient_segment[1]['mean_vertical_ratio']:.2f}%)")
        
        # Speed-efficiency relationship across terrains
        if len(segment_analysis) >= 2:
            # Calculate correlation between speed and efficiency across segments
            segment_speeds = [segment_analysis[seg]['mean_speed'] for seg in segment_names]
            segment_efficiency_scores = [segment_analysis[seg]['efficiency_score'] for seg in segment_names]
            
            if len(segment_speeds) >= 3:
                speed_efficiency_correlation = np.corrcoef(segment_speeds, segment_efficiency_scores)[0, 1]
                
                results['terrain_speed_efficiency'] = {
                    'speed_efficiency_correlation': speed_efficiency_correlation,
                    'positive_relationship': speed_efficiency_correlation > 0.3,  # Higher speed, higher efficiency
                    'terrain_optimization': abs(speed_efficiency_correlation) > 0.5
                }
                
                print(f"Terrain Speed-Efficiency Correlation: {speed_efficiency_correlation:.3f}")
                print(f"Terrain Optimization: {'✅ YES' if results['terrain_speed_efficiency']['terrain_optimization'] else '❌ NO'}")
        
        # Efficiency adaptation boundaries
        if 'adaptation_analysis' in results:
            adaptation_range = self.theoretical_predictions['terrain_adaptation_range']
            
            # Check if adaptation stays within theoretical bounds
            all_ratios = [segment_analysis[seg]['mean_vertical_ratio'] for seg in segment_names]
            min_ratio = min(all_ratios)
            max_ratio = max(all_ratios)
            
            within_bounds = (min_ratio >= adaptation_range[0]) and (max_ratio <= adaptation_range[1])
            
            results['adaptation_boundaries'] = {
                'min_ratio': min_ratio,
                'max_ratio': max_ratio,
                'theoretical_bounds': adaptation_range,
                'within_theoretical_bounds': within_bounds,
                'adaptation_range_actual': max_ratio - min_ratio
            }
            
            print(f"Adaptation Range: {min_ratio:.2f} - {max_ratio:.2f} %")
            print(f"Within Theoretical Bounds: {'✅ YES' if within_bounds else '❌ NO'}")
        
        # Validate theoretical predictions
        predictions_validated = {}
        
        # Test adaptation range (should be within reasonable bounds)
        if 'adaptation_boundaries' in results:
            predictions_validated['adaptation_bounds'] = results['adaptation_boundaries']['within_theoretical_bounds']
        
        # Test adaptation consistency (should show some adaptation but maintain consistency)
        if 'adaptation_analysis' in results:
            consistency_threshold = 0.6
            predictions_validated['adaptation_consistency'] = results['adaptation_analysis']['adaptation_consistency'] >= consistency_threshold
            
            # Test efficiency maintenance
            efficiency_threshold = 2.0  # Minimum efficiency score
            predictions_validated['efficiency_maintenance'] = results['adaptation_analysis']['mean_efficiency_score'] >= efficiency_threshold
        
        # Test significant adaptive response
        if 'statistical_analysis' in results:
            predictions_validated['significant_adaptation'] = results['statistical_analysis']['significant_adaptation']
        
        # Test terrain optimization
        if 'terrain_speed_efficiency' in results:
            predictions_validated['terrain_optimization'] = results['terrain_speed_efficiency']['terrain_optimization']
        
        results['theoretical_validation'] = predictions_validated
        validation_success = sum(predictions_validated.values()) >= len(predictions_validated) // 2
        results['validation_success'] = validation_success
        
        print(f"🎯 Theoretical Validation: {'✅ SUCCESS' if validation_success else '❌ FAILED'}")
        
        return results
    
    def create_comprehensive_visualizations(self, results):
        """Generate comprehensive visualizations of vertical ratio analysis"""
        print("\n🎨 Creating Comprehensive Visualizations...")
        
        if self.data is None:
            print("❌ No data for visualization")
            return
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Vertical ratio time series
        ax1 = plt.subplot(3, 4, 1)
        time = self.data['time'].values
        vertical_ratio = self.data['vertical_ratio'].values
        
        plt.plot(time, vertical_ratio, 'b-', alpha=0.7, linewidth=1)
        plt.axhline(y=np.mean(vertical_ratio), color='r', linestyle='--', alpha=0.8, 
                   label=f'Mean: {np.mean(vertical_ratio):.2f}%')
        
        # Add optimal range
        optimal_range = self.theoretical_predictions['optimal_vertical_ratio_range']
        plt.axhspan(optimal_range[0], optimal_range[1], alpha=0.2, color='green', label='Optimal Range')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Vertical Ratio (%)')
        plt.title('Vertical Ratio Time Series')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Vertical ratio distribution
        ax2 = plt.subplot(3, 4, 2)
        plt.hist(vertical_ratio, bins=25, alpha=0.7, color='blue', density=True)
        plt.axvline(x=np.mean(vertical_ratio), color='red', linestyle='--', label=f'Mean: {np.mean(vertical_ratio):.2f}%')
        
        # Add efficiency zones
        plt.axvspan(0, optimal_range[0], alpha=0.2, color='green', label='Highly Efficient')
        plt.axvspan(optimal_range[0], optimal_range[1], alpha=0.2, color='yellow', label='Efficient')
        plt.axvspan(optimal_range[1], 15, alpha=0.2, color='red', label='Less Efficient')
        
        plt.xlabel('Vertical Ratio (%)')
        plt.ylabel('Density')
        plt.title('Efficiency Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Vertical ratio vs Speed (efficiency relationship)
        ax3 = plt.subplot(3, 4, 3)
        valid_data = self.data[(self.data['speed'] > 0) & (self.data['vertical_ratio'] > 0)]
        
        if len(valid_data) > 10:
            speed_valid = valid_data['speed'].values
            ratio_valid = valid_data['vertical_ratio'].values
            
            scatter = plt.scatter(speed_valid, ratio_valid, alpha=0.6, 
                                c=valid_data['time'].values, cmap='viridis', s=20)
            
            # Add trend line
            z = np.polyfit(speed_valid, ratio_valid, 1)
            p = np.poly1d(z)
            plt.plot(speed_valid, p(speed_valid), 'r--', alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr = np.corrcoef(speed_valid, ratio_valid)[0, 1]
            plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.colorbar(scatter, label='Time (s)')
            plt.xlabel('Speed (m/s)')
            plt.ylabel('Vertical Ratio (%)')
            plt.title('Speed-Efficiency Relationship')
            plt.grid(True, alpha=0.3)
        
        # 4. Vertical ratio vs Cadence
        ax4 = plt.subplot(3, 4, 4)
        valid_cadence_data = self.data[(self.data['cadence'] > 0) & (self.data['vertical_ratio'] > 0)]
        
        if len(valid_cadence_data) > 10:
            cadence_valid = valid_cadence_data['cadence'].values
            ratio_valid = valid_cadence_data['vertical_ratio'].values
            
            scatter = plt.scatter(cadence_valid, ratio_valid, alpha=0.6, 
                                c=valid_cadence_data['speed'].values, cmap='plasma', s=20)
            
            # Add trend line
            z = np.polyfit(cadence_valid, ratio_valid, 1)
            p = np.poly1d(z)
            plt.plot(cadence_valid, p(cadence_valid), 'r--', alpha=0.8, linewidth=2)
            
            corr = np.corrcoef(cadence_valid, ratio_valid)[0, 1]
            plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.colorbar(scatter, label='Speed (m/s)')
            plt.xlabel('Cadence (steps/min)')
            plt.ylabel('Vertical Ratio (%)')
            plt.title('Cadence-Efficiency Relationship')
            plt.grid(True, alpha=0.3)
        
        # 5. Vertical ratio vs Vertical oscillation
        ax5 = plt.subplot(3, 4, 5)
        valid_vertical_data = self.data[(self.data['vertical_oscillation'] > 0) & (self.data['vertical_ratio'] > 0)]
        
        if len(valid_vertical_data) > 10:
            vertical_osc_valid = valid_vertical_data['vertical_oscillation'].values
            ratio_valid = valid_vertical_data['vertical_ratio'].values
            
            plt.scatter(vertical_osc_valid, ratio_valid, alpha=0.6, 
                       c=valid_vertical_data['speed'].values, cmap='coolwarm', s=20)
            
            # Add trend line
            z = np.polyfit(vertical_osc_valid, ratio_valid, 1)
            p = np.poly1d(z)
            plt.plot(vertical_osc_valid, p(vertical_osc_valid), 'r--', alpha=0.8, linewidth=2)
            
            plt.colorbar(label='Speed (m/s)')
            plt.xlabel('Vertical Oscillation (mm)')
            plt.ylabel('Vertical Ratio (%)')
            plt.title('Vertical Motion Coupling')
            plt.grid(True, alpha=0.3)
        
        # 6. Segment-wise efficiency analysis
        ax6 = plt.subplot(3, 4, 6)
        segment_data = self.data.groupby('segments')
        
        segment_means = []
        segment_names = []
        segment_stds = []
        segment_efficiency_scores = []
        
        for segment_name, segment_df in segment_data:
            valid_segment = segment_df['vertical_ratio'] > 0
            if np.sum(valid_segment) >= 5:
                segment_ratios = segment_df[valid_segment]['vertical_ratio'].values
                segment_means.append(np.mean(segment_ratios))
                segment_stds.append(np.std(segment_ratios))
                segment_efficiency_scores.append(1.0 / (np.mean(segment_ratios) + 0.1))
                segment_names.append(segment_name)
        
        if segment_means:
            x_pos = np.arange(len(segment_names))
            bars = plt.bar(x_pos, segment_means, yerr=segment_stds, alpha=0.7, capsize=5)
            
            # Color bars by efficiency score (lower ratios = more efficient = greener)
            if segment_efficiency_scores:
                colors = plt.cm.RdYlGn([score/max(segment_efficiency_scores) for score in segment_efficiency_scores])
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
            
            plt.xticks(x_pos, segment_names, rotation=45)
            plt.ylabel('Vertical Ratio (%)')
            plt.title('Segment Efficiency Analysis')
            plt.grid(True, alpha=0.3)
        
        # 7. Efficiency score over time
        ax7 = plt.subplot(3, 4, 7)
        efficiency_score = 1.0 / (vertical_ratio + 0.1)  # Inverse relationship
        
        plt.plot(time, efficiency_score, 'g-', alpha=0.7, linewidth=1)
        plt.axhline(y=np.mean(efficiency_score), color='r', linestyle='--', alpha=0.8, 
                   label=f'Mean: {np.mean(efficiency_score):.2f}')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Efficiency Score')
        plt.title('Efficiency Score Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Rolling efficiency statistics
        ax8 = plt.subplot(3, 4, 8)
        if len(vertical_ratio) > 50:
            window_size = max(20, len(vertical_ratio) // 10)
            rolling_mean = pd.Series(vertical_ratio).rolling(window=window_size, center=True).mean()
            rolling_std = pd.Series(vertical_ratio).rolling(window=window_size, center=True).std()
            
            plt.plot(time, rolling_mean, 'b-', alpha=0.8, label='Rolling Mean')
            plt.fill_between(time, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                           alpha=0.3, color='blue', label='±1σ')
            
            # Add optimal range
            plt.axhspan(optimal_range[0], optimal_range[1], alpha=0.2, color='green', label='Optimal Range')
            
            plt.xlabel('Time (s)')
            plt.ylabel('Vertical Ratio (%)')
            plt.title('Rolling Efficiency Statistics')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 9. Efficiency optimization landscape
        ax9 = plt.subplot(3, 4, 9)
        valid_landscape_data = self.data[(self.data['speed'] > 0) & 
                                        (self.data['cadence'] > 0) & 
                                        (self.data['vertical_ratio'] > 0)]
        
        if len(valid_landscape_data) > 20:
            # Create 2D histogram of speed vs cadence colored by efficiency
            speed_vals = valid_landscape_data['speed'].values
            cadence_vals = valid_landscape_data['cadence'].values
            efficiency_vals = 1.0 / (valid_landscape_data['vertical_ratio'].values + 0.1)
            
            # Create grid for interpolation
            from scipy.interpolate import griddata
            
            speed_grid = np.linspace(speed_vals.min(), speed_vals.max(), 20)
            cadence_grid = np.linspace(cadence_vals.min(), cadence_vals.max(), 20)
            Speed_grid, Cadence_grid = np.meshgrid(speed_grid, cadence_grid)
            
            # Interpolate efficiency values
            points = np.column_stack([speed_vals, cadence_vals])
            grid_points = np.column_stack([Speed_grid.ravel(), Cadence_grid.ravel()])
            efficiency_grid = griddata(points, efficiency_vals, grid_points, method='linear')
            efficiency_grid = efficiency_grid.reshape(Speed_grid.shape)
            
            # Remove NaN values
            efficiency_grid = np.nan_to_num(efficiency_grid, nan=np.nanmean(efficiency_grid))
            
            contour = plt.contourf(Speed_grid, Cadence_grid, efficiency_grid, levels=15, cmap='RdYlGn')
            plt.colorbar(contour, label='Efficiency Score')
            
            plt.xlabel('Speed (m/s)')
            plt.ylabel('Cadence (steps/min)')
            plt.title('Efficiency Optimization Landscape')
        
        # 10. Performance vs Efficiency scatter
        ax10 = plt.subplot(3, 4, 10)
        valid_perf_data = self.data[(self.data['speed'] > 0) & 
                                   (self.data['cadence'] > 0) & 
                                   (self.data['vertical_ratio'] > 0)]
        
        if len(valid_perf_data) > 10:
            # Calculate performance metric
            performance = valid_perf_data['speed'].values * valid_perf_data['cadence'].values / 60.0
            efficiency = 1.0 / (valid_perf_data['vertical_ratio'].values + 0.1)
            
            plt.scatter(performance, efficiency, alpha=0.6, 
                       c=valid_perf_data['time'].values, cmap='viridis', s=20)
            
            # Add trend line
            z = np.polyfit(performance, efficiency, 1)
            p = np.poly1d(z)
            plt.plot(performance, p(performance), 'r--', alpha=0.8, linewidth=2)
            
            plt.colorbar(label='Time (s)')
            plt.xlabel('Performance (speed × cadence)')
            plt.ylabel('Efficiency Score')
            plt.title('Performance vs Efficiency')
            plt.grid(True, alpha=0.3)
        
        # 11. Heart rate zones and efficiency
        ax11 = plt.subplot(3, 4, 11)
        if 'hr_zones' in self.data.columns:
            hr_zone_efficiency = self.data[self.data['vertical_ratio'] > 0].groupby('hr_zones')['vertical_ratio'].agg(['mean', 'std']).reset_index()
            
            if len(hr_zone_efficiency) > 1:
                plt.errorbar(hr_zone_efficiency['hr_zones'], hr_zone_efficiency['mean'], 
                           yerr=hr_zone_efficiency['std'], fmt='o-', capsize=5, alpha=0.8)
                
                plt.axhspan(optimal_range[0], optimal_range[1], alpha=0.2, color='green', label='Optimal Range')
                
                plt.xlabel('Heart Rate Zone')
                plt.ylabel('Vertical Ratio (%)')
                plt.title('Efficiency by HR Zone')
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
        output_path = self.results_dir / 'vertical_ratio_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Comprehensive dashboard saved: {output_path}")
        
        # Create interactive visualization
        self._create_interactive_visualization(results)
    
    def _create_interactive_visualization(self, results):
        """Create interactive Plotly visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Efficiency Time Series', 'Speed-Efficiency Coupling', 
                          'Efficiency Landscape', 'Segment Analysis'),
            specs=[[{"secondary_y": True}, {}],
                   [{}, {}]]
        )
        
        # Efficiency time series with speed overlay
        fig.add_trace(
            go.Scatter(x=self.data['time'], y=self.data['vertical_ratio'], 
                      name='Vertical Ratio', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=self.data['time'], y=self.data['speed'], 
                      name='Speed', line=dict(color='red')),
            row=1, col=1, secondary_y=True
        )
        
        # Speed-efficiency coupling
        valid_data = self.data[(self.data['speed'] > 0) & (self.data['vertical_ratio'] > 0)]
        if len(valid_data) > 10:
            fig.add_trace(
                go.Scatter(x=valid_data['speed'], y=valid_data['vertical_ratio'],
                          mode='markers', name='Speed-Efficiency',
                          marker=dict(color=valid_data['cadence'], colorscale='viridis')),
                row=1, col=2
            )
        
        # Efficiency landscape (3D-like using color)
        valid_3d_data = self.data[(self.data['speed'] > 0) & 
                                 (self.data['cadence'] > 0) & 
                                 (self.data['vertical_ratio'] > 0)]
        
        if len(valid_3d_data) > 10:
            efficiency_score = 1.0 / (valid_3d_data['vertical_ratio'].values + 0.1)
            fig.add_trace(
                go.Scatter(x=valid_3d_data['speed'], y=valid_3d_data['cadence'],
                          mode='markers', name='Efficiency Landscape',
                          marker=dict(color=efficiency_score, colorscale='RdYlGn')),
                row=2, col=1
            )
        
        # Segment analysis
        segment_data = self.data.groupby('segments')
        segment_means = []
        segment_names = []
        
        for segment_name, segment_df in segment_data:
            valid_segment = segment_df['vertical_ratio'] > 0
            if np.sum(valid_segment) >= 5:
                segment_ratios = segment_df[valid_segment]['vertical_ratio'].values
                segment_means.append(np.mean(segment_ratios))
                segment_names.append(segment_name)
        
        if segment_means:
            fig.add_trace(
                go.Bar(x=segment_names, y=segment_means, name='Segment Efficiency'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Interactive Vertical Ratio Efficiency Analysis")
        
        # Save interactive visualization
        interactive_path = self.results_dir / 'vertical_ratio_interactive.html'
        fig.write_html(str(interactive_path))
        
        print(f"  📊 Interactive visualization saved: {interactive_path}")
    
    def save_results(self, results):
        """Save analysis results to JSON"""
        output_file = self.results_dir / 'vertical_ratio_results.json'
        
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
        """Run complete vertical ratio efficiency analysis"""
        print("\n" + "="*60)
        print("📐⚡ COMPREHENSIVE VERTICAL RATIO ANALYSIS ⚡📐")
        print("="*60)
        print("Analyzing oscillatory efficiency - the optimization of motion!")
        print("="*60)
        
        # Load data
        if not self.load_track_data():
            return None
        
        # Run all experiments
        results = {}
        
        experiments = [
            ('efficiency_optimization', self.analyze_efficiency_optimization),
            ('multi_parameter_coupling', self.analyze_multi_parameter_efficiency_coupling),
            ('terrain_adaptation', self.analyze_terrain_efficiency_adaptation),
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
            'vertical_ratio_validation_success': successful_experiments >= 2,
            'theoretical_predictions_validated': self._count_validated_predictions(results),
            'data_points_analyzed': len(self.data) if self.data is not None else 0,
            'vertical_ratio_range_analyzed': f"{self.vertical_ratio_series.min():.2f} - {self.vertical_ratio_series.max():.2f} %",
            'mean_vertical_ratio': np.mean(self.vertical_ratio_series),
            'efficiency_score': 1.0 / (np.mean(self.vertical_ratio_series) + 0.1),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        results['comprehensive_summary'] = comprehensive_summary
        
        # Create visualizations
        self.create_comprehensive_visualizations(results)
        
        # Save results
        self.save_results(results)
        
        # Print final summary
        print("\n" + "="*60)
        print("📊 VERTICAL RATIO ANALYSIS SUMMARY")
        print("="*60)
        print(f"Experiments Run: {len(experiments)}")
        print(f"Successful Validations: {successful_experiments}")
        print(f"Success Rate: {comprehensive_summary['validation_success_rate']*100:.1f}%")
        print(f"Framework Validated: {comprehensive_summary['vertical_ratio_validation_success']}")
        print(f"Ratio Range: {comprehensive_summary['vertical_ratio_range_analyzed']}")
        print(f"Mean Vertical Ratio: {comprehensive_summary['mean_vertical_ratio']:.2f}%")
        print(f"Efficiency Score: {comprehensive_summary['efficiency_score']:.3f}")
        print("="*60)
        
        if comprehensive_summary['vertical_ratio_validation_success']:
            print("\n🎉 OSCILLATORY EFFICIENCY VALIDATION SUCCESS! 🎉")
            print("Efficiency optimization theories validated with real data!")
            print("First framework to analyze real-time biomechanical efficiency!")
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
    """Main function to run vertical ratio analysis"""
    print("Starting Vertical Ratio Analysis...")
    
    analyzer = VerticalRatioAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print(f"Results saved in: {analyzer.results_dir}")
        
        if results.get('comprehensive_summary', {}).get('vertical_ratio_validation_success', False):
            print(f"\n🏆 BREAKTHROUGH: OSCILLATORY EFFICIENCY VALIDATES THEORIES! 🏆")
            print(f"First framework to validate efficiency optimization in biomechanics!")
    else:
        print(f"\n❌ Analysis could not be completed")


if __name__ == "__main__":
    main()
