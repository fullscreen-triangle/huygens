#!/usr/bin/env python3
"""
Biomechanical Data Validation Script
Analyzes real biomechanical oscillatory data using the universal transformation framework
Specifically designed for gait, knee mechanics, and center of mass dynamics

Usage:
    python validate_biomechanical_data.py --gait experimental-data/gait/gait_cycle_track.json
    python validate_biomechanical_data.py --knee experimental-data/knee/knee_angle.json
    python validate_biomechanical_data.py --stability experimental-data/stability/com.json
    python validate_biomechanical_data.py --all  # Analyze all biomechanical data
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from universal_transformation_framework import analyze_oscillations
from biomechanical_oscillatory_system import BiomechanicalOscillatoryAnalyzer

def load_biomechanical_data(file_path: str) -> Dict[str, Any]:
    """Load and validate biomechanical data file"""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame if it's a list
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])  # Single record
    else:
        raise ValueError("Invalid data format")
    
    return {
        'raw_data': data,
        'dataframe': df,
        'n_records': len(df),
        'columns': list(df.columns),
        'file_path': file_path
    }

def analyze_gait_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze gait cycle and locomotion data"""
    
    print(f"🦶 Analyzing Gait Data: {data['file_path']}")
    print(f"   Records: {data['n_records']}")
    print(f"   Parameters: {', '.join(data['columns'])}")
    
    df = data['dataframe']
    
    # Initialize biomechanical analyzer
    bio_analyzer = BiomechanicalOscillatoryAnalyzer()
    
    results = {}
    
    # 1. Universal oscillatory analysis
    time_column = None
    for col in ['time', 'timestamp', 't']:
        if col in df.columns:
            time_column = col
            break
    
    if time_column is None and len(df) > 1:
        df['time_index'] = range(len(df))
        time_column = 'time_index'
    
    try:
        universal_results = analyze_oscillations(df, time_column=time_column)
        results['universal_analysis'] = universal_results
        
        print(f"   ✅ Universal analysis: {universal_results['meta_analysis']['overall_confidence']:.3f} confidence")
        
    except Exception as e:
        print(f"   ❌ Universal analysis failed: {e}")
        results['universal_analysis'] = {'error': str(e)}
    
    # 2. Biomechanical-specific analysis
    try:
        if isinstance(data['raw_data'], list):
            bio_results = bio_analyzer.analyze_gait_oscillations(data['raw_data'])
            results['biomechanical_analysis'] = bio_results
            
            print(f"   ✅ Biomechanical analysis complete")
            
            # Surface compliance analysis
            if 'surface_compliance_analysis' in bio_results:
                surface = bio_results['surface_compliance_analysis']['estimated_surface_compliance']
                print(f"   🏃 Estimated surface: {surface.surface_type.value} (compliance: {surface.compliance_factor:.3f})")
            
        else:
            print(f"   ⚠️  Biomechanical analysis requires list format data")
            
    except Exception as e:
        print(f"   ❌ Biomechanical analysis failed: {e}")
        results['biomechanical_analysis'] = {'error': str(e)}
    
    # 3. Gait-specific pattern identification
    gait_patterns = identify_gait_patterns(df)
    results['gait_patterns'] = gait_patterns
    
    if gait_patterns['detected_patterns']:
        print(f"   🎯 Gait patterns detected: {len(gait_patterns['detected_patterns'])}")
        for pattern in gait_patterns['detected_patterns'][:3]:
            print(f"      • {pattern['parameter']}: {pattern['frequency']:.3f} Hz ({pattern['type']})")
    
    return results

def analyze_knee_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze knee mechanics and joint oscillations"""
    
    print(f"🦵 Analyzing Knee Data: {data['file_path']}")
    print(f"   Records: {data['n_records']}")
    print(f"   Parameters: {', '.join(data['columns'])}")
    
    df = data['dataframe']
    results = {}
    
    # 1. Universal oscillatory analysis
    time_column = 'time' if 'time' in df.columns else 't' if 't' in df.columns else None
    
    if time_column is None:
        if len(df) > 1:
            df['time_index'] = np.linspace(0, 1, len(df))  # Normalized time
            time_column = 'time_index'
    
    try:
        universal_results = analyze_oscillations(df, time_column=time_column)
        results['universal_analysis'] = universal_results
        
        confidence = universal_results['meta_analysis']['overall_confidence']
        system_type = universal_results['meta_analysis']['system_classification']['type']
        print(f"   ✅ Universal analysis: {confidence:.3f} confidence, {system_type}")
        
    except Exception as e:
        print(f"   ❌ Universal analysis failed: {e}")
        results['universal_analysis'] = {'error': str(e)}
    
    # 2. Knee-specific analysis
    knee_analysis = analyze_knee_mechanics(df)
    results['knee_mechanics'] = knee_analysis
    
    if knee_analysis['oscillation_detected']:
        print(f"   ✅ Knee oscillation detected: {knee_analysis['dominant_frequency']:.3f} Hz")
        print(f"   📊 Range of motion: {knee_analysis['range_of_motion']:.1f} degrees")
    
    # 3. Joint coupling analysis
    if len(df.columns) > 2:  # Multiple parameters
        coupling_analysis = analyze_joint_coupling(df)
        results['joint_coupling'] = coupling_analysis
        
        if coupling_analysis['significant_couplings']:
            print(f"   🔗 Joint couplings detected: {len(coupling_analysis['significant_couplings'])}")
    
    return results

def analyze_stability_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze center of mass and stability dynamics"""
    
    print(f"⚖️  Analyzing Stability Data: {data['file_path']}")
    print(f"   Records: {data['n_records']}")
    print(f"   Parameters: {', '.join(data['columns'])}")
    
    df = data['dataframe']
    results = {}
    
    # 1. Universal oscillatory analysis
    time_column = 'time' if 'time' in df.columns else 't' if 't' in df.columns else None
    
    if time_column is None:
        df['time_index'] = range(len(df))
        time_column = 'time_index'
    
    try:
        universal_results = analyze_oscillations(df, time_column=time_column)
        results['universal_analysis'] = universal_results
        
        confidence = universal_results['meta_analysis']['overall_confidence']
        complexity = universal_results['meta_analysis']['system_classification']['complexity']
        print(f"   ✅ Universal analysis: {confidence:.3f} confidence, {complexity} complexity")
        
    except Exception as e:
        print(f"   ❌ Universal analysis failed: {e}")
        results['universal_analysis'] = {'error': str(e)}
    
    # 2. Stability-specific analysis
    stability_analysis = analyze_stability_dynamics(df)
    results['stability_analysis'] = stability_analysis
    
    if 'stability_metrics' in stability_analysis:
        metrics = stability_analysis['stability_metrics']
        print(f"   📈 Stability index: {metrics.get('overall_stability', 0):.3f}")
        print(f"   🌊 Oscillation amplitude: {metrics.get('oscillation_amplitude', 0):.4f}")
    
    return results

def identify_gait_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify gait-specific oscillatory patterns"""
    
    gait_parameters = {
        'cadence': (1.4, 2.0),      # Hz - steps per second
        'stance_time': (4.0, 6.0),  # Hz - reciprocal of stance duration
        'step_length': (0.5, 2.0),  # Hz - step length variations
        'speed': (0.1, 1.0),        # Hz - speed variations
        'vertical_oscillation': (2.8, 4.0),  # Hz - vertical COM movement
        'com_acceleration': (0.5, 3.0)       # Hz - COM acceleration patterns
    }
    
    detected_patterns = []
    
    for param, expected_range in gait_parameters.items():
        if param in df.columns:
            signal = df[param].values
            
            if len(signal) > 10:
                # Simple frequency analysis
                from scipy.fft import fft, fftfreq
                
                # Compute FFT
                fft_result = fft(signal - np.mean(signal))
                frequencies = fftfreq(len(signal), d=1.0)  # Assuming 1 Hz sampling
                
                # Find dominant frequency
                power_spectrum = np.abs(fft_result)
                positive_freqs = frequencies[frequencies > 0]
                positive_power = power_spectrum[frequencies > 0]
                
                if len(positive_freqs) > 0:
                    dominant_idx = np.argmax(positive_power)
                    dominant_freq = positive_freqs[dominant_idx]
                    
                    # Check if within expected range
                    pattern_type = 'expected' if expected_range[0] <= dominant_freq <= expected_range[1] else 'unexpected'
                    
                    detected_patterns.append({
                        'parameter': param,
                        'frequency': dominant_freq,
                        'expected_range': expected_range,
                        'type': pattern_type,
                        'power': positive_power[dominant_idx],
                        'signal_length': len(signal)
                    })
    
    return {
        'detected_patterns': detected_patterns,
        'n_parameters_analyzed': len([p for p in gait_parameters.keys() if p in df.columns]),
        'expected_matches': len([p for p in detected_patterns if p['type'] == 'expected']),
        'pattern_quality': len([p for p in detected_patterns if p['type'] == 'expected']) / max(1, len(detected_patterns))
    }

def analyze_knee_mechanics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze knee-specific mechanical patterns"""
    
    analysis = {
        'oscillation_detected': False,
        'dominant_frequency': 0,
        'range_of_motion': 0,
        'velocity_patterns': {},
        'acceleration_patterns': {}
    }
    
    # Knee angle analysis
    if 'knee_angle' in df.columns:
        angles = df['knee_angle'].values
        analysis['range_of_motion'] = np.max(angles) - np.min(angles)
        
        # Find oscillation frequency
        if len(angles) > 10:
            angles_centered = angles - np.mean(angles)
            
            # Simple period detection using autocorrelation
            autocorr = np.correlate(angles_centered, angles_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation (periods)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(autocorr[1:], height=np.max(autocorr)*0.3)
            
            if len(peaks) > 0:
                # Convert peak to frequency (assuming 1000 Hz sampling for knee data)
                period_samples = peaks[0] + 1
                frequency = 1000 / period_samples if period_samples > 0 else 0
                
                analysis['oscillation_detected'] = True
                analysis['dominant_frequency'] = frequency
    
    # Knee velocity analysis
    if 'knee_velocity' in df.columns:
        velocities = df['knee_velocity'].values
        
        analysis['velocity_patterns'] = {
            'mean_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'velocity_range': np.max(velocities) - np.min(velocities),
            'velocity_variability': np.std(velocities) / (np.mean(np.abs(velocities)) + 1e-10)
        }
    
    return analysis

def analyze_joint_coupling(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze coupling between joint parameters"""
    
    # Look for correlations between different parameters
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) < 2:
        return {'significant_couplings': []}
    
    correlations = []
    significant_couplings = []
    
    for i, col1 in enumerate(numeric_columns):
        for j, col2 in enumerate(numeric_columns):
            if i < j:  # Avoid duplicate pairs
                corr = np.corrcoef(df[col1].values, df[col2].values)[0, 1]
                
                correlations.append({
                    'param1': col1,
                    'param2': col2,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
                
                if abs(corr) > 0.5:  # Significant coupling threshold
                    significant_couplings.append({
                        'param1': col1,
                        'param2': col2,
                        'correlation': corr,
                        'coupling_strength': 'strong' if abs(corr) > 0.7 else 'moderate'
                    })
    
    return {
        'correlations': correlations,
        'significant_couplings': significant_couplings,
        'strongest_coupling': max(correlations, key=lambda x: x['abs_correlation']) if correlations else None
    }

def analyze_stability_dynamics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze center of mass and stability dynamics"""
    
    analysis = {}
    
    # COM acceleration analysis
    if 'com_acceleration' in df.columns:
        com_acc = df['com_acceleration'].values
        
        analysis['com_metrics'] = {
            'mean_acceleration': np.mean(com_acc),
            'rms_acceleration': np.sqrt(np.mean(com_acc**2)),
            'acceleration_variability': np.std(com_acc),
            'peak_acceleration': np.max(np.abs(com_acc))
        }
    
    # General stability metrics
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 0:
        # Calculate overall signal stability
        stability_scores = []
        
        for col in numeric_columns:
            signal = df[col].values
            if len(signal) > 5:
                # Stability as inverse of coefficient of variation
                cv = np.std(signal) / (np.abs(np.mean(signal)) + 1e-10)
                stability = 1 / (1 + cv)
                stability_scores.append(stability)
        
        if stability_scores:
            analysis['stability_metrics'] = {
                'overall_stability': np.mean(stability_scores),
                'stability_consistency': 1 - np.std(stability_scores),
                'oscillation_amplitude': np.mean([np.std(df[col].values) for col in numeric_columns])
            }
    
    return analysis

def generate_comprehensive_report(all_results: Dict[str, Dict]) -> str:
    """Generate comprehensive analysis report"""
    
    report = """
=== BIOMECHANICAL OSCILLATORY ANALYSIS REPORT ===

EXECUTIVE SUMMARY:
"""
    
    total_analyses = len(all_results)
    successful_analyses = len([r for r in all_results.values() if 'error' not in r.get('universal_analysis', {})])
    
    report += f"- Total data files analyzed: {total_analyses}\n"
    report += f"- Successful analyses: {successful_analyses}\n"
    report += f"- Analysis success rate: {successful_analyses/total_analyses*100:.1f}%\n\n"
    
    for data_type, results in all_results.items():
        report += f"{data_type.upper()} ANALYSIS:\n"
        
        if 'universal_analysis' in results and 'error' not in results['universal_analysis']:
            ua = results['universal_analysis']
            meta = ua.get('meta_analysis', {})
            
            report += f"  - System type: {meta.get('system_classification', {}).get('type', 'unknown')}\n"
            report += f"  - Complexity: {meta.get('system_classification', {}).get('complexity', 'unknown')}\n"
            report += f"  - Confidence: {meta.get('overall_confidence', 0):.3f}\n"
            
            if 'individual_signals' in ua:
                report += f"  - Signals analyzed: {len(ua['individual_signals'])}\n"
                
                for signal in ua['individual_signals']:
                    de_type = signal['transformation'].differential_equation.get('type', 'unknown')
                    conf = signal['transformation'].confidence_score
                    report += f"    • {signal['signal_name']}: {de_type} (conf: {conf:.3f})\n"
        
        # Add specific analysis results
        if 'gait_patterns' in results:
            gp = results['gait_patterns']
            if gp.get('detected_patterns'):
                report += f"  - Gait patterns detected: {len(gp['detected_patterns'])}\n"
                for pattern in gp['detected_patterns'][:3]:
                    report += f"    • {pattern['parameter']}: {pattern['frequency']:.3f} Hz\n"
        
        if 'knee_mechanics' in results:
            km = results['knee_mechanics']
            if km.get('oscillation_detected'):
                report += f"  - Knee oscillation: {km['dominant_frequency']:.3f} Hz\n"
                report += f"  - Range of motion: {km['range_of_motion']:.1f}°\n"
        
        if 'stability_metrics' in results.get('stability_analysis', {}):
            sm = results['stability_analysis']['stability_metrics']
            report += f"  - Stability index: {sm.get('overall_stability', 0):.3f}\n"
        
        report += "\n"
    
    # Overall insights
    report += "OVERALL INSIGHTS:\n"
    
    # Collect all confidence scores
    confidences = []
    for results in all_results.values():
        if 'universal_analysis' in results and 'meta_analysis' in results['universal_analysis']:
            conf = results['universal_analysis']['meta_analysis'].get('overall_confidence', 0)
            confidences.append(conf)
    
    if confidences:
        report += f"- Mean analysis confidence: {np.mean(confidences):.3f}\n"
        report += f"- Analysis consistency: {1 - np.std(confidences):.3f}\n"
    
    # System types
    system_types = []
    for results in all_results.values():
        if 'universal_analysis' in results and 'meta_analysis' in results['universal_analysis']:
            sys_type = results['universal_analysis']['meta_analysis'].get('system_classification', {}).get('type', 'unknown')
            system_types.append(sys_type)
    
    if system_types:
        from collections import Counter
        type_counts = Counter(system_types)
        report += f"- Dominant system types: {dict(type_counts)}\n"
    
    report += f"\nAnalysis completed using Universal Oscillatory Transformation Framework\n"
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Biomechanical Oscillatory Data Validation')
    parser.add_argument('--gait', help='Path to gait data file')
    parser.add_argument('--knee', help='Path to knee mechanics data file')
    parser.add_argument('--stability', help='Path to stability/COM data file')
    parser.add_argument('--all', action='store_true', help='Analyze all available biomechanical data')
    parser.add_argument('--output', default='biomechanical_analysis_results', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BIOMECHANICAL OSCILLATORY DATA VALIDATION")
    print("Universal Transformation Framework for Biomechanics")
    print("=" * 80)
    
    all_results = {}
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.all:
        # Analyze all biomechanical data
        data_files = [
            ('gait', 'experimental-data/gait/gait_cycle_track.json'),
            ('gait_combined', 'experimental-data/gait/combined_time_series.json'),
            ('knee_angle', 'experimental-data/knee/knee_angle.json'),
            ('knee_velocity', 'experimental-data/knee/knee_angle.json'),  # Same file, different analysis
            ('knee_mechanics', 'experimental-data/knee/knee_muscle_tendon_mechanics.json'),
            ('stability_com', 'experimental-data/stability/com.json'),
            ('stability_control', 'experimental-data/stability/control_loop.json'),
            ('stability_damped', 'experimental-data/stability/damped.json')
        ]
        
        for data_type, file_path in data_files:
            if os.path.exists(file_path):
                print(f"\n📂 Loading {data_type}: {file_path}")
                try:
                    data = load_biomechanical_data(file_path)
                    
                    if 'gait' in data_type:
                        results = analyze_gait_data(data)
                    elif 'knee' in data_type:
                        results = analyze_knee_data(data)
                    elif 'stability' in data_type:
                        results = analyze_stability_data(data)
                    else:
                        # Default universal analysis
                        results = {'universal_analysis': analyze_oscillations(data['dataframe'])}
                    
                    all_results[data_type] = results
                    
                except Exception as e:
                    print(f"❌ Error analyzing {data_type}: {e}")
                    all_results[data_type] = {'error': str(e)}
            else:
                print(f"⚠️  File not found: {file_path}")
    
    else:
        # Analyze specific files
        if args.gait:
            data = load_biomechanical_data(args.gait)
            all_results['gait'] = analyze_gait_data(data)
        
        if args.knee:
            data = load_biomechanical_data(args.knee)
            all_results['knee'] = analyze_knee_data(data)
        
        if args.stability:
            data = load_biomechanical_data(args.stability)
            all_results['stability'] = analyze_stability_data(data)
    
    if not all_results:
        print("❌ No data files specified. Use --gait, --knee, --stability, or --all")
        return
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 80)
    
    report = generate_comprehensive_report(all_results)
    print(report)
    
    # Save report to file
    report_path = os.path.join(args.output, 'biomechanical_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Full report saved to: {report_path}")
    print(f"📊 Analysis results directory: {args.output}/")
    
    # Summary statistics
    total_files = len(all_results)
    successful = len([r for r in all_results.values() if 'error' not in r])
    
    print(f"\n✨ Analysis complete!")
    print(f"   📈 {successful}/{total_files} files analyzed successfully")
    print(f"   🎯 Results saved to {args.output}/")

if __name__ == "__main__":
    main()
