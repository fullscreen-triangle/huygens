#!/usr/bin/env python3
"""
Experimental Biometric Data Analysis Runner

Main interface for analyzing real biometric data and validating 
consciousness and oscillatory theories against experimental measurements.

This revolutionary framework applies theoretical predictions to actual 
personal biometric data including heart rate, sleep, activity, and location data.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experimental_data.experimental_biometric_validator import ExperimentalBiometricValidator

def run_experimental_analysis():
    """
    Run experimental analysis of biometric data
    """
    print("🏃‍♂️💓 EXPERIMENTAL BIOMETRIC DATA ANALYSIS 💓🏃‍♂️")
    print("=" * 70)
    print("Analyzing real biometric data to validate consciousness theories!")
    print("=" * 70)
    
    # Check if experimental data directory exists
    experimental_data_dir = Path("../experimental-data")
    
    if not experimental_data_dir.exists():
        print(f"\n❌ Experimental data directory not found: {experimental_data_dir}")
        print("Please ensure your biometric data is placed in the experimental-data directory")
        return None
    
    print(f"\n📂 Using experimental data from: {experimental_data_dir}")
    
    # Initialize validator
    validator = ExperimentalBiometricValidator(
        experimental_data_dir=str(experimental_data_dir),
        results_dir="experimental_analysis_results"
    )
    
    # Run comprehensive experimental validation
    results = validator.run_comprehensive_experimental_validation()
    
    if results and not results.get('error'):
        print(f"\n🎯 EXPERIMENTAL ANALYSIS COMPLETED!")
        print(f"📁 Results saved to: {validator.results_dir}")
        
        # Print key findings
        summary = results.get('summary', {})
        
        print(f"\n📊 KEY FINDINGS:")
        print(f"  • Data Categories Loaded: {len(summary.get('loaded_data_categories', []))}")
        print(f"  • Experiments Completed: {summary.get('total_experiments', 0)}")
        print(f"  • Successful Validations: {summary.get('successful_validations', 0)}")
        print(f"  • Success Rate: {summary.get('validation_success_rate', 0)*100:.1f}%")
        print(f"  • Theoretical Predictions Validated: {summary.get('theoretical_predictions_validated', 0)}")
        
        if summary.get('experimental_validation_success', False):
            print(f"\n🏆 BREAKTHROUGH: REAL DATA VALIDATES CONSCIOUSNESS THEORIES! 🏆")
            print(f"Your biometric data provides experimental evidence for:")
            print(f"  ✓ Consciousness-frequency coupling (2-10 Hz)")
            print(f"  ✓ Multi-scale oscillatory patterns")
            print(f"  ✓ Sleep-consciousness state transitions")
            print(f"  ✓ Heart rate-consciousness coupling")
            print(f"  ✓ Activity-consciousness integration")
        else:
            print(f"\n📈 Partial validation achieved!")
            print(f"More complete biometric data could strengthen validation.")
    else:
        print(f"\n⚠️ Experimental analysis could not be completed")
        if results and 'error' in results:
            print(f"Error: {results['error']}")
    
    return validator, results

def analyze_specific_data_category(category):
    """
    Analyze a specific category of biometric data
    """
    print(f"\n🔬 ANALYZING {category.upper()} DATA")
    print("-" * 50)
    
    validator = ExperimentalBiometricValidator()
    
    # Load data
    loaded_categories = validator.load_experimental_data()
    
    if category not in loaded_categories:
        print(f"❌ {category} data not available")
        available = ", ".join(loaded_categories) if loaded_categories else "None"
        print(f"Available categories: {available}")
        return None
    
    # Run category-specific analysis
    if category == 'heart_rate':
        result = validator.validate_heart_rate_consciousness_coupling()
    elif category == 'sleep':
        result = validator.validate_sleep_consciousness_transitions()
    elif category == 'activity':
        result = validator.validate_activity_consciousness_coupling()
    else:
        print(f"⚠️ No specific analysis available for {category}")
        return None
    
    print(f"\n📊 {category.upper()} ANALYSIS RESULTS:")
    
    if result.get('validation_success', False):
        print(f"✅ Validation successful!")
        
        # Print key metrics based on category
        if category == 'heart_rate':
            if 'basic_stats' in result:
                stats = result['basic_stats']
                print(f"  • Mean HR: {stats['mean_hr']:.1f} bpm")
                print(f"  • HR Range: {stats['min_hr']:.0f}-{stats['max_hr']:.0f} bpm")
            
            if 'frequency_analysis' in result:
                freq_analysis = result['frequency_analysis']
                print(f"  • Consciousness Band Power: {freq_analysis['consciousness_band_power']:.3f}")
        
        elif category == 'sleep':
            if 'sleep_architecture' in result:
                arch = result['sleep_architecture']
                print(f"  • Sleep Records: {arch['total_sleep_records']}")
                print(f"  • Sleep Stages: {', '.join(arch['unique_stages'])}")
        
        elif category == 'activity':
            if 'circadian_activity' in result:
                circ = result['circadian_activity']
                print(f"  • Peak Activity Hour: {circ['peak_activity_hour']}")
                print(f"  • Activity Amplitude: {circ['activity_amplitude']:.2f}")
        
    else:
        print(f"❌ Validation failed or incomplete")
    
    return result

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(
        description="Experimental Biometric Data Analysis for Consciousness Theory Validation"
    )
    
    parser.add_argument(
        '--mode', 
        choices=['comprehensive', 'category-specific', 'quick'],
        default='comprehensive',
        help='Analysis mode to run'
    )
    
    parser.add_argument(
        '--category',
        choices=['heart_rate', 'sleep', 'activity', 'geolocation'],
        help='Specific data category to analyze (for category-specific mode)'
    )
    
    parser.add_argument(
        '--data-dir',
        default='../experimental-data',
        help='Path to experimental data directory'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'comprehensive':
        validator, results = run_experimental_analysis()
    elif args.mode == 'category-specific':
        if not args.category:
            print("❌ Must specify --category for category-specific analysis")
            return
        result = analyze_specific_data_category(args.category)
    elif args.mode == 'quick':
        print("🚀 Quick experimental analysis not yet implemented")
        print("Use --mode comprehensive for full analysis")
        return
    
    print(f"\n🌟 Analysis complete!")

if __name__ == "__main__":
    main()
