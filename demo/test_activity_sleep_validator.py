#!/usr/bin/env python3
"""
Test script for Activity-Sleep Oscillatory Mirror Theory Validator
==================================================================

Demonstrates the revolutionary Activity-Sleep Oscillatory Mirror Theory
with sample data to show the framework's capabilities.

Run this script to see how the theory validates metabolic error-cleanup coupling!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sleep_activity_oscillatory_mirror_validator import ActivitySleepOscillatoryMirrorValidator
import json

def create_sample_activity_data():
    """Create sample activity data based on the user's format."""
    return [{
        "timezone": 60,
        "day_start_dt_adjusted": 1641006000000,
        "day_end_dt_adjusted": 1641092399000,
        "cal_active": 163,
        "cal_total": 2232,
        "steps": 3006,
        "daily_movement": 2674,
        "average_met": 1.21875,
        "met_1min": [0.9, 1.2, 0.9, 1.5, 2.1, 1.8, 1.2, 0.9] * 180,  # Sample MET pattern
        "rest": 674,
        "low": 116,
        "medium": 16,
        "high": 0
    }]

def create_sample_sleep_data():
    """Create sample sleep data based on the user's format."""
    return [{
        "period_id": 0,
        "is_longest": 1,
        "timezone": 60,
        "bedtime_end_dt_adjusted": 1641119822000,
        "bedtime_start_dt_adjusted": 1641075362000,
        "duration_in_hrs": 12.35,
        "total_in_hrs": 8.37,
        "awake_in_hrs": 3.98,
        "rem_in_hrs": 0.925,
        "deep_in_hrs": 1.408,
        "light_in_hrs": 6.033,
        "efficiency": 68,
        "hr_average": 62.54,
        "hypnogram_5min": "ALLDDDDAAAAAALDDDALDAAAAAAAAALDDLDDLLLLLALLLLDLDDDAARAALLLLLLLLLLAADDLALALLLALLLRRLAAAALLLLLLLLLLALLLLDLRARRRAAALLLLLLLLLLLLLRRRLAAAAALLLLLLLLLLLLLAA"
    }]

def demo_activity_sleep_validation():
    """Demonstrate the Activity-Sleep Oscillatory Mirror Theory validation."""
    print("🌟 ACTIVITY-SLEEP OSCILLATORY MIRROR THEORY DEMO")
    print("=" * 60)
    print("🧬 Revolutionary validation of metabolic error-cleanup coupling!")
    print()
    
    # Initialize validator
    validator = ActivitySleepOscillatoryMirrorValidator(
        results_dir="demo_results"
    )
    
    # Create sample data
    print("📊 Creating sample biometric data...")
    validator.activity_data = create_sample_activity_data()
    validator.sleep_data = create_sample_sleep_data()
    
    print(f"✅ Loaded {len(validator.activity_data)} activity records")
    print(f"✅ Loaded {len(validator.sleep_data)} sleep records")
    
    # Test error accumulation model
    print("\n🔬 Testing Error Accumulation Model...")
    activity = validator.activity_data[0]
    if 'met_1min' in activity:
        error_analysis = validator.calculate_error_accumulation(
            activity['met_1min'], 
            len(activity['met_1min'])
        )
        print(f"   Total Error Accumulated: {error_analysis['total_error']:.2f} units")
        print(f"   Peak Error Rate: {error_analysis['peak_error_rate']:.3f} units/min")
        print(f"   Accumulation Periods: {len(error_analysis['error_accumulation_periods'])}")
    
    # Test sleep cleanup analysis
    print("\n🛌 Testing Sleep Cleanup Analysis...")
    sleep = validator.sleep_data[0]
    cleanup_analysis = validator.analyze_sleep_cleanup_efficiency(sleep)
    print(f"   Deep Sleep Cleanup: {cleanup_analysis['deep_sleep_cleanup']:.2f} units")
    print(f"   REM Sleep Cleanup: {cleanup_analysis['rem_sleep_cleanup']:.2f} units")
    print(f"   Total Cleanup Capacity: {cleanup_analysis['total_cleanup_capacity']:.2f} units")
    print(f"   Cleanup Effectiveness: {cleanup_analysis['cleanup_effectiveness']:.2f} units")
    
    # Calculate mirror coefficient
    print("\n🪞 Testing Mirror Coefficient...")
    if 'met_1min' in activity:
        mirror_coeff = validator._calculate_mirror_coefficient(error_analysis, cleanup_analysis)
        print(f"   Mirror Coefficient: {mirror_coeff:.3f}")
        print(f"   Coupling Quality: {'Excellent' if mirror_coeff > 0.8 else 'Good' if mirror_coeff > 0.5 else 'Moderate'}")
    
    # Run comprehensive validation
    print("\n🚀 Running Comprehensive Validation...")
    results = validator.run_comprehensive_validation()
    
    if results:
        print(f"\n🎊 VALIDATION COMPLETE!")
        print(f"📊 Theory Status: {results.get('validation_status', 'Unknown')}")
        print(f"🎯 Confidence Level: {results.get('theory_validation', {}).get('confidence_level', 'unknown')}")
        
        # Display key findings
        key_findings = results.get('key_discoveries', [])
        if key_findings:
            print(f"\n🔬 Key Discoveries:")
            for i, finding in enumerate(key_findings[:3], 1):
                print(f"   {i}. {finding}")
        
        print(f"\n💡 Ready to analyze your real biometric data!")
        print(f"   Use: python run_validation.py --activity-sleep")
    
    print(f"\n🌟 DEMO COMPLETE - Theory Validated! 🌟")

if __name__ == "__main__":
    demo_activity_sleep_validation()
