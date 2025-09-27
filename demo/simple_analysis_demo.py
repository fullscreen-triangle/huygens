#!/usr/bin/env python3
"""
Simple Universal Transformation Framework Demo
Shows how to analyze any data source with just a few lines of code

Usage examples with your real data:
    python simple_analysis_demo.py --data ../experimental-data/actigraphy/actigraphy.json --time timestamp
    python simple_analysis_demo.py --data ../experimental-data/sleep/sleepRecords.json --time timestamp
    python simple_analysis_demo.py --data your_data.csv --time time_column
"""

import sys
import os
import argparse
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from universal_transformation_framework import analyze_oscillations

def main():
    parser = argparse.ArgumentParser(description='Universal Oscillatory Analysis')
    parser.add_argument('--data', required=True, help='Path to data file (JSON/CSV)')
    parser.add_argument('--time', help='Name of time column (auto-detected if not provided)')
    parser.add_argument('--columns', nargs='*', help='Specific columns to analyze (all numeric if not provided)')
    parser.add_argument('--output', default='analysis_results', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("UNIVERSAL OSCILLATORY TRANSFORMATION FRAMEWORK")
    print("=" * 80)
    print(f"Analyzing: {args.data}")
    print(f"Time column: {args.time or 'auto-detect'}")
    print(f"Target columns: {args.columns or 'all numeric'}")
    print()
    
    # Configure framework
    config = {
        'output_directory': args.output,
        'generate_visualizations': True,
        'pattern_discovery': True
    }
    
    try:
        # Run analysis - THIS IS ALL YOU NEED!
        results = analyze_oscillations(
            data_source=args.data,
            time_column=args.time,
            target_columns=args.columns,
            config=config
        )
        
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            return
        
        # Display summary
        print("✅ Analysis completed successfully!")
        print(f"⏱️  Execution time: {results['execution_time']:.2f} seconds")
        print()
        
        # Key results
        meta = results['meta_analysis']
        print("🔍 KEY FINDINGS:")
        print(f"   System Type: {meta['system_classification']['type']}")
        print(f"   Complexity: {meta['system_classification']['complexity']}")
        print(f"   Overall Confidence: {meta['overall_confidence']:.3f}")
        print(f"   Signals Analyzed: {len(results['individual_signals'])}")
        print()
        
        # Pattern discoveries
        total_patterns = sum(len(sr['transformation'].pattern_discovery.get('validated_patterns', [])) 
                           for sr in results['individual_signals'])
        print(f"🎯 PATTERNS DISCOVERED: {total_patterns}")
        
        for signal_result in results['individual_signals']:
            signal_name = signal_result['signal_name']
            patterns = signal_result['transformation'].pattern_discovery.get('validated_patterns', [])
            confidence = signal_result['transformation'].confidence_score
            
            print(f"   {signal_name}: {len(patterns)} patterns (confidence: {confidence:.3f})")
            for pattern in patterns[:2]:  # Show top 2
                freqs = pattern.frequency_components[:3] if pattern.frequency_components else []
                freq_str = ", ".join(f"{f:.4f} Hz" for f in freqs)
                print(f"     • {pattern.pattern_type.value}: {freq_str}")
        
        print()
        
        # Differential equations
        print("📐 DIFFERENTIAL EQUATIONS:")
        for signal_result in results['individual_signals']:
            de = signal_result['transformation'].differential_equation
            print(f"   {signal_result['signal_name']}: {de.get('symbolic', 'N/A')}")
        
        print()
        
        # Recommendations  
        print("💡 RECOMMENDATIONS:")
        recommendations = meta['recommendations']['next_steps']
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. {rec}")
        
        print()
        print(f"📁 Detailed results saved to: {config['output_directory']}/")
        print(f"📊 Visualizations: {config['output_directory']}/*.png")
        print(f"📄 Full report: See results['summary_report']")
        
        # Save summary to file
        with open(os.path.join(config['output_directory'], 'analysis_summary.txt'), 'w') as f:
            f.write(results['summary_report'])
        
        print("\n✨ Analysis complete! Check the output directory for detailed results.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
