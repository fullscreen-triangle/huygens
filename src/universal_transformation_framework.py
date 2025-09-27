"""
Universal Transformation Framework
Master controller for oscillatory analysis of any data source
Simple interface: provide data source and time column
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
import os

from universal_oscillatory_transform import (
    TransformationResult, DifferentialForm, 
    UniversalDataIngestion, DifferentialEquationGenerator,
    LaplaceTransformEngine, SEntropyAnalyzer
)
from pattern_discovery_engine import PatternDiscoveryEngine, OscillatoryPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalTransformationFramework:
    """
    Universal oscillatory transformation framework
    Transforms any data into differential equations, applies s-entropy analysis,
    and discovers patterns without prior system knowledge
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize framework with configuration"""
        
        self.config = config or self._default_config()
        
        # Initialize components
        self.ingestion = UniversalDataIngestion()
        self.de_generator = DifferentialEquationGenerator()
        self.laplace_engine = LaplaceTransformEngine()
        self.s_entropy = SEntropyAnalyzer()
        self.pattern_engine = PatternDiscoveryEngine()
        
        # Results storage
        self.last_transformation = None
        self.transformation_history = []
        
        logger.info("Universal Transformation Framework initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'differential_forms': [DifferentialForm.TIME, DifferentialForm.INFO, DifferentialForm.ENTROPY],
            'pattern_discovery': True,
            'cross_validation': True,
            'generate_visualizations': True,
            'output_directory': 'transformation_results',
            'confidence_threshold': 0.5,
            'max_data_points': 50000,  # Memory management
            'auto_downsample': True
        }
    
    def transform(self, data_source: Union[str, pd.DataFrame, np.ndarray, Dict],
                 time_column: Optional[str] = None,
                 target_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main transformation function - the simple interface requested
        
        Args:
            data_source: Any data source (file path, DataFrame, array, dict)
            time_column: Name of time column (auto-detected if None)
            target_columns: Specific columns to analyze (all numeric if None)
        
        Returns:
            Comprehensive transformation and analysis results
        """
        
        logger.info("Starting universal transformation...")
        start_time = datetime.now()
        
        try:
            # Step 1: Data Ingestion
            logger.info("Step 1: Universal Data Ingestion")
            time_vector, data_matrix, metadata = self._ingest_and_prepare_data(
                data_source, time_column, target_columns
            )
            
            # Step 2: Multi-Signal Analysis
            logger.info("Step 2: Multi-Signal Transformation")
            transformation_results = []
            
            for i in range(data_matrix.shape[1]):
                signal_data = data_matrix[:, i]
                signal_name = metadata['column_names'][i]
                
                logger.info(f"  Analyzing signal: {signal_name}")
                
                # Generate differential equation
                de_result = self.de_generator.generate_first_order_de(time_vector, signal_data)
                
                # Laplace transform
                laplace_result = self.laplace_engine.transform_to_laplace(de_result, time_vector, signal_data)
                
                # S-entropy analysis in all forms
                entropy_results = {}
                for diff_form in self.config['differential_forms']:
                    entropy_results[diff_form.value] = self.s_entropy.analyze_s_entropy(
                        time_vector, signal_data, diff_form
                    )
                
                # Create transformation result
                transform_result = TransformationResult(
                    original_data=signal_data,
                    differential_equation=de_result,
                    laplace_transform=laplace_result,
                    s_entropy_analysis=entropy_results[DifferentialForm.TIME.value],  # Primary form
                    pattern_discovery={},  # Filled in next step
                    oscillatory_features={},  # Filled in next step
                    confidence_score=0.0  # Computed later
                )
                
                # Pattern discovery
                if self.config['pattern_discovery']:
                    logger.info(f"    Pattern discovery for {signal_name}")
                    pattern_results = self.pattern_engine.discover_patterns(transform_result)
                    transform_result.pattern_discovery = pattern_results
                
                # Compute overall confidence
                transform_result.confidence_score = self._compute_overall_confidence(transform_result)
                
                transformation_results.append({
                    'signal_name': signal_name,
                    'transformation': transform_result,
                    'entropy_all_forms': entropy_results
                })
            
            # Step 3: Cross-Signal Analysis
            logger.info("Step 3: Cross-Signal Analysis")
            cross_analysis = self._perform_cross_signal_analysis(transformation_results, time_vector)
            
            # Step 4: Meta-Analysis
            logger.info("Step 4: Meta-Analysis and Synthesis")
            meta_analysis = self._perform_meta_analysis(transformation_results, cross_analysis)
            
            # Step 5: Generate Comprehensive Results
            comprehensive_results = {
                'metadata': metadata,
                'individual_signals': transformation_results,
                'cross_signal_analysis': cross_analysis,
                'meta_analysis': meta_analysis,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'framework_version': '1.0.0',
                'configuration': self.config
            }
            
            # Store results
            self.last_transformation = comprehensive_results
            self.transformation_history.append(comprehensive_results)
            
            # Generate outputs if requested
            if self.config['generate_visualizations']:
                self._generate_visualizations(comprehensive_results)
            
            # Generate summary report
            summary = self._generate_summary_report(comprehensive_results)
            comprehensive_results['summary_report'] = summary
            
            logger.info(f"Universal transformation completed in {comprehensive_results['execution_time']:.2f} seconds")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Transformation failed: {str(e)}")
            return {
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'success': False
            }
    
    def _ingest_and_prepare_data(self, data_source: Any, time_column: Optional[str],
                                target_columns: Optional[List[str]]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Ingest and prepare data for transformation"""
        
        # Basic ingestion
        time_vector, data_matrix = self.ingestion.ingest_data(data_source, time_column)
        
        # Handle target columns
        if isinstance(data_source, pd.DataFrame):
            all_columns = data_source.columns.tolist()
            numeric_columns = data_source.select_dtypes(include=[np.number]).columns.tolist()
            
            if target_columns:
                # Validate target columns
                invalid_columns = [col for col in target_columns if col not in all_columns]
                if invalid_columns:
                    raise ValueError(f"Target columns not found: {invalid_columns}")
                
                # Filter to target columns
                target_indices = [all_columns.index(col) for col in target_columns if col in numeric_columns]
                data_matrix = data_matrix[:, target_indices]
                column_names = [col for col in target_columns if col in numeric_columns]
            else:
                column_names = [col for col in numeric_columns if col != time_column]
        else:
            # Generate column names for other data types
            n_signals = data_matrix.shape[1] if data_matrix.ndim > 1 else 1
            column_names = [f'signal_{i}' for i in range(n_signals)]
            if data_matrix.ndim == 1:
                data_matrix = data_matrix.reshape(-1, 1)
        
        # Data quality checks and preprocessing
        original_length = len(time_vector)
        
        # Remove NaN values
        valid_mask = ~np.isnan(data_matrix).any(axis=1) & ~np.isnan(time_vector)
        time_vector = time_vector[valid_mask]
        data_matrix = data_matrix[valid_mask]
        
        # Downsample if too large
        if len(time_vector) > self.config['max_data_points'] and self.config['auto_downsample']:
            downsample_factor = len(time_vector) // self.config['max_data_points']
            time_vector = time_vector[::downsample_factor]
            data_matrix = data_matrix[::downsample_factor]
            logger.info(f"Downsampled data by factor {downsample_factor}")
        
        # Generate metadata
        metadata = {
            'original_length': original_length,
            'processed_length': len(time_vector),
            'n_signals': data_matrix.shape[1],
            'column_names': column_names,
            'time_span': time_vector[-1] - time_vector[0] if len(time_vector) > 1 else 0,
            'sampling_rate': len(time_vector) / (time_vector[-1] - time_vector[0]) if len(time_vector) > 1 and time_vector[-1] != time_vector[0] else 1,
            'data_quality': self._assess_data_quality(time_vector, data_matrix)
        }
        
        return time_vector, data_matrix, metadata
    
    def _assess_data_quality(self, time_vector: np.ndarray, data_matrix: np.ndarray) -> Dict:
        """Assess quality of input data"""
        
        # Temporal regularity
        time_diffs = np.diff(time_vector)
        temporal_regularity = 1 - (np.std(time_diffs) / (np.mean(time_diffs) + 1e-10))
        
        # Signal quality metrics
        signal_qualities = []
        for i in range(data_matrix.shape[1]):
            signal = data_matrix[:, i]
            
            # SNR estimation (simplified)
            signal_power = np.var(signal)
            noise_estimate = np.var(np.diff(signal))  # Rough noise estimate
            snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
            
            # Dynamic range
            dynamic_range = (np.max(signal) - np.min(signal)) / (np.std(signal) + 1e-10)
            
            signal_qualities.append({
                'snr_db': snr,
                'dynamic_range': dynamic_range,
                'entropy': -np.sum((signal/np.sum(np.abs(signal))) * np.log2(np.abs(signal/np.sum(np.abs(signal))) + 1e-10))
            })
        
        return {
            'temporal_regularity': temporal_regularity,
            'mean_snr': np.mean([sq['snr_db'] for sq in signal_qualities]),
            'mean_dynamic_range': np.mean([sq['dynamic_range'] for sq in signal_qualities]),
            'overall_quality_score': (temporal_regularity + min(1, np.mean([sq['snr_db'] for sq in signal_qualities])/20)) / 2
        }
    
    def _perform_cross_signal_analysis(self, transformation_results: List[Dict], 
                                     time_vector: np.ndarray) -> Dict:
        """Perform cross-signal correlation and coupling analysis"""
        
        if len(transformation_results) < 2:
            return {'message': 'Single signal - no cross-analysis possible'}
        
        n_signals = len(transformation_results)
        
        # Extract signals
        signals = np.column_stack([tr['transformation'].original_data for tr in transformation_results])
        signal_names = [tr['signal_name'] for tr in transformation_results]
        
        # Cross-correlation analysis
        cross_correlations = np.corrcoef(signals.T)
        
        # Phase coupling analysis
        phase_couplings = np.zeros((n_signals, n_signals))
        
        for i in range(n_signals):
            for j in range(i+1, n_signals):
                signal_i = signals[:, i]
                signal_j = signals[:, j]
                
                # Hilbert transform for phase
                from scipy.signal import hilbert
                analytic_i = hilbert(signal_i)
                analytic_j = hilbert(signal_j)
                
                phase_i = np.angle(analytic_i)
                phase_j = np.angle(analytic_j)
                
                # Phase locking value
                phase_diff = phase_i - phase_j
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                
                phase_couplings[i, j] = plv
                phase_couplings[j, i] = plv
        
        # Frequency domain coupling
        frequency_couplings = {}
        
        for i in range(n_signals):
            for j in range(i+1, n_signals):
                # Get dominant frequencies from each signal
                trans_i = transformation_results[i]['transformation']
                trans_j = transformation_results[j]['transformation']
                
                if 'frequency_response' in trans_i.laplace_transform:
                    freq_i = trans_i.laplace_transform['frequency_response']['dominant_frequencies']
                    freq_j = trans_j.laplace_transform['frequency_response']['dominant_frequencies']
                    
                    # Check for harmonic relationships
                    harmonic_relationships = []
                    for fi in freq_i[:3]:  # Top 3 frequencies
                        for fj in freq_j[:3]:
                            if fi > 0 and fj > 0:
                                ratio = max(fi, fj) / min(fi, fj)
                                if abs(ratio - round(ratio)) < 0.1:  # Close to integer ratio
                                    harmonic_relationships.append({
                                        'freq_i': fi,
                                        'freq_j': fj,
                                        'ratio': ratio,
                                        'harmonic_order': int(round(ratio))
                                    })
                    
                    frequency_couplings[f"{signal_names[i]}_{signal_names[j]}"] = harmonic_relationships
        
        # Entropy coupling
        entropy_couplings = {}
        
        for i in range(n_signals):
            for j in range(i+1, n_signals):
                entropy_i = transformation_results[i]['transformation'].s_entropy_analysis['total_s_entropy']
                entropy_j = transformation_results[j]['transformation'].s_entropy_analysis['total_s_entropy']
                
                # Mutual information between entropy series (simplified)
                entropy_coupling = abs(entropy_i - entropy_j) / (entropy_i + entropy_j + 1e-10)
                entropy_couplings[f"{signal_names[i]}_{signal_names[j]}"] = entropy_coupling
        
        return {
            'cross_correlations': cross_correlations,
            'phase_couplings': phase_couplings,
            'frequency_couplings': frequency_couplings,
            'entropy_couplings': entropy_couplings,
            'signal_names': signal_names,
            'strongest_coupling': self._find_strongest_coupling(cross_correlations, signal_names)
        }
    
    def _find_strongest_coupling(self, correlation_matrix: np.ndarray, signal_names: List[str]) -> Dict:
        """Find the strongest coupling between signals"""
        
        n = len(signal_names)
        max_corr = 0
        max_pair = ('', '')
        
        for i in range(n):
            for j in range(i+1, n):
                corr = abs(correlation_matrix[i, j])
                if corr > max_corr:
                    max_corr = corr
                    max_pair = (signal_names[i], signal_names[j])
        
        return {
            'signal_pair': max_pair,
            'correlation': max_corr,
            'coupling_strength': 'strong' if max_corr > 0.7 else 'moderate' if max_corr > 0.4 else 'weak'
        }
    
    def _perform_meta_analysis(self, transformation_results: List[Dict], 
                              cross_analysis: Dict) -> Dict:
        """Perform meta-analysis across all signals and transformations"""
        
        # Collect all differential equation types
        de_types = [tr['transformation'].differential_equation['type'] for tr in transformation_results]
        dominant_de_type = max(set(de_types), key=de_types.count)
        
        # Collect all pattern types
        all_patterns = []
        for tr in transformation_results:
            if 'validated_patterns' in tr['transformation'].pattern_discovery:
                all_patterns.extend(tr['transformation'].pattern_discovery['validated_patterns'])
        
        pattern_types = [p.pattern_type.value for p in all_patterns]
        dominant_pattern_type = max(set(pattern_types), key=pattern_types.count) if pattern_types else 'none'
        
        # Overall system classification
        system_classification = self._classify_overall_system(
            dominant_de_type, dominant_pattern_type, cross_analysis
        )
        
        # Confidence metrics
        individual_confidences = [tr['transformation'].confidence_score for tr in transformation_results]
        overall_confidence = np.mean(individual_confidences)
        
        # Interchangeable form analysis across signals
        form_analysis = self._analyze_interchangeable_forms_meta(transformation_results)
        
        return {
            'system_classification': system_classification,
            'dominant_de_type': dominant_de_type,
            'dominant_pattern_type': dominant_pattern_type,
            'overall_confidence': overall_confidence,
            'confidence_distribution': {
                'mean': overall_confidence,
                'std': np.std(individual_confidences),
                'min': min(individual_confidences),
                'max': max(individual_confidences)
            },
            'interchangeable_forms_meta': form_analysis,
            'complexity_metrics': self._compute_system_complexity(transformation_results, all_patterns),
            'recommendations': self._generate_system_recommendations(system_classification, overall_confidence)
        }
    
    def _classify_overall_system(self, de_type: str, pattern_type: str, cross_analysis: Dict) -> Dict:
        """Classify the overall system based on all analyses"""
        
        # Base classification from differential equation
        if de_type == 'oscillatory':
            base_class = 'oscillatory_system'
        elif de_type == 'nonlinear':
            base_class = 'nonlinear_system'
        else:
            base_class = 'linear_system'
        
        # Modify based on patterns
        if pattern_type == 'chaotic':
            system_type = 'chaotic_system'
            complexity = 'high'
        elif pattern_type == 'fractal':
            system_type = 'fractal_system'
            complexity = 'very_high'
        elif pattern_type == 'hybrid':
            system_type = 'hybrid_system'
            complexity = 'high'
        elif pattern_type == 'periodic':
            system_type = 'periodic_system'
            complexity = 'low'
        else:
            system_type = base_class
            complexity = 'moderate'
        
        # Consider cross-signal coupling
        if 'strongest_coupling' in cross_analysis:
            coupling_strength = cross_analysis['strongest_coupling']['coupling_strength']
            if coupling_strength == 'strong':
                system_type += '_coupled'
                complexity = 'high' if complexity == 'moderate' else complexity
        
        return {
            'type': system_type,
            'complexity': complexity,
            'base_classification': base_class,
            'pattern_influence': pattern_type,
            'coupling_influence': cross_analysis.get('strongest_coupling', {}).get('coupling_strength', 'none')
        }
    
    def _analyze_interchangeable_forms_meta(self, transformation_results: List[Dict]) -> Dict:
        """Analyze interchangeable forms across all signals"""
        
        form_scores = {form.value: [] for form in DifferentialForm}
        
        for tr in transformation_results:
            entropy_forms = tr['entropy_all_forms']
            
            for form_name, entropy_data in entropy_forms.items():
                if 'total_s_entropy' in entropy_data:
                    form_scores[form_name].append(entropy_data['total_s_entropy'])
        
        # Compute consistency across signals for each form
        form_consistency = {}
        form_dominance = {}
        
        for form_name, scores in form_scores.items():
            if scores:
                form_consistency[form_name] = 1 - (np.std(scores) / (np.mean(scores) + 1e-10))
                form_dominance[form_name] = np.mean(scores)
        
        # Find most consistent and dominant form
        most_consistent_form = max(form_consistency.keys(), key=lambda k: form_consistency[k]) if form_consistency else 'dt'
        most_dominant_form = max(form_dominance.keys(), key=lambda k: form_dominance[k]) if form_dominance else 'dt'
        
        return {
            'form_consistency': form_consistency,
            'form_dominance': form_dominance,
            'most_consistent_form': most_consistent_form,
            'most_dominant_form': most_dominant_form,
            'interchangeability_score': np.mean(list(form_consistency.values())) if form_consistency else 0
        }
    
    def _compute_system_complexity(self, transformation_results: List[Dict], all_patterns: List) -> Dict:
        """Compute overall system complexity metrics"""
        
        # Count different types of elements
        n_signals = len(transformation_results)
        n_patterns = len(all_patterns)
        
        # Frequency domain complexity
        total_frequencies = 0
        for tr in transformation_results:
            if 'frequency_response' in tr['transformation'].laplace_transform:
                freq_resp = tr['transformation'].laplace_transform['frequency_response']
                total_frequencies += len(freq_resp.get('dominant_frequencies', []))
        
        # Entropy complexity
        entropy_complexities = []
        for tr in transformation_results:
            entropy_data = tr['transformation'].s_entropy_analysis
            if 'tridimensional_matrix' in entropy_data:
                condition_num = entropy_data['tridimensional_matrix'].get('condition_number', 1)
                entropy_complexities.append(condition_num)
        
        mean_entropy_complexity = np.mean(entropy_complexities) if entropy_complexities else 1
        
        # Overall complexity score
        complexity_score = (
            n_signals * 0.2 +  # Multi-signal adds complexity
            n_patterns * 0.3 +  # More patterns = more complex
            total_frequencies * 0.1 +  # Frequency richness
            np.log10(mean_entropy_complexity + 1) * 0.4  # Entropy complexity
        )
        
        return {
            'n_signals': n_signals,
            'n_patterns': n_patterns,
            'total_frequencies': total_frequencies,
            'mean_entropy_complexity': mean_entropy_complexity,
            'complexity_score': complexity_score,
            'complexity_category': self._categorize_complexity(complexity_score)
        }
    
    def _categorize_complexity(self, score: float) -> str:
        """Categorize complexity score"""
        if score < 2:
            return 'simple'
        elif score < 5:
            return 'moderate'
        elif score < 10:
            return 'complex'
        else:
            return 'very_complex'
    
    def _generate_system_recommendations(self, classification: Dict, confidence: float) -> Dict:
        """Generate recommendations for further analysis"""
        
        recommendations = []
        
        # Based on system type
        system_type = classification['type']
        
        if 'chaotic' in system_type:
            recommendations.extend([
                'Perform Lyapunov exponent analysis',
                'Reconstruct phase space attractors',
                'Analyze correlation dimension'
            ])
        elif 'fractal' in system_type:
            recommendations.extend([
                'Compute fractal dimension using box-counting',
                'Perform multifractal analysis',
                'Calculate Hurst exponent'
            ])
        elif 'oscillatory' in system_type:
            recommendations.extend([
                'Detailed harmonic analysis',
                'Phase-amplitude coupling analysis',
                'Resonance characterization'
            ])
        
        # Based on coupling
        if 'coupled' in system_type:
            recommendations.extend([
                'Network analysis of signal interactions',
                'Granger causality testing',
                'Synchronization analysis'
            ])
        
        # Based on confidence
        if confidence < 0.5:
            recommendations.extend([
                'Improve data quality (reduce noise, increase duration)',
                'Increase sampling rate',
                'Apply preprocessing filters'
            ])
        elif confidence > 0.8:
            recommendations.extend([
                'Proceed with predictive modeling',
                'Parameter estimation and validation',
                'Real-time analysis implementation'
            ])
        
        return {
            'analysis_recommendations': recommendations,
            'confidence_assessment': 'high' if confidence > 0.7 else 'moderate' if confidence > 0.4 else 'low',
            'next_steps': self._prioritize_next_steps(recommendations, classification, confidence)
        }
    
    def _prioritize_next_steps(self, recommendations: List[str], classification: Dict, confidence: float) -> List[str]:
        """Prioritize next analysis steps"""
        
        priority_order = []
        
        # High priority based on confidence
        if confidence < 0.5:
            priority_order.extend([rec for rec in recommendations if 'data quality' in rec or 'preprocessing' in rec])
        
        # Medium priority based on system type
        system_specific = [rec for rec in recommendations if any(word in rec.lower() 
                          for word in ['lyapunov', 'fractal', 'harmonic', 'network'])]
        priority_order.extend(system_specific[:2])  # Top 2 system-specific
        
        # Lower priority
        remaining = [rec for rec in recommendations if rec not in priority_order]
        priority_order.extend(remaining[:3])  # Add up to 3 more
        
        return priority_order[:5]  # Return top 5 priorities
    
    def _compute_overall_confidence(self, transform_result: TransformationResult) -> float:
        """Compute overall confidence score for a transformation"""
        
        # Differential equation fit quality
        de_confidence = transform_result.differential_equation.get('goodness_of_fit', 0.5)
        
        # Pattern discovery confidence
        if 'validated_patterns' in transform_result.pattern_discovery:
            patterns = transform_result.pattern_discovery['validated_patterns']
            pattern_confidence = np.mean([p.confidence for p in patterns]) if patterns else 0.3
        else:
            pattern_confidence = 0.3
        
        # Entropy analysis stability
        entropy_data = transform_result.s_entropy_analysis
        if 'interchangeable_analysis' in entropy_data:
            entropy_confidence = entropy_data['interchangeable_analysis'].get('form_stability', 0.5)
        else:
            entropy_confidence = 0.5
        
        # Weighted combination
        overall_confidence = (
            de_confidence * 0.4 +
            pattern_confidence * 0.4 +
            entropy_confidence * 0.2
        )
        
        return min(1.0, overall_confidence)
    
    def _generate_visualizations(self, results: Dict) -> None:
        """Generate comprehensive visualizations"""
        
        output_dir = self.config['output_directory']
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary plot
        self._create_summary_plot(results, os.path.join(output_dir, 'transformation_summary.png'))
        
        # Individual signal plots
        for i, signal_result in enumerate(results['individual_signals']):
            self._create_signal_plot(signal_result, os.path.join(output_dir, f'signal_{i}_analysis.png'))
        
        # Cross-analysis plot if multiple signals
        if len(results['individual_signals']) > 1:
            self._create_cross_analysis_plot(results, os.path.join(output_dir, 'cross_analysis.png'))
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def _create_summary_plot(self, results: Dict, save_path: str) -> None:
        """Create summary visualization"""
        
        n_signals = len(results['individual_signals'])
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Universal Transformation Framework - Summary', fontsize=16)
        
        # Plot 1: Time series of all signals
        for signal_result in results['individual_signals']:
            data = signal_result['transformation'].original_data
            axes[0, 0].plot(data, alpha=0.7, label=signal_result['signal_name'])
        
        axes[0, 0].set_title('Original Signals')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Confidence scores
        signal_names = [sr['signal_name'] for sr in results['individual_signals']]
        confidences = [sr['transformation'].confidence_score for sr in results['individual_signals']]
        
        axes[0, 1].bar(signal_names, confidences, alpha=0.7)
        axes[0, 1].set_title('Transformation Confidence')
        axes[0, 1].set_ylabel('Confidence Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Pattern types distribution
        all_pattern_types = []
        for signal_result in results['individual_signals']:
            if 'validated_patterns' in signal_result['transformation'].pattern_discovery:
                patterns = signal_result['transformation'].pattern_discovery['validated_patterns']
                all_pattern_types.extend([p.pattern_type.value for p in patterns])
        
        if all_pattern_types:
            from collections import Counter
            pattern_counts = Counter(all_pattern_types)
            axes[1, 0].bar(pattern_counts.keys(), pattern_counts.values(), alpha=0.7)
            axes[1, 0].set_title('Discovered Pattern Types')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No patterns discovered', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Pattern Discovery')
        
        # Plot 4: System classification summary
        meta = results['meta_analysis']
        classification_text = f"System Type: {meta['system_classification']['type']}\n"
        classification_text += f"Complexity: {meta['system_classification']['complexity']}\n"
        classification_text += f"Overall Confidence: {meta['overall_confidence']:.3f}\n"
        classification_text += f"Dominant DE Type: {meta['dominant_de_type']}\n"
        classification_text += f"Dominant Pattern: {meta['dominant_pattern_type']}"
        
        axes[1, 1].text(0.05, 0.95, classification_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontsize=10, fontfamily='monospace')
        axes[1, 1].set_title('System Classification')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_signal_plot(self, signal_result: Dict, save_path: str) -> None:
        """Create detailed plot for individual signal"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Signal Analysis: {signal_result["signal_name"]}', fontsize=16)
        
        transform = signal_result['transformation']
        data = transform.original_data
        
        # Plot 1: Original signal
        axes[0, 0].plot(data, alpha=0.8)
        axes[0, 0].set_title('Original Signal')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Frequency domain
        if 'frequency_response' in transform.laplace_transform:
            freq_resp = transform.laplace_transform['frequency_response']
            axes[0, 1].loglog(freq_resp['frequencies'], freq_resp['power_spectral_density'])
            axes[0, 1].set_title('Power Spectral Density')
            axes[0, 1].set_xlabel('Frequency (Hz)')
            axes[0, 1].set_ylabel('PSD')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Differential equation visualization
        de_type = transform.differential_equation['type']
        de_text = f"DE Type: {de_type}\n"
        de_text += f"Equation: {transform.differential_equation.get('symbolic', 'N/A')}\n"
        de_text += f"R²: {transform.differential_equation.get('goodness_of_fit', 0):.4f}"
        
        axes[0, 2].text(0.05, 0.95, de_text, transform=axes[0, 2].transAxes,
                       verticalalignment='top', fontsize=10, fontfamily='monospace')
        axes[0, 2].set_title('Differential Equation')
        axes[0, 2].axis('off')
        
        # Plot 4: S-entropy tridimensional matrix
        entropy_data = transform.s_entropy_analysis
        if 'tridimensional_matrix' in entropy_data:
            entropy_matrix = entropy_data['tridimensional_matrix']['entropy_matrix']
            im = axes[1, 0].imshow(entropy_matrix, cmap='viridis', aspect='equal')
            axes[1, 0].set_title('S-Entropy Matrix')
            axes[1, 0].set_xlabel('Entropy Dimension')
            axes[1, 0].set_ylabel('Entropy Dimension')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 5: Pattern confidence
        if 'validated_patterns' in transform.pattern_discovery:
            patterns = transform.pattern_discovery['validated_patterns']
            if patterns:
                pattern_names = [f"{p.pattern_type.value}_{i}" for i, p in enumerate(patterns)]
                confidences = [p.confidence for p in patterns]
                
                axes[1, 1].bar(range(len(patterns)), confidences, alpha=0.7)
                axes[1, 1].set_title('Pattern Confidences')
                axes[1, 1].set_xticks(range(len(patterns)))
                axes[1, 1].set_xticklabels(pattern_names, rotation=45)
                axes[1, 1].set_ylabel('Confidence')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No patterns found', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Plot 6: Overall metrics
        metrics_text = f"Overall Confidence: {transform.confidence_score:.3f}\n"
        metrics_text += f"Total S-Entropy: {entropy_data.get('total_s_entropy', 0):.3f}\n"
        
        if 'validated_patterns' in transform.pattern_discovery:
            patterns = transform.pattern_discovery['validated_patterns']
            metrics_text += f"Patterns Found: {len(patterns)}\n"
            if patterns:
                metrics_text += f"Best Pattern: {max(patterns, key=lambda p: p.confidence).pattern_type.value}\n"
        
        axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=10, fontfamily='monospace')
        axes[1, 2].set_title('Summary Metrics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cross_analysis_plot(self, results: Dict, save_path: str) -> None:
        """Create cross-signal analysis visualization"""
        
        cross_analysis = results['cross_signal_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Cross-Signal Analysis', fontsize=16)
        
        # Plot 1: Cross-correlation matrix
        corr_matrix = cross_analysis['cross_correlations']
        signal_names = cross_analysis['signal_names']
        
        im1 = axes[0, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 0].set_title('Cross-Correlation Matrix')
        axes[0, 0].set_xticks(range(len(signal_names)))
        axes[0, 0].set_yticks(range(len(signal_names)))
        axes[0, 0].set_xticklabels(signal_names, rotation=45)
        axes[0, 0].set_yticklabels(signal_names)
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Phase coupling matrix
        phase_matrix = cross_analysis['phase_couplings']
        
        im2 = axes[0, 1].imshow(phase_matrix, cmap='viridis', vmin=0, vmax=1)
        axes[0, 1].set_title('Phase Coupling Matrix')
        axes[0, 1].set_xticks(range(len(signal_names)))
        axes[0, 1].set_yticks(range(len(signal_names)))
        axes[0, 1].set_xticklabels(signal_names, rotation=45)
        axes[0, 1].set_yticklabels(signal_names)
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot 3: Strongest coupling info
        strongest = cross_analysis['strongest_coupling']
        coupling_text = f"Strongest Coupling:\n"
        coupling_text += f"Signals: {strongest['signal_pair'][0]} ↔ {strongest['signal_pair'][1]}\n"
        coupling_text += f"Correlation: {strongest['correlation']:.4f}\n"
        coupling_text += f"Strength: {strongest['coupling_strength']}"
        
        axes[1, 0].text(0.05, 0.95, coupling_text, transform=axes[1, 0].transAxes,
                       verticalalignment='top', fontsize=12, fontfamily='monospace')
        axes[1, 0].set_title('Coupling Summary')
        axes[1, 0].axis('off')
        
        # Plot 4: Frequency coupling visualization
        freq_couplings = cross_analysis['frequency_couplings']
        
        if freq_couplings:
            coupling_info = []
            for pair, harmonics in freq_couplings.items():
                if harmonics:
                    coupling_info.append(f"{pair}: {len(harmonics)} harmonic relations")
            
            if coupling_info:
                freq_text = "Harmonic Relationships:\n" + "\n".join(coupling_info[:5])
            else:
                freq_text = "No significant harmonic relationships found"
        else:
            freq_text = "No frequency coupling data available"
        
        axes[1, 1].text(0.05, 0.95, freq_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontsize=10, fontfamily='monospace')
        axes[1, 1].set_title('Frequency Coupling')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, results: Dict) -> str:
        """Generate comprehensive text summary report"""
        
        metadata = results['metadata']
        meta_analysis = results['meta_analysis']
        
        report = f"""
=== UNIVERSAL OSCILLATORY TRANSFORMATION FRAMEWORK ===
COMPREHENSIVE ANALYSIS REPORT

EXECUTIVE SUMMARY:
- Data processed: {metadata['n_signals']} signal(s), {metadata['processed_length']:,} data points
- Analysis duration: {results['execution_time']:.2f} seconds
- Overall system confidence: {meta_analysis['overall_confidence']:.3f}
- System classification: {meta_analysis['system_classification']['type']}
- System complexity: {meta_analysis['system_classification']['complexity']}

DATA CHARACTERISTICS:
- Time span: {metadata['time_span']:.2f} time units
- Sampling rate: {metadata['sampling_rate']:.4f} Hz
- Data quality score: {metadata['data_quality']['overall_quality_score']:.3f}
- Temporal regularity: {metadata['data_quality']['temporal_regularity']:.3f}

DIFFERENTIAL EQUATION ANALYSIS:
- Dominant equation type: {meta_analysis['dominant_de_type']}
- Signal-specific equations:
"""
        
        for i, signal_result in enumerate(results['individual_signals']):
            de = signal_result['transformation'].differential_equation
            report += f"  {signal_result['signal_name']}: {de['type']} (R² = {de.get('goodness_of_fit', 0):.3f})\n"
        
        report += f"""
PATTERN DISCOVERY:
- Dominant pattern type: {meta_analysis['dominant_pattern_type']}
- Total patterns discovered: {sum(len(sr['transformation'].pattern_discovery.get('validated_patterns', [])) for sr in results['individual_signals'])}
"""
        
        # Add pattern details
        for signal_result in results['individual_signals']:
            patterns = signal_result['transformation'].pattern_discovery.get('validated_patterns', [])
            if patterns:
                report += f"\n  {signal_result['signal_name']} patterns:\n"
                for pattern in patterns:
                    report += f"    - {pattern.pattern_type.value}: confidence {pattern.confidence:.3f}\n"
        
        report += f"""
S-ENTROPY ANALYSIS:
- Interchangeable forms analysis:
  - Most consistent form: {meta_analysis['interchangeable_forms_meta']['most_consistent_form']}
  - Most dominant form: {meta_analysis['interchangeable_forms_meta']['most_dominant_form']}
  - Interchangeability score: {meta_analysis['interchangeable_forms_meta']['interchangeability_score']:.3f}
"""
        
        if len(results['individual_signals']) > 1:
            cross_analysis = results['cross_signal_analysis']
            strongest = cross_analysis['strongest_coupling']
            
            report += f"""
CROSS-SIGNAL ANALYSIS:
- Strongest coupling: {strongest['signal_pair'][0]} ↔ {strongest['signal_pair'][1]}
- Coupling strength: {strongest['coupling_strength']} (r = {strongest['correlation']:.3f})
- Frequency harmonics detected: {len(cross_analysis['frequency_couplings'])} signal pairs
"""
        
        report += f"""
SYSTEM COMPLEXITY:
- Complexity score: {meta_analysis['complexity_metrics']['complexity_score']:.2f}
- Complexity category: {meta_analysis['complexity_metrics']['complexity_category']}
- Contributing factors:
  - Number of signals: {meta_analysis['complexity_metrics']['n_signals']}
  - Number of patterns: {meta_analysis['complexity_metrics']['n_patterns']}
  - Frequency components: {meta_analysis['complexity_metrics']['total_frequencies']}

RECOMMENDATIONS:
"""
        
        recommendations = meta_analysis['recommendations']
        for i, rec in enumerate(recommendations['next_steps'], 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
CONFIDENCE ASSESSMENT: {recommendations['confidence_assessment'].upper()}

Framework Version: {results['framework_version']}
Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report

# Convenience function for simple usage
def analyze_oscillations(data_source: Union[str, pd.DataFrame, np.ndarray, Dict],
                        time_column: Optional[str] = None,
                        target_columns: Optional[List[str]] = None,
                        config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Simple interface for oscillatory analysis
    
    Usage:
        results = analyze_oscillations('data.csv', time_column='timestamp')
        results = analyze_oscillations(dataframe, time_column='time')
        results = analyze_oscillations({'time': t, 'signal': y})
    """
    
    framework = UniversalTransformationFramework(config)
    return framework.transform(data_source, time_column, target_columns)

def main():
    """Test the universal transformation framework"""
    
    # Generate complex test data
    t = np.linspace(0, 20, 2000)
    
    # Multi-component signals
    signal1 = (2 * np.sin(2 * np.pi * 0.5 * t) +           # Fundamental
              1 * np.sin(2 * np.pi * 1.0 * t) +           # Harmonic  
              0.5 * np.sin(2 * np.pi * 3.7 * t) +         # Incommensurate
              0.2 * np.random.normal(size=len(t)))         # Noise
    
    signal2 = (1.5 * np.cos(2 * np.pi * 0.5 * t + np.pi/4) +  # Phase shifted fundamental
              0.8 * t * np.exp(-0.1 * t) +                    # Trend
              0.3 * np.random.normal(size=len(t)))             # Noise
    
    # Create test data
    test_data = pd.DataFrame({
        'time': t,
        'oscillatory_signal': signal1,
        'trending_signal': signal2,
        'random_signal': np.random.normal(size=len(t))
    })
    
    print("=== UNIVERSAL TRANSFORMATION FRAMEWORK TEST ===")
    print(f"Test data: {len(test_data)} points, 3 signals, 20 time units")
    print()
    
    # Run analysis
    results = analyze_oscillations(test_data, time_column='time')
    
    # Display results
    if 'error' in results:
        print(f"Analysis failed: {results['error']}")
    else:
        print(results['summary_report'])

if __name__ == "__main__":
    main()
