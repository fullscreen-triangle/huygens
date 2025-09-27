"""
Universal Pattern Discovery Engine
Discovers oscillatory patterns without prior system knowledge
Applies sophisticated algorithms to transformed data
"""

import numpy as np
import pandas as pd
from scipy import signal, stats, cluster, spatial
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

from universal_oscillatory_transform import (
    TransformationResult, DifferentialForm, 
    UniversalDataIngestion, DifferentialEquationGenerator,
    LaplaceTransformEngine, SEntropyAnalyzer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of oscillatory patterns"""
    PERIODIC = "periodic"
    QUASI_PERIODIC = "quasi_periodic"
    CHAOTIC = "chaotic"
    FRACTAL = "fractal"
    NOISE = "noise"
    HYBRID = "hybrid"

@dataclass
class OscillatoryPattern:
    """Discovered oscillatory pattern"""
    pattern_type: PatternType
    frequency_components: List[float]
    amplitude_components: List[float]
    phase_relationships: List[float]
    confidence: float
    entropy_signature: Dict[str, float]
    differential_form: DifferentialForm
    mathematical_description: str

class PatternDiscoveryEngine:
    """Universal pattern discovery for any oscillatory system"""
    
    def __init__(self):
        self.discovered_patterns = []
        self.pattern_library = {}
        self.confidence_threshold = 0.7
        
    def discover_patterns(self, transformation_result: TransformationResult) -> Dict[str, Any]:
        """Discover patterns from transformation results"""
        
        logger.info("Starting universal pattern discovery...")
        
        # Extract data
        t = np.linspace(0, len(transformation_result.original_data), len(transformation_result.original_data))
        y = transformation_result.original_data
        
        # Multi-method pattern discovery
        patterns = {
            'frequency_domain': self._discover_frequency_patterns(transformation_result),
            'time_domain': self._discover_time_patterns(t, y),
            'entropy_domain': self._discover_entropy_patterns(transformation_result),
            'differential_domain': self._discover_differential_patterns(transformation_result),
            'hybrid_patterns': self._discover_hybrid_patterns(transformation_result)
        }
        
        # Pattern classification and ranking
        classified_patterns = self._classify_patterns(patterns)
        
        # Cross-validate patterns
        validated_patterns = self._cross_validate_patterns(classified_patterns, t, y)
        
        # Generate pattern signatures
        pattern_signatures = self._generate_pattern_signatures(validated_patterns)
        
        return {
            'discovered_patterns': patterns,
            'classified_patterns': classified_patterns,
            'validated_patterns': validated_patterns,
            'pattern_signatures': pattern_signatures,
            'discovery_confidence': self._compute_discovery_confidence(validated_patterns),
            'recommended_analysis_path': self._recommend_analysis_path(classified_patterns)
        }
    
    def _discover_frequency_patterns(self, result: TransformationResult) -> List[OscillatoryPattern]:
        """Discover patterns in frequency domain"""
        
        patterns = []
        laplace_data = result.laplace_transform
        
        if 'frequency_response' in laplace_data:
            freq_resp = laplace_data['frequency_response']
            frequencies = freq_resp['frequencies']
            psd = freq_resp['power_spectral_density']
            
            # Find spectral peaks
            peaks, properties = signal.find_peaks(psd, height=np.max(psd)*0.1, distance=5)
            
            if len(peaks) > 0:
                peak_freqs = frequencies[peaks]
                peak_powers = psd[peaks]
                
                # Analyze peak relationships
                for i, (freq, power) in enumerate(zip(peak_freqs, peak_powers)):
                    
                    # Check for harmonics
                    harmonics = []
                    for j, other_freq in enumerate(peak_freqs):
                        if i != j:
                            ratio = other_freq / freq if freq > 0 else 0
                            if abs(ratio - round(ratio)) < 0.1 and 1 < ratio < 10:
                                harmonics.append(other_freq)
                    
                    # Determine pattern type
                    if len(harmonics) >= 2:
                        pattern_type = PatternType.PERIODIC
                        confidence = min(0.9, len(harmonics) * 0.2)
                    elif len(harmonics) == 1:
                        pattern_type = PatternType.QUASI_PERIODIC
                        confidence = 0.6
                    else:
                        pattern_type = PatternType.NOISE
                        confidence = 0.3
                    
                    # Create pattern
                    pattern = OscillatoryPattern(
                        pattern_type=pattern_type,
                        frequency_components=[freq] + harmonics,
                        amplitude_components=[power] + [psd[np.argmin(np.abs(frequencies - h))] for h in harmonics],
                        phase_relationships=[0] * (len(harmonics) + 1),  # Phase analysis would require complex FFT
                        confidence=confidence,
                        entropy_signature={'frequency_entropy': -np.sum((psd/np.sum(psd)) * np.log2(psd/np.sum(psd) + 1e-10))},
                        differential_form=DifferentialForm.TIME,
                        mathematical_description=f"Oscillatory at {freq:.4f} Hz with {len(harmonics)} harmonics"
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _discover_time_patterns(self, t: np.ndarray, y: np.ndarray) -> List[OscillatoryPattern]:
        """Discover patterns in time domain"""
        
        patterns = []
        
        # Autocorrelation analysis
        autocorr = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find periodic peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.5, distance=10)
        peaks += 1  # Adjust for offset
        
        if len(peaks) > 0:
            # Extract periods
            dt = np.mean(np.diff(t))
            periods = peaks * dt
            
            for i, period in enumerate(periods):
                frequency = 1 / period if period > 0 else 0
                
                # Estimate amplitude by fitting sinusoid
                try:
                    def sine_model(t, A, phi):
                        return A * np.sin(2 * np.pi * frequency * t + phi)
                    
                    from scipy.optimize import curve_fit
                    popt, _ = curve_fit(sine_model, t, y, p0=[np.std(y), 0])
                    amplitude = abs(popt[0])
                    phase = popt[1]
                    
                    # Calculate goodness of fit
                    y_pred = sine_model(t, *popt)
                    r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                    
                    confidence = min(0.9, r_squared)
                    
                except:
                    amplitude = autocorr[peaks[i]]
                    phase = 0
                    confidence = 0.4
                
                # Determine pattern type based on regularity
                if confidence > 0.8:
                    pattern_type = PatternType.PERIODIC
                elif confidence > 0.5:
                    pattern_type = PatternType.QUASI_PERIODIC
                else:
                    pattern_type = PatternType.NOISE
                
                pattern = OscillatoryPattern(
                    pattern_type=pattern_type,
                    frequency_components=[frequency],
                    amplitude_components=[amplitude],
                    phase_relationships=[phase],
                    confidence=confidence,
                    entropy_signature={'autocorr_entropy': -np.sum((autocorr[:50]/np.sum(autocorr[:50])) * 
                                                                  np.log2(autocorr[:50]/np.sum(autocorr[:50]) + 1e-10))},
                    differential_form=DifferentialForm.TIME,
                    mathematical_description=f"Periodic signal with period {period:.4f} seconds"
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _discover_entropy_patterns(self, result: TransformationResult) -> List[OscillatoryPattern]:
        """Discover patterns in entropy domain"""
        
        patterns = []
        entropy_data = result.s_entropy_analysis
        
        if 'tridimensional_matrix' in entropy_data:
            entropy_matrix = entropy_data['tridimensional_matrix']['entropy_matrix']
            eigenvals = entropy_data['tridimensional_matrix']['eigenvalues']
            
            # Analyze entropy eigenstructure
            dominant_eigenval = np.max(np.real(eigenvals))
            eigenval_ratio = np.max(np.real(eigenvals)) / (np.min(np.real(eigenvals)) + 1e-10)
            
            # Determine pattern type from entropy structure
            if eigenval_ratio > 10:
                pattern_type = PatternType.FRACTAL
                confidence = min(0.8, eigenval_ratio / 20)
            elif eigenval_ratio > 3:
                pattern_type = PatternType.CHAOTIC
                confidence = min(0.7, eigenval_ratio / 10)
            else:
                pattern_type = PatternType.PERIODIC
                confidence = 0.5
            
            # Extract entropy frequencies (conceptual)
            entropy_freqs = np.real(eigenvals) / (2 * np.pi)  # Convert eigenvalues to frequencies
            
            pattern = OscillatoryPattern(
                pattern_type=pattern_type,
                frequency_components=entropy_freqs.tolist(),
                amplitude_components=np.abs(eigenvals).tolist(),
                phase_relationships=np.angle(eigenvals).tolist(),
                confidence=confidence,
                entropy_signature={
                    'total_entropy': entropy_data['total_s_entropy'],
                    'matrix_determinant': entropy_data['tridimensional_matrix']['determinant'],
                    'eigenvalue_ratio': eigenval_ratio
                },
                differential_form=DifferentialForm.ENTROPY,
                mathematical_description=f"Entropy eigenstructure with ratio {eigenval_ratio:.2f}"
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _discover_differential_patterns(self, result: TransformationResult) -> List[OscillatoryPattern]:
        """Discover patterns in differential equation domain"""
        
        patterns = []
        de_data = result.differential_equation
        
        eq_type = de_data.get('type', 'unknown')
        coefficients = de_data.get('coefficients', {})
        
        if eq_type == 'oscillatory':
            omega = coefficients.get('omega', 1)
            damping = coefficients.get('damping', 0.1)
            
            # Natural frequency and damping analysis
            natural_freq = omega / (2 * np.pi)
            damped_freq = omega * np.sqrt(1 - damping**2) / (2 * np.pi) if damping < 1 else 0
            
            # Determine oscillatory behavior
            if damping < 0.1:
                pattern_type = PatternType.PERIODIC
                confidence = 0.9
            elif damping < 1:
                pattern_type = PatternType.QUASI_PERIODIC
                confidence = 0.7
            else:
                pattern_type = PatternType.NOISE
                confidence = 0.4
            
            pattern = OscillatoryPattern(
                pattern_type=pattern_type,
                frequency_components=[natural_freq, damped_freq] if damped_freq > 0 else [natural_freq],
                amplitude_components=[1, np.exp(-damping)] if damped_freq > 0 else [1],
                phase_relationships=[0, np.pi/2] if damped_freq > 0 else [0],
                confidence=confidence,
                entropy_signature={
                    'damping_ratio': damping,
                    'quality_factor': 1/(2*damping) if damping > 0 else float('inf')
                },
                differential_form=DifferentialForm.TIME,
                mathematical_description=f"Second-order oscillator: ω={omega:.4f}, ζ={damping:.4f}"
            )
            
            patterns.append(pattern)
            
        elif eq_type == 'nonlinear':
            # Nonlinear system analysis
            a, b, c = coefficients.get('a', 0), coefficients.get('b', 0), coefficients.get('c', 0)
            
            # Check for potential chaos (simplified analysis)
            if abs(a) > 0.1:  # Nonlinear term present
                pattern_type = PatternType.CHAOTIC
                confidence = 0.6
            else:
                pattern_type = PatternType.QUASI_PERIODIC
                confidence = 0.5
            
            pattern = OscillatoryPattern(
                pattern_type=pattern_type,
                frequency_components=[abs(b) / (2 * np.pi)],  # Linear term frequency
                amplitude_components=[abs(c)],  # Constant term
                phase_relationships=[0],
                confidence=confidence,
                entropy_signature={'nonlinearity': abs(a)},
                differential_form=DifferentialForm.TIME,
                mathematical_description=f"Nonlinear system: dy/dt = {a:.4f}y² + {b:.4f}y + {c:.4f}"
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _discover_hybrid_patterns(self, result: TransformationResult) -> List[OscillatoryPattern]:
        """Discover hybrid patterns combining multiple domains"""
        
        patterns = []
        
        # Check interchangeability analysis
        if 'interchangeable_analysis' in result.s_entropy_analysis:
            interch = result.s_entropy_analysis['interchangeable_analysis']
            interch_score = interch['interchangeability_score']
            dominant_form = interch['dominant_form']
            
            if interch_score > 0.7:
                # High interchangeability suggests hybrid behavior
                pattern_type = PatternType.HYBRID
                confidence = interch_score
                
                # Extract features from all differential forms
                freq_components = []
                amp_components = []
                
                # Get frequency from differential equation
                if result.differential_equation.get('type') == 'oscillatory':
                    omega = result.differential_equation['coefficients'].get('omega', 1)
                    freq_components.append(omega / (2 * np.pi))
                    amp_components.append(1.0)
                
                # Get entropy contribution
                total_entropy = result.s_entropy_analysis.get('total_s_entropy', 0)
                freq_components.append(total_entropy)  # Entropy as a frequency-like measure
                amp_components.append(interch_score)
                
                pattern = OscillatoryPattern(
                    pattern_type=pattern_type,
                    frequency_components=freq_components,
                    amplitude_components=amp_components,
                    phase_relationships=[0] * len(freq_components),
                    confidence=confidence,
                    entropy_signature={
                        'interchangeability': interch_score,
                        'dominant_form': dominant_form,
                        'form_stability': interch['form_stability']
                    },
                    differential_form=DifferentialForm(dominant_form),
                    mathematical_description=f"Hybrid pattern with {dominant_form} dominance"
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _classify_patterns(self, patterns: Dict[str, List[OscillatoryPattern]]) -> List[OscillatoryPattern]:
        """Classify and merge similar patterns"""
        
        all_patterns = []
        for domain_patterns in patterns.values():
            all_patterns.extend(domain_patterns)
        
        if not all_patterns:
            return []
        
        # Sort by confidence
        all_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        # Merge similar patterns
        merged_patterns = []
        for pattern in all_patterns:
            
            # Check if similar pattern already exists
            is_duplicate = False
            for existing in merged_patterns:
                if self._patterns_similar(pattern, existing):
                    # Merge by taking higher confidence pattern but combining features
                    if pattern.confidence > existing.confidence:
                        existing.confidence = (existing.confidence + pattern.confidence) / 2
                        existing.frequency_components.extend(pattern.frequency_components)
                        existing.amplitude_components.extend(pattern.amplitude_components)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_patterns.append(pattern)
        
        return merged_patterns
    
    def _patterns_similar(self, p1: OscillatoryPattern, p2: OscillatoryPattern, threshold: float = 0.2) -> bool:
        """Check if two patterns are similar"""
        
        # Type similarity
        if p1.pattern_type != p2.pattern_type:
            return False
        
        # Frequency similarity
        if p1.frequency_components and p2.frequency_components:
            freq1 = np.mean(p1.frequency_components)
            freq2 = np.mean(p2.frequency_components)
            if abs(freq1 - freq2) / (max(freq1, freq2) + 1e-10) > threshold:
                return False
        
        return True
    
    def _cross_validate_patterns(self, patterns: List[OscillatoryPattern], 
                                t: np.ndarray, y: np.ndarray) -> List[OscillatoryPattern]:
        """Cross-validate discovered patterns"""
        
        validated_patterns = []
        
        for pattern in patterns:
            
            # Validate by reconstruction
            validation_score = self._validate_pattern_reconstruction(pattern, t, y)
            
            if validation_score > 0.3:  # Minimum validation threshold
                # Adjust confidence based on validation
                pattern.confidence = pattern.confidence * validation_score
                validated_patterns.append(pattern)
        
        return validated_patterns
    
    def _validate_pattern_reconstruction(self, pattern: OscillatoryPattern, 
                                       t: np.ndarray, y: np.ndarray) -> float:
        """Validate pattern by attempting to reconstruct the signal"""
        
        try:
            # Reconstruct signal based on pattern
            if pattern.pattern_type in [PatternType.PERIODIC, PatternType.QUASI_PERIODIC]:
                reconstructed = np.zeros_like(y)
                
                for i, (freq, amp, phase) in enumerate(zip(
                    pattern.frequency_components[:len(pattern.amplitude_components)],
                    pattern.amplitude_components,
                    pattern.phase_relationships[:len(pattern.amplitude_components)]
                )):
                    if freq > 0:
                        reconstructed += amp * np.sin(2 * np.pi * freq * t + phase)
                
                # Calculate reconstruction quality
                correlation = np.corrcoef(y, reconstructed)[0, 1]
                rmse = np.sqrt(np.mean((y - reconstructed)**2))
                normalized_rmse = rmse / (np.std(y) + 1e-10)
                
                validation_score = abs(correlation) * (1 - min(1, normalized_rmse))
                return max(0, validation_score)
            
            else:
                # For other pattern types, use entropy-based validation
                return min(1.0, pattern.confidence)
                
        except:
            return 0.3  # Default moderate validation
    
    def _generate_pattern_signatures(self, patterns: List[OscillatoryPattern]) -> Dict[str, Any]:
        """Generate unique signatures for discovered patterns"""
        
        signatures = {}
        
        for i, pattern in enumerate(patterns):
            signature = {
                'pattern_id': f"pattern_{i}",
                'type': pattern.pattern_type.value,
                'dominant_frequency': np.mean(pattern.frequency_components) if pattern.frequency_components else 0,
                'frequency_spread': np.std(pattern.frequency_components) if len(pattern.frequency_components) > 1 else 0,
                'amplitude_ratio': max(pattern.amplitude_components) / (min(pattern.amplitude_components) + 1e-10) if len(pattern.amplitude_components) > 1 else 1,
                'confidence': pattern.confidence,
                'differential_form': pattern.differential_form.value,
                'entropy_fingerprint': hash(str(sorted(pattern.entropy_signature.items()))) % 10000,
                'complexity_score': len(pattern.frequency_components) * pattern.confidence
            }
            
            signatures[f"pattern_{i}"] = signature
        
        return signatures
    
    def _compute_discovery_confidence(self, patterns: List[OscillatoryPattern]) -> float:
        """Compute overall discovery confidence"""
        
        if not patterns:
            return 0.0
        
        # Weight by pattern confidence and count
        confidence_sum = sum(p.confidence for p in patterns)
        pattern_count_factor = min(1.0, len(patterns) / 3)  # Normalize for 3 patterns
        
        overall_confidence = (confidence_sum / len(patterns)) * pattern_count_factor
        return min(1.0, overall_confidence)
    
    def _recommend_analysis_path(self, patterns: List[OscillatoryPattern]) -> Dict[str, Any]:
        """Recommend next analysis steps based on discovered patterns"""
        
        if not patterns:
            return {'recommendation': 'increase_data_quality', 'reason': 'no_patterns_found'}
        
        # Analyze pattern types
        pattern_types = [p.pattern_type for p in patterns]
        dominant_type = max(set(pattern_types), key=pattern_types.count)
        
        recommendations = {
            PatternType.PERIODIC: {
                'recommendation': 'harmonic_analysis',
                'reason': 'clear_periodic_structure',
                'suggested_methods': ['fourier_analysis', 'harmonic_fitting', 'spectral_analysis']
            },
            PatternType.QUASI_PERIODIC: {
                'recommendation': 'nonlinear_analysis',
                'reason': 'quasi_periodic_detected',
                'suggested_methods': ['phase_space_reconstruction', 'lyapunov_exponents', 'correlation_dimension']
            },
            PatternType.CHAOTIC: {
                'recommendation': 'chaos_analysis',
                'reason': 'chaotic_behavior_indicated',
                'suggested_methods': ['attractor_reconstruction', 'entropy_analysis', 'bifurcation_analysis']
            },
            PatternType.FRACTAL: {
                'recommendation': 'fractal_analysis',
                'reason': 'fractal_structure_detected',
                'suggested_methods': ['box_counting', 'hurst_exponent', 'multifractal_analysis']
            },
            PatternType.HYBRID: {
                'recommendation': 'multidomain_analysis',
                'reason': 'hybrid_patterns_require_combined_approach',
                'suggested_methods': ['wavelet_analysis', 'hilbert_huang_transform', 'empirical_mode_decomposition']
            },
            PatternType.NOISE: {
                'recommendation': 'noise_characterization',
                'reason': 'dominant_noise_characteristics',
                'suggested_methods': ['noise_modeling', 'stochastic_analysis', 'filtering_techniques']
            }
        }
        
        base_rec = recommendations.get(dominant_type, {
            'recommendation': 'general_analysis',
            'reason': 'mixed_pattern_types',
            'suggested_methods': ['comprehensive_spectral_analysis']
        })
        
        # Add confidence-based recommendations
        avg_confidence = np.mean([p.confidence for p in patterns])
        
        if avg_confidence < 0.5:
            base_rec['additional_recommendation'] = 'improve_data_quality'
            base_rec['data_suggestions'] = ['increase_sampling_rate', 'reduce_noise', 'extend_measurement_duration']
        elif avg_confidence > 0.8:
            base_rec['additional_recommendation'] = 'detailed_characterization'
            base_rec['characterization_suggestions'] = ['parameter_estimation', 'model_validation', 'predictive_analysis']
        
        return base_rec

def main():
    """Test pattern discovery engine"""
    
    # Generate complex test signal
    t = np.linspace(0, 20, 2000)
    
    # Multi-component signal
    signal1 = 2 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz
    signal2 = 1 * np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz harmonic
    signal3 = 0.5 * np.sin(2 * np.pi * 3.7 * t)  # Incommensurate frequency
    noise = 0.2 * np.random.normal(size=len(t))
    
    y = signal1 + signal2 + signal3 + noise
    
    # Create transformation result (simplified)
    from universal_oscillatory_transform import TransformationResult
    
    # Run transformations
    ingestion = UniversalDataIngestion()
    time_vec, data_matrix = ingestion.ingest_data({'time': t, 'signal': y}, 'time')
    
    de_gen = DifferentialEquationGenerator()
    de_result = de_gen.generate_first_order_de(time_vec, data_matrix[:, 0])
    
    laplace_engine = LaplaceTransformEngine()
    laplace_result = laplace_engine.transform_to_laplace(de_result, time_vec, data_matrix[:, 0])
    
    s_entropy = SEntropyAnalyzer()
    entropy_result = s_entropy.analyze_s_entropy(time_vec, data_matrix[:, 0])
    
    # Create transformation result
    transform_result = TransformationResult(
        original_data=data_matrix[:, 0],
        differential_equation=de_result,
        laplace_transform=laplace_result,
        s_entropy_analysis=entropy_result,
        pattern_discovery={},
        oscillatory_features={},
        confidence_score=0.0
    )
    
    # Discover patterns
    engine = PatternDiscoveryEngine()
    discovery_result = engine.discover_patterns(transform_result)
    
    print(f"\nPattern Discovery Results:")
    print(f"Patterns found: {len(discovery_result['validated_patterns'])}")
    print(f"Discovery confidence: {discovery_result['discovery_confidence']:.4f}")
    print(f"Recommended analysis: {discovery_result['recommended_analysis_path']['recommendation']}")
    
    for i, pattern in enumerate(discovery_result['validated_patterns']):
        print(f"\nPattern {i+1}:")
        print(f"  Type: {pattern.pattern_type.value}")
        print(f"  Frequencies: {[f'{f:.4f}' for f in pattern.frequency_components[:3]]}")
        print(f"  Confidence: {pattern.confidence:.4f}")
        print(f"  Description: {pattern.mathematical_description}")

if __name__ == "__main__":
    main()
