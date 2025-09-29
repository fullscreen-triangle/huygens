"""
Biomechanical Oscillatory System Analysis
Surface Compliance Effects on Gait Oscillatory Parameters
Analyzes coupled oscillations in gait, knee mechanics, and center of mass dynamics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, integrate
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import json

# Import universal framework
from universal_transformation_framework import analyze_oscillations
from universal_oscillatory_transform import DifferentialForm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurfaceType(Enum):
    """Types of surface compliance"""
    RIGID = "rigid"           # Concrete, asphalt
    COMPLIANT = "compliant"   # Grass, track
    SOFT = "soft"            # Sand, foam
    VARIABLE = "variable"     # Mixed surfaces

@dataclass
class GaitOscillation:
    """Gait oscillatory pattern"""
    cadence_hz: float
    step_length_m: float
    stance_time_s: float
    swing_time_s: float
    vertical_oscillation_mm: float
    duty_factor: float
    stride_factor: float
    froude_number: float

@dataclass
class SurfaceCompliance:
    """Surface compliance parameters"""
    stiffness: float          # N/m
    damping: float           # Ns/m
    surface_type: SurfaceType
    compliance_factor: float  # 0-1 scale

class BiomechanicalOscillatoryAnalyzer:
    """
    Analyzes biomechanical oscillatory systems with surface compliance effects
    """
    
    def __init__(self):
        # Biomechanical constants
        self.body_mass = 70  # kg (typical)
        self.leg_length = 0.9  # m (typical)
        self.g = 9.81  # gravity
        
        # Surface compliance models
        self.surface_models = {
            SurfaceType.RIGID: SurfaceCompliance(50000, 100, SurfaceType.RIGID, 0.1),
            SurfaceType.COMPLIANT: SurfaceCompliance(10000, 200, SurfaceType.COMPLIANT, 0.5),
            SurfaceType.SOFT: SurfaceCompliance(2000, 500, SurfaceType.SOFT, 0.9),
            SurfaceType.VARIABLE: SurfaceCompliance(15000, 300, SurfaceType.VARIABLE, 0.6)
        }
        
        # Oscillatory frequency ranges (Hz)
        self.gait_frequencies = {
            'cadence': (1.4, 2.0),      # steps/second
            'stride': (0.7, 1.0),       # strides/second  
            'vertical_osc': (2.8, 4.0), # 2x stride frequency
            'com_oscillation': (1.4, 2.0), # center of mass
            'knee_angle': (0.7, 1.0),   # knee flexion-extension
            'ankle_spring': (2.0, 8.0)  # ankle spring oscillations
        }
    
    def analyze_gait_oscillations(self, gait_data: List[Dict]) -> Dict[str, Any]:
        """Analyze gait data for oscillatory patterns and surface compliance effects"""
        
        logger.info("Analyzing gait oscillatory patterns...")
        
        # Convert to DataFrame
        df = pd.DataFrame(gait_data)
        
        # Extract oscillatory parameters
        oscillatory_params = self._extract_oscillatory_parameters(df)
        
        # Surface compliance analysis
        surface_analysis = self._analyze_surface_compliance_effects(df, oscillatory_params)
        
        # Coupled oscillator modeling
        coupled_model = self._model_coupled_gait_oscillators(df, surface_analysis)
        
        # Phase coupling analysis
        phase_coupling = self._analyze_gait_phase_coupling(df)
        
        # Frequency domain analysis using universal framework
        frequency_analysis = self._perform_frequency_domain_analysis(df)
        
        return {
            'oscillatory_parameters': oscillatory_params,
            'surface_compliance_analysis': surface_analysis,
            'coupled_oscillator_model': coupled_model,
            'phase_coupling_analysis': phase_coupling,
            'frequency_domain_analysis': frequency_analysis,
            'gait_efficiency_metrics': self._compute_gait_efficiency(df, surface_analysis),
            'surface_optimization': self._optimize_surface_compliance(df, oscillatory_params)
        }
    
    def _extract_oscillatory_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract key oscillatory parameters from gait data"""
        
        params = {}
        
        # Cadence oscillations (steps/minute to Hz)
        if 'cadence' in df.columns:
            cadence_spm = df['cadence'].values
            cadence_hz = cadence_spm / 60  # Convert to Hz
            params['cadence'] = {
                'mean_hz': np.mean(cadence_hz),
                'std_hz': np.std(cadence_hz),
                'oscillation_amplitude': np.max(cadence_hz) - np.min(cadence_hz),
                'variability': np.std(cadence_hz) / np.mean(cadence_hz)
            }
        
        # Step length oscillations
        if 'step_length' in df.columns:
            step_length = df['step_length'].values / 1000  # Convert mm to m
            params['step_length'] = {
                'mean_m': np.mean(step_length),
                'std_m': np.std(step_length),
                'oscillation_amplitude': np.max(step_length) - np.min(step_length),
                'variability': np.std(step_length) / np.mean(step_length)
            }
        
        # Stance time oscillations
        if 'stance_time' in df.columns:
            stance_time = df['stance_time'].values / 1000  # Convert ms to s
            params['stance_time'] = {
                'mean_s': np.mean(stance_time),
                'std_s': np.std(stance_time),
                'oscillation_amplitude': np.max(stance_time) - np.min(stance_time),
                'duty_factor': np.mean(stance_time) / (np.mean(stance_time) + np.mean(df.get('swing_time', stance_time)))
            }
        
        # Vertical oscillation
        if 'vertical_oscillation' in df.columns:
            vert_osc = df['vertical_oscillation'].values / 1000  # Convert mm to m
            params['vertical_oscillation'] = {
                'mean_m': np.mean(vert_osc),
                'std_m': np.std(vert_osc),
                'oscillation_amplitude': np.max(vert_osc) - np.min(vert_osc),
                'energy_cost': np.mean(vert_osc) * self.body_mass * self.g  # Potential energy cost
            }
        
        # Center of mass acceleration oscillations
        if 'com_acceleration' in df.columns:
            com_acc = df['com_acceleration'].values
            params['com_acceleration'] = {
                'mean_ms2': np.mean(com_acc),
                'std_ms2': np.std(com_acc),
                'oscillation_amplitude': np.max(com_acc) - np.min(com_acc),
                'rms_acceleration': np.sqrt(np.mean(com_acc**2))
            }
        
        return params
    
    def _analyze_surface_compliance_effects(self, df: pd.DataFrame, 
                                          oscillatory_params: Dict) -> Dict[str, Any]:
        """Analyze how surface compliance affects oscillatory parameters"""
        
        # Estimate surface compliance from gait characteristics
        estimated_compliance = self._estimate_surface_compliance(df)
        
        # Model compliance effects on oscillatory parameters
        compliance_effects = {}
        
        # Effect on cadence
        if 'cadence' in oscillatory_params:
            cadence_data = oscillatory_params['cadence']
            # More compliant surfaces typically reduce cadence slightly
            compliance_factor = estimated_compliance.compliance_factor
            
            compliance_effects['cadence'] = {
                'baseline_hz': cadence_data['mean_hz'],
                'compliance_effect': -0.1 * compliance_factor,  # Negative correlation
                'adjusted_hz': cadence_data['mean_hz'] * (1 - 0.05 * compliance_factor),
                'energy_cost_factor': 1 + 0.2 * compliance_factor  # More energy on soft surfaces
            }
        
        # Effect on step length  
        if 'step_length' in oscillatory_params:
            step_data = oscillatory_params['step_length']
            compliance_factor = estimated_compliance.compliance_factor
            
            compliance_effects['step_length'] = {
                'baseline_m': step_data['mean_m'],
                'compliance_effect': -0.05 * compliance_factor,  # Shorter steps on soft surfaces
                'adjusted_m': step_data['mean_m'] * (1 - 0.03 * compliance_factor),
                'ground_reaction_force': self._estimate_grf(step_data['mean_m'], estimated_compliance)
            }
        
        # Effect on stance time
        if 'stance_time' in oscillatory_params:
            stance_data = oscillatory_params['stance_time']
            compliance_factor = estimated_compliance.compliance_factor
            
            compliance_effects['stance_time'] = {
                'baseline_s': stance_data['mean_s'],
                'compliance_effect': 0.1 * compliance_factor,  # Longer stance on soft surfaces
                'adjusted_s': stance_data['mean_s'] * (1 + 0.05 * compliance_factor),
                'contact_mechanics': self._model_contact_mechanics(stance_data['mean_s'], estimated_compliance)
            }
        
        # Effect on vertical oscillation
        if 'vertical_oscillation' in oscillatory_params:
            vert_data = oscillatory_params['vertical_oscillation']
            compliance_factor = estimated_compliance.compliance_factor
            
            compliance_effects['vertical_oscillation'] = {
                'baseline_m': vert_data['mean_m'],
                'compliance_effect': -0.15 * compliance_factor,  # Less vertical motion on soft surfaces
                'adjusted_m': vert_data['mean_m'] * (1 - 0.1 * compliance_factor),
                'spring_damper_model': self._model_leg_spring_damper(vert_data['mean_m'], estimated_compliance)
            }
        
        return {
            'estimated_surface_compliance': estimated_compliance,
            'parameter_effects': compliance_effects,
            'surface_feedback_model': self._create_surface_feedback_model(estimated_compliance),
            'optimization_potential': self._assess_optimization_potential(compliance_effects)
        }
    
    def _estimate_surface_compliance(self, df: pd.DataFrame) -> SurfaceCompliance:
        """Estimate surface compliance from gait characteristics"""
        
        # Use stance time and vertical oscillation as indicators
        mean_stance_time = df['stance_time'].mean() / 1000 if 'stance_time' in df.columns else 0.2
        mean_vert_osc = df['vertical_oscillation'].mean() / 1000 if 'vertical_oscillation' in df.columns else 0.08
        
        # Longer stance time and lower vertical oscillation suggest softer surface
        compliance_indicator = mean_stance_time * (1 / mean_vert_osc) if mean_vert_osc > 0 else 1.0
        
        # Map to surface type
        if compliance_indicator < 2.0:
            surface_type = SurfaceType.RIGID
            compliance_factor = 0.2
        elif compliance_indicator < 4.0:
            surface_type = SurfaceType.COMPLIANT
            compliance_factor = 0.5
        elif compliance_indicator < 6.0:
            surface_type = SurfaceType.SOFT  
            compliance_factor = 0.8
        else:
            surface_type = SurfaceType.VARIABLE
            compliance_factor = 0.6
        
        base_model = self.surface_models[surface_type]
        
        return SurfaceCompliance(
            stiffness=base_model.stiffness * (1 - compliance_factor * 0.5),
            damping=base_model.damping * (1 + compliance_factor * 0.5),
            surface_type=surface_type,
            compliance_factor=compliance_factor
        )
    
    def _estimate_grf(self, step_length: float, surface: SurfaceCompliance) -> Dict:
        """Estimate ground reaction force based on step length and surface"""
        
        # Simplified GRF model
        peak_vertical_grf = self.body_mass * self.g * (2.0 + step_length)  # Peak vertical force
        contact_time = 0.2 * (1 + surface.compliance_factor * 0.1)  # Contact time increases with compliance
        
        return {
            'peak_vertical_n': peak_vertical_grf,
            'contact_time_s': contact_time,
            'impulse_ns': peak_vertical_grf * contact_time * 0.5,  # Simplified impulse
            'loading_rate_n_s': peak_vertical_grf / (contact_time * 0.3)  # Loading rate
        }
    
    def _model_contact_mechanics(self, stance_time: float, surface: SurfaceCompliance) -> Dict:
        """Model foot-surface contact mechanics"""
        
        # Spring-damper model for contact
        k_surface = surface.stiffness
        c_surface = surface.damping
        
        # Natural frequency of foot-surface system
        m_foot = 1.5  # kg (approximate foot mass)
        omega_n = np.sqrt(k_surface / m_foot)
        zeta = c_surface / (2 * np.sqrt(k_surface * m_foot))
        
        return {
            'natural_frequency_hz': omega_n / (2 * np.pi),
            'damping_ratio': zeta,
            'contact_stiffness_n_m': k_surface,
            'contact_damping_ns_m': c_surface,
            'system_type': 'overdamped' if zeta > 1 else 'underdamped' if zeta < 1 else 'critically_damped'
        }
    
    def _model_leg_spring_damper(self, vertical_osc: float, surface: SurfaceCompliance) -> Dict:
        """Model leg as spring-damper system interacting with surface"""
        
        # Leg spring stiffness (typical values)
        k_leg = 25000  # N/m (typical leg stiffness)
        
        # Combined system: leg spring in series with surface
        k_combined = 1 / (1/k_leg + 1/surface.stiffness)
        
        # System dynamics
        omega_n = np.sqrt(k_combined / self.body_mass)
        natural_freq_hz = omega_n / (2 * np.pi)
        
        return {
            'leg_stiffness_n_m': k_leg,
            'combined_stiffness_n_m': k_combined,
            'natural_frequency_hz': natural_freq_hz,
            'predicted_vertical_osc_m': vertical_osc * (k_leg / k_combined),
            'spring_compression_m': self.body_mass * self.g / k_combined
        }
    
    def _create_surface_feedback_model(self, surface: SurfaceCompliance) -> Dict:
        """Create feedback model for surface compliance effects"""
        
        # Feedback equations:
        # step_length = f(surface_stiffness, cadence)  
        # cadence = f(surface_damping, energy_cost)
        # stance_time = f(surface_compliance, body_mass)
        
        def step_length_feedback(cadence, surface_stiffness):
            # Softer surfaces reduce optimal step length
            base_step_length = 0.8  # m
            stiffness_factor = surface_stiffness / 25000  # Normalized
            return base_step_length * stiffness_factor * (cadence / 1.67)  # Normalized cadence
        
        def cadence_feedback(energy_cost, surface_damping):
            # Higher damping reduces preferred cadence
            base_cadence = 1.67  # Hz
            damping_factor = surface_damping / 200  # Normalized
            return base_cadence / (1 + 0.1 * damping_factor * energy_cost)
        
        def stance_time_feedback(surface_compliance, body_mass):
            # More compliant surfaces increase stance time
            base_stance = 0.2  # s
            mass_factor = body_mass / 70  # Normalized
            return base_stance * (1 + 0.2 * surface_compliance) * mass_factor
        
        return {
            'step_length_feedback': step_length_feedback,
            'cadence_feedback': cadence_feedback, 
            'stance_time_feedback': stance_time_feedback,
            'feedback_coupling_strength': 0.3,  # Moderate coupling
            'time_constant_s': 2.0  # Adaptation time constant
        }
    
    def _model_coupled_gait_oscillators(self, df: pd.DataFrame, 
                                       surface_analysis: Dict) -> Dict[str, Any]:
        """Model gait as system of coupled oscillators"""
        
        # Define coupled oscillator system
        # x1 = cadence, x2 = step_length, x3 = stance_time, x4 = vertical_osc
        
        def coupled_gait_dynamics(t, y, surface_params):
            """Coupled gait oscillator equations"""
            cadence, step_length, stance_time, vert_osc = y
            
            # Natural frequencies (rad/s)
            omega_cadence = 2 * np.pi * 1.67  # ~1.67 Hz
            omega_step = 2 * np.pi * 1.0     # ~1.0 Hz variation
            omega_stance = 2 * np.pi * 2.0   # ~2.0 Hz (twice cadence)
            omega_vert = 2 * np.pi * 3.34    # ~3.34 Hz (twice cadence)
            
            # Coupling strengths
            k_coupling = 0.1  # Moderate coupling
            surface_effect = surface_params['compliance_factor']
            
            # Coupled equations
            dcadence_dt = (-omega_cadence**2 * cadence - 
                          k_coupling * (cadence - step_length * 1.67) -
                          surface_effect * 0.1 * cadence)
            
            dstep_dt = (-omega_step**2 * step_length -
                       k_coupling * (step_length - cadence / 1.67) -
                       k_coupling * (step_length - stance_time * 4))
            
            dstance_dt = (-omega_stance**2 * stance_time -
                         k_coupling * (stance_time - step_length / 4) +
                         surface_effect * 0.05)
            
            dvert_dt = (-omega_vert**2 * vert_osc -
                       k_coupling * (vert_osc - cadence * 0.05) -
                       surface_effect * 0.1 * vert_osc)
            
            return [dcadence_dt, dstep_dt, dstance_dt, dvert_dt]
        
        # Initial conditions from data
        if len(df) > 0:
            y0 = [
                df['cadence'].iloc[0] / 60 if 'cadence' in df.columns else 1.67,
                df['step_length'].iloc[0] / 1000 if 'step_length' in df.columns else 0.8,
                df['stance_time'].iloc[0] / 1000 if 'stance_time' in df.columns else 0.2,
                df['vertical_oscillation'].iloc[0] / 1000 if 'vertical_oscillation' in df.columns else 0.08
            ]
        else:
            y0 = [1.67, 0.8, 0.2, 0.08]
        
        # Solve system
        t_span = (0, 10)  # 10 second simulation
        t_eval = np.linspace(0, 10, 1000)
        
        try:
            solution = integrate.solve_ivp(
                coupled_gait_dynamics, 
                t_span, 
                y0, 
                t_eval=t_eval,
                args=(surface_analysis['estimated_surface_compliance'].__dict__,),
                method='RK45'
            )
            
            return {
                'solution': solution,
                'stable': np.all(np.abs(solution.y[:, -100:]) < 10),  # Check if bounded
                'coupling_matrix': self._compute_coupling_matrix(),
                'eigenvalues': self._compute_system_eigenvalues(),
                'natural_frequencies': [1.67, 1.0, 2.0, 3.34],  # Hz
                'damping_ratios': [0.1, 0.15, 0.1, 0.2]
            }
            
        except Exception as e:
            logger.warning(f"Coupled oscillator simulation failed: {e}")
            return {
                'solution': None,
                'stable': False,
                'error': str(e)
            }
    
    def _compute_coupling_matrix(self) -> np.ndarray:
        """Compute coupling matrix between gait oscillators"""
        # 4x4 matrix for cadence, step_length, stance_time, vertical_osc
        coupling = np.array([
            [1.0,  0.3,  0.2,  0.4],  # Cadence coupling
            [0.3,  1.0,  0.5,  0.2],  # Step length coupling  
            [0.2,  0.5,  1.0,  0.1],  # Stance time coupling
            [0.4,  0.2,  0.1,  1.0]   # Vertical oscillation coupling
        ])
        return coupling
    
    def _compute_system_eigenvalues(self) -> List[complex]:
        """Compute eigenvalues of coupled gait system"""
        # Simplified linearized system matrix
        A = np.array([
            [-0.1,  0.1,  0.0,  0.0],
            [ 0.1, -0.2,  0.1,  0.0], 
            [ 0.0,  0.1, -0.3,  0.0],
            [ 0.1,  0.0,  0.0, -0.4]
        ])
        eigenvals = np.linalg.eigvals(A)
        return eigenvals.tolist()
    
    def _analyze_gait_phase_coupling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze phase coupling between gait parameters"""
        
        if len(df) < 10:
            return {'error': 'Insufficient data for phase analysis'}
        
        phase_analysis = {}
        
        # Extract signals for phase analysis
        signals = {}
        
        if 'cadence' in df.columns:
            signals['cadence'] = df['cadence'].values
        if 'step_length' in df.columns:
            signals['step_length'] = df['step_length'].values
        if 'stance_time' in df.columns:
            signals['stance_time'] = df['stance_time'].values
        if 'vertical_oscillation' in df.columns:
            signals['vertical_oscillation'] = df['vertical_oscillation'].values
        
        # Compute phase relationships using Hilbert transform
        from scipy.signal import hilbert
        
        phase_relationships = {}
        
        for name1, signal1 in signals.items():
            for name2, signal2 in signals.items():
                if name1 != name2:
                    # Compute instantaneous phases
                    analytic1 = hilbert(signal1 - np.mean(signal1))
                    analytic2 = hilbert(signal2 - np.mean(signal2))
                    
                    phase1 = np.angle(analytic1)
                    phase2 = np.angle(analytic2)
                    
                    # Phase difference
                    phase_diff = phase1 - phase2
                    
                    # Phase locking value
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    
                    # Phase coherence
                    coherence = 1 - np.var(phase_diff) / (np.pi**2)
                    
                    phase_relationships[f"{name1}_{name2}"] = {
                        'phase_locking_value': plv,
                        'phase_coherence': max(0, coherence),
                        'mean_phase_diff': np.mean(phase_diff),
                        'phase_diff_std': np.std(phase_diff)
                    }
        
        return {
            'phase_relationships': phase_relationships,
            'strongest_coupling': max(phase_relationships.items(), 
                                    key=lambda x: x[1]['phase_locking_value']) if phase_relationships else None,
            'phase_synchronization_index': np.mean([pr['phase_locking_value'] 
                                                  for pr in phase_relationships.values()]) if phase_relationships else 0
        }
    
    def _perform_frequency_domain_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform frequency domain analysis using universal framework"""
        
        # Prepare data for universal framework
        time_column = 'time' if 'time' in df.columns else None
        
        if time_column is None:
            # Create time column
            df = df.copy()
            df['time'] = np.arange(len(df))
            time_column = 'time'
        
        try:
            # Use universal transformation framework
            results = analyze_oscillations(df, time_column=time_column)
            
            return {
                'universal_analysis': results,
                'dominant_frequencies': self._extract_dominant_frequencies(results),
                'gait_specific_patterns': self._identify_gait_patterns(results),
                'surface_coupling_evidence': self._find_surface_coupling_evidence(results)
            }
            
        except Exception as e:
            logger.warning(f"Universal framework analysis failed: {e}")
            return {'error': str(e)}
    
    def _extract_dominant_frequencies(self, results: Dict) -> Dict[str, List[float]]:
        """Extract dominant frequencies from universal analysis results"""
        
        dominant_freqs = {}
        
        if 'individual_signals' in results:
            for signal_result in results['individual_signals']:
                signal_name = signal_result['signal_name']
                transform = signal_result['transformation']
                
                if 'frequency_response' in transform.laplace_transform:
                    freq_resp = transform.laplace_transform['frequency_response']
                    dom_freqs = freq_resp.get('dominant_frequencies', [])
                    dominant_freqs[signal_name] = dom_freqs.tolist() if hasattr(dom_freqs, 'tolist') else list(dom_freqs)
        
        return dominant_freqs
    
    def _identify_gait_patterns(self, results: Dict) -> Dict[str, Any]:
        """Identify gait-specific patterns from universal analysis"""
        
        patterns = {}
        
        if 'individual_signals' in results:
            for signal_result in results['individual_signals']:
                signal_name = signal_result['signal_name']
                
                # Check if signal matches expected gait frequencies
                if 'validated_patterns' in signal_result['transformation'].pattern_discovery:
                    gait_patterns = []
                    for pattern in signal_result['transformation'].pattern_discovery['validated_patterns']:
                        # Check if frequencies match gait ranges
                        for freq in pattern.frequency_components:
                            for gait_param, freq_range in self.gait_frequencies.items():
                                if freq_range[0] <= freq <= freq_range[1]:
                                    gait_patterns.append({
                                        'gait_parameter': gait_param,
                                        'detected_frequency': freq,
                                        'expected_range': freq_range,
                                        'pattern_type': pattern.pattern_type.value,
                                        'confidence': pattern.confidence
                                    })
                    
                    patterns[signal_name] = gait_patterns
        
        return patterns
    
    def _find_surface_coupling_evidence(self, results: Dict) -> Dict[str, Any]:
        """Find evidence of surface coupling in universal analysis results"""
        
        evidence = {}
        
        # Look for cross-signal coupling that might indicate surface effects
        if 'cross_signal_analysis' in results:
            cross_analysis = results['cross_signal_analysis']
            
            # Strong coupling between stance_time and step_length suggests surface effects
            if 'frequency_couplings' in cross_analysis:
                for pair, harmonics in cross_analysis['frequency_couplings'].items():
                    if 'stance' in pair and 'step' in pair:
                        evidence['stance_step_coupling'] = {
                            'harmonic_relationships': len(harmonics),
                            'coupling_strength': 'strong' if len(harmonics) > 2 else 'moderate',
                            'surface_indication': True
                        }
        
        # Look for frequency modulation that might indicate surface adaptation
        if 'meta_analysis' in results:
            complexity = results['meta_analysis'].get('complexity_metrics', {})
            if complexity.get('complexity_score', 0) > 5:
                evidence['complexity_indication'] = {
                    'high_complexity': True,
                    'possible_surface_adaptation': True,
                    'complexity_score': complexity.get('complexity_score', 0)
                }
        
        return evidence
    
    def _compute_gait_efficiency(self, df: pd.DataFrame, surface_analysis: Dict) -> Dict[str, Any]:
        """Compute gait efficiency metrics"""
        
        efficiency_metrics = {}
        
        # Energy cost estimate
        if all(col in df.columns for col in ['speed', 'cadence', 'vertical_oscillation']):
            mean_speed = df['speed'].mean()
            mean_cadence = df['cadence'].mean() / 60  # Convert to Hz
            mean_vert_osc = df['vertical_oscillation'].mean() / 1000  # Convert to m
            
            # Simplified energy cost model
            metabolic_cost = (
                2.0 * mean_speed +  # Speed cost
                0.5 * mean_cadence**2 +  # Cadence cost (quadratic)
                10.0 * mean_vert_osc  # Vertical oscillation cost
            )
            
            # Surface compliance adjustment
            surface_factor = surface_analysis['estimated_surface_compliance'].compliance_factor
            adjusted_cost = metabolic_cost * (1 + 0.3 * surface_factor)
            
            efficiency_metrics['energy_cost'] = {
                'base_metabolic_cost': metabolic_cost,
                'surface_adjusted_cost': adjusted_cost,
                'efficiency_index': mean_speed / adjusted_cost,
                'surface_penalty': 0.3 * surface_factor
            }
        
        # Mechanical efficiency
        if 'step_length' in df.columns and 'cadence' in df.columns:
            step_length = df['step_length'].mean() / 1000  # Convert to m
            cadence = df['cadence'].mean() / 60  # Convert to Hz
            
            stride_length = 2 * step_length
            stride_frequency = cadence / 2
            
            # Optimal stride frequency (from literature)
            optimal_stride_freq = 0.85  # Hz
            frequency_efficiency = 1 - abs(stride_frequency - optimal_stride_freq) / optimal_stride_freq
            
            efficiency_metrics['mechanical_efficiency'] = {
                'stride_frequency_hz': stride_frequency,
                'optimal_frequency_hz': optimal_stride_freq,
                'frequency_efficiency': max(0, frequency_efficiency),
                'stride_length_m': stride_length
            }
        
        return efficiency_metrics
    
    def _optimize_surface_compliance(self, df: pd.DataFrame, 
                                   oscillatory_params: Dict) -> Dict[str, Any]:
        """Optimize surface compliance for gait efficiency"""
        
        def gait_efficiency_objective(compliance_factor):
            """Objective function for surface compliance optimization"""
            
            # Model how different compliance affects gait parameters
            if 'cadence' in oscillatory_params:
                base_cadence = oscillatory_params['cadence']['mean_hz']
                adjusted_cadence = base_cadence * (1 - 0.05 * compliance_factor)
                cadence_cost = abs(adjusted_cadence - 1.67)**2  # Penalty from optimal
            else:
                cadence_cost = 0
            
            if 'step_length' in oscillatory_params:
                base_step_length = oscillatory_params['step_length']['mean_m']  
                adjusted_step_length = base_step_length * (1 - 0.03 * compliance_factor)
                step_cost = abs(adjusted_step_length - 0.8)**2  # Penalty from optimal
            else:
                step_cost = 0
            
            # Energy cost increases with compliance
            energy_cost = compliance_factor * 0.5
            
            return cadence_cost + step_cost + energy_cost
        
        # Optimize compliance factor
        try:
            result = optimize.minimize_scalar(
                gait_efficiency_objective,
                bounds=(0, 1),
                method='bounded'
            )
            
            optimal_compliance = result.x
            optimal_cost = result.fun
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            optimal_compliance = 0.3  # Default moderate compliance
            optimal_cost = 1.0
        
        return {
            'optimal_compliance_factor': optimal_compliance,
            'optimal_surface_type': self._compliance_to_surface_type(optimal_compliance),
            'cost_reduction': optimal_cost,
            'optimization_successful': result.success if 'result' in locals() else False,
            'recommendations': self._generate_surface_recommendations(optimal_compliance)
        }
    
    def _compliance_to_surface_type(self, compliance_factor: float) -> SurfaceType:
        """Convert compliance factor to surface type"""
        if compliance_factor < 0.25:
            return SurfaceType.RIGID
        elif compliance_factor < 0.6:
            return SurfaceType.COMPLIANT
        elif compliance_factor < 0.85:
            return SurfaceType.SOFT
        else:
            return SurfaceType.VARIABLE
    
    def _generate_surface_recommendations(self, optimal_compliance: float) -> List[str]:
        """Generate surface recommendations based on optimal compliance"""
        
        recommendations = []
        
        if optimal_compliance < 0.3:
            recommendations.extend([
                "Consider firmer surfaces for improved efficiency",
                "Track or road running may be optimal",
                "Minimize soft surface training"
            ])
        elif optimal_compliance < 0.7:
            recommendations.extend([
                "Mixed surface training is beneficial",
                "Alternate between track and grass/trail",
                "Moderate compliance surfaces optimal"
            ])
        else:
            recommendations.extend([
                "Soft surface training may be beneficial",
                "Beach/sand running could improve strength",
                "Focus on compliance adaptation"
            ])
        
        return recommendations
    
    def _assess_optimization_potential(self, compliance_effects: Dict) -> Dict[str, Any]:
        """Assess potential for gait optimization through surface modification"""
        
        potential_improvements = {}
        
        for param, effects in compliance_effects.items():
            if 'compliance_effect' in effects:
                effect_magnitude = abs(effects['compliance_effect'])
                
                if effect_magnitude > 0.1:
                    potential_improvements[param] = {
                        'improvement_potential': 'high',
                        'effect_magnitude': effect_magnitude,
                        'optimization_value': effect_magnitude * 10  # Arbitrary scaling
                    }
                elif effect_magnitude > 0.05:
                    potential_improvements[param] = {
                        'improvement_potential': 'moderate',
                        'effect_magnitude': effect_magnitude,
                        'optimization_value': effect_magnitude * 10
                    }
                else:
                    potential_improvements[param] = {
                        'improvement_potential': 'low',
                        'effect_magnitude': effect_magnitude,
                        'optimization_value': effect_magnitude * 10
                    }
        
        total_potential = sum(pi['optimization_value'] for pi in potential_improvements.values())
        
        return {
            'parameter_improvements': potential_improvements,
            'total_optimization_potential': total_potential,
            'priority_parameters': sorted(potential_improvements.items(),
                                        key=lambda x: x[1]['optimization_value'],
                                        reverse=True)[:3]
        }

def main():
    """Test biomechanical oscillatory analysis"""
    
    # Create sample gait data
    sample_gait_data = [
        {"time": 9, "dist": 38.96, "heart_rate": 111.0, "stance_time": 169.0, "cadence": 102.0,
         "speed": 5.02, "stance_time_balance": 50.03, "step_length": 364.0, "vertical_ratio": 5.84,
         "vertical_oscillation": 97.5, "com_acceleration": 0.85},
        {"time": 12, "dist": 53.57, "heart_rate": 113.0, "stance_time": 167.0, "cadence": 101.0,
         "speed": 5.412, "stance_time_balance": 49.87, "step_length": 1658.0, "vertical_ratio": 5.84,
         "vertical_oscillation": 92.7, "com_acceleration": 0.63},
        {"time": 14, "dist": 69.81, "heart_rate": 120.0, "stance_time": 166.0, "cadence": 100.0,
         "speed": 5.598, "stance_time_balance": 49.87, "step_length": 1665.0, "vertical_ratio": 5.65,
         "vertical_oscillation": 93.7, "com_acceleration": 0.52},
        {"time": 15, "dist": 78.12, "heart_rate": 120.0, "stance_time": 168.0, "cadence": 100.0,
         "speed": 5.589, "stance_time_balance": 50.0, "step_length": 1672.0, "vertical_ratio": 5.46,
         "vertical_oscillation": 93.7, "com_acceleration": 0.47},
        {"time": 16, "dist": 86.67, "heart_rate": 120.0, "stance_time": 168.0, "cadence": 99.0,
         "speed": 5.589, "stance_time_balance": 50.09, "step_length": 1701.0, "vertical_ratio": 5.34,
         "vertical_oscillation": 93.7, "com_acceleration": 0.42}
    ]
    
    print("=== BIOMECHANICAL OSCILLATORY SYSTEM ANALYSIS ===")
    
    analyzer = BiomechanicalOscillatoryAnalyzer()
    results = analyzer.analyze_gait_oscillations(sample_gait_data)
    
    print("\nGait Oscillatory Parameters:")
    for param, data in results['oscillatory_parameters'].items():
        print(f"  {param}: mean = {data.get('mean_hz', data.get('mean_m', data.get('mean_s', data.get('mean_ms2', 'N/A')))):.4f}")
    
    print(f"\nEstimated Surface: {results['surface_compliance_analysis']['estimated_surface_compliance'].surface_type.value}")
    print(f"Compliance Factor: {results['surface_compliance_analysis']['estimated_surface_compliance'].compliance_factor:.3f}")
    
    if 'energy_cost' in results['gait_efficiency_metrics']:
        efficiency = results['gait_efficiency_metrics']['energy_cost']['efficiency_index']
        print(f"\nGait Efficiency Index: {efficiency:.3f}")
    
    optimal = results['surface_optimization']
    print(f"\nOptimal Surface Compliance: {optimal['optimal_compliance_factor']:.3f}")
    print(f"Recommended Surface: {optimal['optimal_surface_type'].value}")
    
    print("\nSurface Recommendations:")
    for rec in optimal['recommendations']:
        print(f"  • {rec}")

if __name__ == "__main__":
    main()
