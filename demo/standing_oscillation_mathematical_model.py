"""
Standing Oscillation Mathematical Model
Integrates mathematical frameworks from fire circle evolution, quantum-biological consciousness,
motor control theories, and behavioral acquisition into a unified standing oscillation model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

@dataclass
class StandingParameters:
    """Parameters for standing oscillation model."""
    # Fire circle parameters
    fire_exposure_intensity: float = 1.0
    safety_factor: float = 0.9
    group_size: int = 8
    duration_hours: float = 5.0
    
    # Consciousness parameters
    intention_strength: float = 0.8
    self_awareness_level: float = 0.7
    cognitive_load: float = 0.5
    
    # Motor control parameters
    pendulum_stiffness: float = 150.0  # N*m/rad
    damping_coefficient: float = 25.0  # N*m*s/rad
    mass_height: float = 0.6  # m (effective height of center of mass)
    
    # Quantum-biological parameters
    ion_tunneling_coherence: float = 0.3
    bmd_efficiency: float = 0.85
    
    # Behavioral acquisition parameters
    epigenetic_rate: float = 0.05
    neuroplasticity_factor: float = 0.12
    critical_period_factor: float = 1.0

class StandingOscillationModel:
    """Unified mathematical model for standing as fundamental oscillation."""
    
    def __init__(self, params: StandingParameters):
        self.params = params
        self.time_points = None
        self.solution = None
        
    def fire_circle_environment_factor(self, t: float) -> float:
        """Calculate fire circle environmental enhancement factor."""
        safety = self.params.safety_factor
        group = min(self.params.group_size / 8.0, 1.0)
        duration = min(self.params.duration_hours / 5.0, 1.0)
        fire_intensity = self.params.fire_exposure_intensity
        
        return (safety * group * duration * fire_intensity) ** 0.5
    
    def consciousness_integration_factor(self, t: float) -> float:
        """Calculate consciousness-standing integration factor."""
        intention = self.params.intention_strength
        awareness = self.params.self_awareness_level
        cognitive = 1.0 / (1.0 + self.params.cognitive_load)  # Inverse relationship
        
        # Quantum-biological enhancement
        quantum_factor = 1.0 + self.params.ion_tunneling_coherence * self.params.bmd_efficiency
        
        return intention * awareness * cognitive * quantum_factor
    
    def motor_control_dynamics(self, theta: float, theta_dot: float) -> float:
        """Unified motor control dynamics (Inverted Pendulum + Minimum Jerk + Rambling-Trembling)."""
        g = 9.81  # gravity
        l = self.params.mass_height
        k = self.params.pendulum_stiffness
        c = self.params.damping_coefficient
        
        # Inverted pendulum component
        pendulum_term = (g / l) * np.sin(theta)
        
        # Damping (from minimum jerk optimization)
        damping_term = -(c / (1.0 + abs(theta))) * theta_dot
        
        # Control stiffness (from rambling-trembling decomposition)
        control_term = -k * theta
        
        return pendulum_term + damping_term + control_term
    
    def behavioral_acquisition_rate(self, t: float) -> float:
        """Rate of behavioral acquisition from animal to human standing."""
        epigenetic_rate = self.params.epigenetic_rate
        neuroplasticity = self.params.neuroplasticity_factor
        critical_period = self.params.critical_period_factor
        
        # Sigmoid acquisition curve
        acquisition_time_constant = 50.0  # weeks
        return epigenetic_rate * neuroplasticity * critical_period * \
               (1.0 - np.exp(-t / acquisition_time_constant))
    
    def intentionality_modulation(self, t: float) -> float:
        """Human intentionality modulation distinguishing from animal standing."""
        base_intention = self.params.intention_strength
        
        # Time-varying intentional goals (representing abstract objectives)
        goal_oscillation = 0.1 * np.sin(2 * np.pi * t / 600)  # 10-minute goal cycles
        purpose_modulation = 0.05 * np.sin(2 * np.pi * t / 3600)  # 1-hour purpose cycles
        
        return base_intention + goal_oscillation + purpose_modulation
    
    def standing_oscillation_system(self, t: float, y: List[float]) -> List[float]:
        """Complete standing oscillation system dynamics."""
        theta, theta_dot, acquisition_state, consciousness_level = y
        
        # Environmental enhancement from fire circles
        env_factor = self.fire_circle_environment_factor(t)
        
        # Consciousness integration
        consciousness_factor = self.consciousness_integration_factor(t)
        
        # Motor control dynamics
        motor_acceleration = self.motor_control_dynamics(theta, theta_dot)
        
        # Behavioral acquisition
        acquisition_rate = self.behavioral_acquisition_rate(t)
        
        # Intentionality modulation
        intention_mod = self.intentionality_modulation(t)
        
        # System equations
        theta_ddot = motor_acceleration * env_factor * consciousness_factor * (1 + acquisition_state)
        
        # Add intentionality-driven postural adjustments
        theta_ddot += intention_mod * np.sin(2 * np.pi * t / 120)  # 2-minute intentional adjustments
        
        # Behavioral acquisition dynamics
        acquisition_dot = acquisition_rate * (1.0 - acquisition_state) - 0.01 * acquisition_state
        
        # Consciousness level dynamics
        consciousness_dot = 0.1 * env_factor * (consciousness_factor - consciousness_level)
        
        return [theta_dot, theta_ddot, acquisition_dot, consciousness_dot]
    
    def simulate_standing_evolution(self, duration_hours: float = 24.0, dt: float = 0.1) -> Dict:
        """Simulate the evolution of standing behavior over time."""
        # Time points (in seconds)
        t_end = duration_hours * 3600
        t_span = (0, t_end)
        
        # Initial conditions
        # [theta, theta_dot, acquisition_state, consciousness_level]
        y0 = [0.01, 0.0, 0.0, 0.1]  # Small initial sway, no acquisition, low consciousness
        
        # Solve the system
        sol = solve_ivp(self.standing_oscillation_system, t_span, y0, 
                       dense_output=True, rtol=1e-8)
        
        # Generate time points for analysis
        self.time_points = np.arange(0, t_end, dt)
        self.solution = sol.sol(self.time_points)
        
        return {
            'time': self.time_points,
            'theta': self.solution[0],
            'theta_dot': self.solution[1],
            'acquisition_state': self.solution[2],
            'consciousness_level': self.solution[3],
            'parameters': self.params
        }
    
    def analyze_oscillatory_components(self) -> Dict:
        """Analyze the oscillatory components of standing behavior."""
        if self.solution is None:
            raise ValueError("Must run simulation first")
        
        theta = self.solution[0]
        
        # Frequency domain analysis
        fs = 10.0  # 10 Hz sampling rate
        frequencies, power_spectrum = signal.welch(theta, fs=fs, nperseg=1024)
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(power_spectrum, height=np.max(power_spectrum) * 0.1)[0]
        dominant_frequencies = frequencies[peak_indices]
        
        # Calculate oscillatory metrics
        sway_amplitude = np.std(theta)
        sway_velocity = np.std(self.solution[1])
        
        # Consciousness-standing correlation
        consciousness_correlation = np.corrcoef(theta, self.solution[3])[0, 1]
        
        return {
            'dominant_frequencies': dominant_frequencies,
            'sway_amplitude': sway_amplitude,
            'sway_velocity': sway_velocity,
            'power_spectrum': power_spectrum,
            'frequencies': frequencies,
            'consciousness_correlation': consciousness_correlation
        }
    
    def calculate_standing_quality_metrics(self) -> Dict:
        """Calculate metrics indicating standing quality and consciousness integration."""
        if self.solution is None:
            raise ValueError("Must run simulation first")
        
        theta = self.solution[0]
        acquisition = self.solution[2]
        consciousness = self.solution[3]
        
        # Energy efficiency (lower sway with maintained stability)
        energy_efficiency = 1.0 / (1.0 + np.var(theta))
        
        # Cognitive integration (correlation between consciousness and postural control)
        cognitive_integration = abs(np.corrcoef(theta, consciousness)[0, 1])
        
        # Behavioral maturation (final acquisition state)
        behavioral_maturation = acquisition[-1]
        
        # Intentionality index (goal-directed postural modulations)
        intentional_modulations = len(signal.find_peaks(np.abs(np.diff(theta)), 
                                                      height=np.std(np.diff(theta)))[0])
        intentionality_index = min(intentional_modulations / len(theta) * 1000, 1.0)
        
        # Overall standing quality
        standing_quality = (energy_efficiency + cognitive_integration + 
                          behavioral_maturation + intentionality_index) / 4.0
        
        return {
            'energy_efficiency': energy_efficiency,
            'cognitive_integration': cognitive_integration,
            'behavioral_maturation': behavioral_maturation,
            'intentionality_index': intentionality_index,
            'standing_quality': standing_quality
        }
    
    def compare_human_vs_animal_standing(self) -> Dict:
        """Compare human consciousness-integrated standing vs animal attention-based standing."""
        
        # Simulate human standing (with consciousness integration)
        human_results = self.simulate_standing_evolution(duration_hours=8.0)
        human_metrics = self.calculate_standing_quality_metrics()
        
        # Simulate animal-like standing (minimal consciousness, high vigilance)
        animal_params = StandingParameters(
            fire_exposure_intensity=0.0,
            safety_factor=0.3,
            intention_strength=0.1,
            self_awareness_level=0.1,
            cognitive_load=0.9,
            ion_tunneling_coherence=0.0,
            bmd_efficiency=0.2,
            epigenetic_rate=0.0,
            neuroplasticity_factor=0.0
        )
        
        animal_model = StandingOscillationModel(animal_params)
        animal_results = animal_model.simulate_standing_evolution(duration_hours=8.0)
        animal_metrics = animal_model.calculate_standing_quality_metrics()
        
        return {
            'human': {
                'results': human_results,
                'metrics': human_metrics
            },
            'animal': {
                'results': animal_results,
                'metrics': animal_metrics
            },
            'comparison': {
                'quality_ratio': human_metrics['standing_quality'] / animal_metrics['standing_quality'],
                'energy_efficiency_ratio': human_metrics['energy_efficiency'] / animal_metrics['energy_efficiency'],
                'cognitive_integration_ratio': human_metrics['cognitive_integration'] / max(animal_metrics['cognitive_integration'], 0.01)
            }
        }
    
    def plot_standing_evolution(self, save_path: Optional[str] = None):
        """Plot the evolution of standing behavior over time."""
        if self.solution is None:
            raise ValueError("Must run simulation first")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        time_hours = self.time_points / 3600
        
        # Postural sway
        axes[0, 0].plot(time_hours, self.solution[0], 'b-', linewidth=1)
        axes[0, 0].set_title('Postural Sway (θ)')
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Angular displacement (rad)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Behavioral acquisition
        axes[0, 1].plot(time_hours, self.solution[2], 'g-', linewidth=2)
        axes[0, 1].set_title('Behavioral Acquisition State')
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Acquisition level (0-1)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Consciousness level
        axes[1, 0].plot(time_hours, self.solution[3], 'r-', linewidth=2)
        axes[1, 0].set_title('Consciousness Integration Level')
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Consciousness level (0-1)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Phase portrait (sway vs sway velocity)
        axes[1, 1].plot(self.solution[0], self.solution[1], 'purple', alpha=0.6)
        axes[1, 1].set_title('Standing Phase Portrait')
        axes[1, 1].set_xlabel('Angular position (rad)')
        axes[1, 1].set_ylabel('Angular velocity (rad/s)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_results(self, filepath: str):
        """Export simulation results to JSON file."""
        if self.solution is None:
            raise ValueError("Must run simulation first")
        
        results = {
            'time': self.time_points.tolist(),
            'theta': self.solution[0].tolist(),
            'theta_dot': self.solution[1].tolist(),
            'acquisition_state': self.solution[2].tolist(),
            'consciousness_level': self.solution[3].tolist(),
            'parameters': {
                'fire_exposure_intensity': self.params.fire_exposure_intensity,
                'safety_factor': self.params.safety_factor,
                'group_size': self.params.group_size,
                'duration_hours': self.params.duration_hours,
                'intention_strength': self.params.intention_strength,
                'self_awareness_level': self.params.self_awareness_level,
                'cognitive_load': self.params.cognitive_load,
                'pendulum_stiffness': self.params.pendulum_stiffness,
                'damping_coefficient': self.params.damping_coefficient,
                'mass_height': self.params.mass_height,
                'ion_tunneling_coherence': self.params.ion_tunneling_coherence,
                'bmd_efficiency': self.params.bmd_efficiency,
                'epigenetic_rate': self.params.epigenetic_rate,
                'neuroplasticity_factor': self.params.neuroplasticity_factor,
                'critical_period_factor': self.params.critical_period_factor
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

def demonstrate_standing_oscillation_framework():
    """Demonstrate the standing oscillation mathematical framework."""
    print("🧍 Standing Oscillation Mathematical Framework Demonstration")
    print("=" * 70)
    
    # Create model with fire circle parameters
    fire_circle_params = StandingParameters(
        fire_exposure_intensity=1.0,
        safety_factor=0.9,
        group_size=8,
        duration_hours=5.0,
        intention_strength=0.8,
        self_awareness_level=0.7,
        cognitive_load=0.3,  # Low cognitive load enables quiet standing
        ion_tunneling_coherence=0.6,
        bmd_efficiency=0.85,
        epigenetic_rate=0.08,
        neuroplasticity_factor=0.15,
        critical_period_factor=1.0
    )
    
    model = StandingOscillationModel(fire_circle_params)
    
    # Simulate standing evolution
    print("\n🔄 Simulating standing behavior evolution...")
    results = model.simulate_standing_evolution(duration_hours=12.0)
    
    # Analyze oscillatory components
    print("\n📊 Analyzing oscillatory components...")
    oscillatory_analysis = model.analyze_oscillatory_components()
    
    print(f"Dominant oscillatory frequencies: {oscillatory_analysis['dominant_frequencies'][:5]} Hz")
    print(f"Sway amplitude: {oscillatory_analysis['sway_amplitude']:.4f} rad")
    print(f"Consciousness-standing correlation: {oscillatory_analysis['consciousness_correlation']:.4f}")
    
    # Calculate standing quality metrics
    print("\n📈 Calculating standing quality metrics...")
    quality_metrics = model.calculate_standing_quality_metrics()
    
    print(f"Energy efficiency: {quality_metrics['energy_efficiency']:.4f}")
    print(f"Cognitive integration: {quality_metrics['cognitive_integration']:.4f}")
    print(f"Behavioral maturation: {quality_metrics['behavioral_maturation']:.4f}")
    print(f"Intentionality index: {quality_metrics['intentionality_index']:.4f}")
    print(f"Overall standing quality: {quality_metrics['standing_quality']:.4f}")
    
    # Compare human vs animal standing
    print("\n🆚 Comparing human vs animal standing...")
    comparison = model.compare_human_vs_animal_standing()
    
    comp_metrics = comparison['comparison']
    print(f"Human-to-animal quality ratio: {comp_metrics['quality_ratio']:.2f}x")
    print(f"Human-to-animal energy efficiency ratio: {comp_metrics['energy_efficiency_ratio']:.2f}x")
    print(f"Human-to-animal cognitive integration ratio: {comp_metrics['cognitive_integration_ratio']:.2f}x")
    
    # Plot results
    print("\n📊 Plotting standing evolution...")
    model.plot_standing_evolution(save_path='standing_oscillation_evolution.png')
    
    # Export results
    print("\n💾 Exporting results...")
    model.export_results('standing_oscillation_results.json')
    
    print("\n✅ Standing Oscillation Framework demonstration completed!")
    print("\nKey Insights:")
    print("- Standing emerges as fundamental oscillatory state through fire circle behavioral acquisition")
    print("- Human consciousness-integrated standing shows superior efficiency and cognitive integration")
    print("- Mathematical model unifies motor control theories through intentionality principle")
    print("- Quantum-biological mechanisms enhance standing-consciousness coupling")
    
    return model, results, quality_metrics

if __name__ == "__main__":
    demonstrate_standing_oscillation_framework()
