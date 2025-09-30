# Contributing to Universal Oscillatory Framework for Cardiovascular Analysis

We welcome contributions to the Universal Oscillatory Framework! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Scientific Contributions](#scientific-contributions)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of signal processing and cardiovascular physiology
- Familiarity with the Universal Oscillatory Framework principles

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/huygens.git
   cd huygens
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev,docs]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

6. Run tests to verify setup:
   ```bash
   pytest
   ```

## Development Process

### Branching Strategy

- `main`: Stable release branch
- `develop`: Development branch for new features
- `feature/your-feature`: Feature branches
- `hotfix/issue-description`: Emergency fixes

### Workflow

1. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes
3. Write or update tests
4. Update documentation
5. Run the test suite:
   ```bash
   pytest
   ```

6. Check code style:
   ```bash
   black src/ demo/ tests/
   flake8 src/ demo/ tests/
   mypy src/
   ```

7. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new oscillatory analysis feature"
   ```

8. Push to your fork and create a pull request

### Commit Message Convention

We use conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Code Style

### Python Style Guidelines

- Follow PEP 8
- Use Black for code formatting (line length: 88)
- Use type hints for all public functions
- Write descriptive docstrings following NumPy style

### Example Function:

```python
def extract_oscillatory_signature(
    signal: np.ndarray, 
    sampling_rate: float,
    scales: Optional[List[str]] = None
) -> OscillatorySignature:
    """
    Extract multi-scale oscillatory signatures from cardiovascular signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input cardiovascular signal (ECG, PPG, etc.)
    sampling_rate : float
        Signal sampling rate in Hz
    scales : List[str], optional
        Oscillatory scales to analyze. If None, all scales are used.
        
    Returns
    -------
    OscillatorySignature
        Extracted oscillatory signatures across specified scales
        
    Raises
    ------
    ValueError
        If sampling_rate is not positive
    SignalProcessingError
        If signal quality is insufficient for analysis
        
    Notes
    -----
    This function implements the Universal Oscillatory Framework
    principles as described in Sachikonye (2024).
    
    Examples
    --------
    >>> ecg_data = np.random.randn(1000)
    >>> signature = extract_oscillatory_signature(ecg_data, 250.0)
    >>> print(signature.quantum_coherence)
    """
    pass
```

## Testing

### Test Structure

- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Demo tests in `demo/tests/`

### Writing Tests

- Use pytest framework
- Aim for >90% code coverage
- Test both normal and edge cases
- Include oscillatory framework validation tests

### Example Test:

```python
import pytest
import numpy as np
from src.cardiovascular_oscillatory_suite import UniversalCardiovascularFramework

class TestOscillatoryExtraction:
    """Test oscillatory signature extraction functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.framework = UniversalCardiovascularFramework()
        self.sample_ecg = np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, 2500))  # 1 Hz sine wave
        
    def test_extract_signature_valid_input(self):
        """Test signature extraction with valid input."""
        signature = self.framework.extract_oscillatory_signature(
            self.sample_ecg, 
            sampling_rate=250.0
        )
        
        assert signature is not None
        assert hasattr(signature, 'quantum_coherence')
        assert signature.scales_analyzed == 8
        
    def test_extract_signature_invalid_sampling_rate(self):
        """Test signature extraction with invalid sampling rate."""
        with pytest.raises(ValueError, match="sampling_rate must be positive"):
            self.framework.extract_oscillatory_signature(
                self.sample_ecg, 
                sampling_rate=-1.0
            )
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_oscillatory_framework.py

# Run tests with specific marker
pytest -m "not slow"
```

## Documentation

### Code Documentation

- All public functions must have comprehensive docstrings
- Use NumPy docstring style
- Include mathematical equations when relevant
- Reference scientific papers appropriately

### API Documentation

Documentation is built with Sphinx:

```bash
cd docs
make html
```

### Scientific Documentation

When contributing scientific features:

- Include references to peer-reviewed literature
- Provide mathematical derivations in LaTeX format
- Add validation studies comparing with established methods
- Document oscillatory framework theoretical foundations

## Submitting Changes

### Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG.md
4. Create pull request with:
   - Clear title and description
   - Reference any related issues
   - Include test results
   - Describe scientific validation if applicable

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Scientific enhancement

## Scientific Validation
- [ ] Theoretical foundation verified
- [ ] Comparison with existing methods
- [ ] Performance benchmarks included

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Code documentation updated
- [ ] API documentation generated
- [ ] Scientific references added

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] Documentation updated
```

## Scientific Contributions

### Oscillatory Framework Enhancements

When contributing to the oscillatory framework:

1. **Theoretical Basis**: Ensure new features align with Universal Oscillatory Framework principles
2. **Mathematical Rigor**: Include formal mathematical derivations
3. **Experimental Validation**: Provide empirical validation against established methods
4. **Performance Analysis**: Document computational complexity improvements

### Adding New Oscillatory Scales

To add new oscillatory scales:

1. Define the frequency range and physiological significance
2. Implement extraction algorithms
3. Validate against known physiological phenomena
4. Update the eight-scale hierarchy documentation

### Cardiovascular Analysis Improvements

For cardiovascular-specific enhancements:

1. Validate against clinical standards (ESC guidelines, etc.)
2. Test with diverse patient populations
3. Ensure real-time processing compatibility
4. Document clinical significance

## Questions and Support

- Create an issue for bugs or feature requests
- Use discussions for questions about the framework
- Email maintainers for sensitive security issues
- Join our research community for scientific discussions

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Academic publications (for significant scientific contributions)
- Release notes
- Documentation

Thank you for contributing to the advancement of oscillatory cardiovascular analysis!
