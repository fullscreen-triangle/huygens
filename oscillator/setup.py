from setuptools import setup, find_packages

setup(
    name="st-stellas-oscillator-demo",
    version="1.0.0",
    author="Kundai Farai Sachikonye",
    author_email="kundai.sachikonye@wzw.tum.de",
    description="St. Stellas Grand Equivalent Circuit Oscillator Demo Package",
    long_description="""
    A comprehensive demonstration package for St. Stellas oscillators that operate through 
    tri-dimensional S-entropy coordinates. This package includes traditional oscillators 
    with their St. Stellas Grand Equivalent Circuit transformations, enabling simultaneous 
    multi-dimensional oscillatory behavior while maintaining circuit equivalence principles.
    
    Features:
    - Traditional oscillators (Van der Pol, Harmonic, Duffing, LC Tank, etc.)
    - St. Stellas Grand Equivalent Circuit transformations
    - Tri-dimensional S-entropy coordinate analysis
    - Miraculous circuit behavior demonstrations
    - Cross-domain pattern transfer validation
    - Comprehensive oscillator testing framework
    """,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "sympy>=1.9.0",
        "pandas>=1.3.0",
        "control>=0.9.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "st-stellas-demo=st_stellas_oscillator.cli:main",
            "oscillator-transform=st_stellas_oscillator.transform:main",
            "run-oscillator-tests=st_stellas_oscillator.tests:run_all_tests",
        ],
    },
    include_package_data=True,
    package_data={
        "st_stellas_oscillator": ["data/*.json", "templates/*.py"],
    },
)
