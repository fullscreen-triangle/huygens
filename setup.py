"""
Universal Oscillatory Framework for Cardiovascular Analysis

A scientific computing framework implementing the Universal Oscillatory Framework 
for cardiovascular signal processing and physiological analysis.
"""

from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.0'
DESCRIPTION = 'Universal Oscillatory Framework for Cardiovascular Analysis'
LONG_DESCRIPTION = 'A comprehensive computational framework implementing multi-scale oscillatory coupling for cardiovascular signal processing, achieving O(1) computational complexity through S-entropy coordinate navigation.'

# Setting up
setup(
    name="cardiovascular-oscillatory-framework",
    version=VERSION,
    author="Kundai Farai Sachikonye",
    author_email="kundai.sachikonye@wzw.tum.de",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "scipy>=1.7.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
        "matplotlib>=3.4.0,<4.0.0",
        "seaborn>=0.11.0,<1.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "heartpy>=1.2.7,<2.0.0",
        "neurokit2>=0.2.0,<1.0.0",
        "pyhrv>=0.4.0,<1.0.0",
        "pywavelets>=1.3.0,<2.0.0",
        "pydantic>=1.8.0,<2.0.0",
        "jsonschema>=4.0.0,<5.0.0",
        "pyyaml>=6.0.0,<7.0.0",
        "numba>=0.56.0,<1.0.0",
        "plotly>=5.0.0,<6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0,<8.0.0",
            "pytest-cov>=4.0.0,<5.0.0",
            "black>=22.0.0,<24.0.0",
            "flake8>=4.0.0,<7.0.0",
            "mypy>=0.950,<2.0.0",
            "pre-commit>=2.17.0,<4.0.0",
        ],
        "docs": [
            "sphinx>=4.5.0,<7.0.0",
            "sphinx-rtd-theme>=1.0.0,<2.0.0",
            "myst-parser>=0.17.0,<1.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=11.0.0,<12.0.0",
        ],
        "ml": [
            "tensorflow>=2.8.0,<3.0.0",
            "torch>=1.11.0,<2.0.0",
        ],
    },
    keywords=['cardiovascular', 'oscillatory', 'signal-processing', 'physiology', 'hrv', 'ecg', 'ppg'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'cardiovascular-analysis=src.analyze_cardiovascular_data:main',
            'oscillatory-demo=demo.run_comprehensive_demo:main',
        ],
    },
    include_package_data=True,
    package_data={
        'src': ['*.json', '*.yaml', '*.yml'],
        'demo': ['*.json', '*.yaml'],
        'docs': ['*.md', '*.rst'],
        'experimental-data': ['**/*.json', '**/*.csv'],
    },
    zip_safe=False,
    project_urls={
        "Bug Tracker": "https://github.com/fullscreen-triangle/huygens/issues",
        "Documentation": "https://huygens.readthedocs.io/",
        "Source Code": "https://github.com/fullscreen-triangle/huygens",
    },
)
