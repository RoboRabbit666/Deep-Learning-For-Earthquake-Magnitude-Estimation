# Earthquake Magnitude Estimation using Deep Learning

## Overview
This repository contains the implementation of deep learning approaches for single-station earthquake magnitude estimation, building upon and enhancing the MagNet architecture.

## Repository Structure
```bash
earthquake-magnitude-estimation/
├── LICENSE
├── README.md
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_preprocessing.ipynb        # Data preparation and preprocessing
│   ├── 02_baseline_magnet.ipynb          # Original MagNet implementation
│   ├── 03_deeper_magnet.ipynb            # Enhanced deeper architecture
│   ├── 04_50_random_runs.ipynb           # Data splitting analysis
│   └── 05_cross_regional_analysis.ipynb   # Cross-regional studies
├── data/
│   └── README.md                         # Data access instructions (no actual data)
└── requirements.txt            # Project dependencies

## Key Features
- Enhanced MagNet architecture with deeper layers and residual connections
- Comprehensive analysis of data-splitting strategies
- Cross-regional adaptation studies
- Uncertainty quantification in magnitude estimates

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Required packages listed in requirements.txt

## Usage
The notebooks should be run in numerical order. Each notebook is self-contained with detailed documentation and comments.

## Data
This project uses the following datasets:
- STEAD (STanford EArthquake Dataset)
- INSTANCE (Italian Seismic Dataset)
- SCEDC (Southern California Earthquake Data Center)

Note: Due to size constraints, this repository does not include data. Instructions for data access are provided in data/README.md

## Setup
```bash
pip install -r requirements.txt
