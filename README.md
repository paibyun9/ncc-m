# Neolithic Cultural Complex (NCC) Spatial Analysis

This repository contains the Python code and dataset used for the spatial statistical analysis of Neolithic megalithic sites, as described in the *Nature* submission titled "[Your Paper Title]". The analysis applies Ripley's K-function and Moran's I to assess clustering patterns and spatial autocorrelation of Neolithic sites across 17 European countries (including Turkey).

## Repository Contents
- `NCC-M.py`: The main Python script for spatial statistical analysis.
- `data.csv`: The dataset containing 94 Neolithic sites, with columns: `No`, `Site Name`, `Structure Type`, `Country`, `Nearest Town`, `Latitude`, `Longitude`, `UTM Zone`, `UTM Easting`, and `Start_Year`.
- `requirements.txt`: A file listing the required Python library versions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/paibyun9/ncc-m.git
   cd ncc-m
