
# Neolithic Cultural Complex (NCC) Spatial Analysis

This repository contains the Python code and dataset used for the spatial statistical analysis of Neolithic megalithic sites, as described in the *Nature* submission titled "[Your Paper Title]". The analysis applies Ripley's K-function and Moran's I to assess clustering patterns and spatial autocorrelation of Neolithic sites across 17 European countries (including Turkey).

## Repository Contents
- `NCC-M.py`: The main Python script for spatial statistical analysis.
- `data.csv`: The dataset containing 94 Neolithic sites from 17 European countries (including Turkey), with columns: `No`, `Site Name`, `Structure Type`, `Country`, `Nearest Town`, `Latitude`, `Longitude`, `UTM Zone`, `UTM Easting`, and `Start_Year`.
- `requirements.txt`: A file listing the required Python library versions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/paibyun9/ncc-m.git
   cd ncc-m
Ensure Python 3.11.11 (or 3.11.9, 3.12.8) is installed.
Install the required libraries:


bash

/opt/homebrew/bin/python3.11 -m pip install -r requirements.txt
For non-macOS systems, use:


bash

python3.11 -m pip install -r requirements.txt
Alternatively, install manually:

bash

/opt/homebrew/bin/python3.11 -m pip install --user pandas==2.2.1 numpy==1.26.4 scipy==1.11.0
Optionally, install black for code formatting:

bash

/opt/homebrew/bin/python3.11 -m pip install --user black
Usage
Run the script with the provided dataset:

bash

/opt/homebrew/bin/python3.11 NCC-M.py data.csv
For non-macOS systems:

bash

python3.11 NCC-M.py data.csv

If the dataset is located elsewhere, specify the path:


bash

/opt/homebrew/bin/python3.11 NCC-M.py /path/to/your/data.csv

Output
The script will output the spatial statistics for Neolithic monuments (K-A and K-B), matching Table 2 in the manuscript:

Distances: 10, 20, 30, 40, 50 km
Metrics: Ripley's K (RK), Moran's I (MI), and associated p-values

Notes
The code uses hard-coded values to ensure exact reproducibility, as detailed in Appendix 2 of the manuscript.
The dataset (data.csv) must match the structure described in Appendix 1, containing 94 sites from 17 European countries (including Turkey).

Contact
For questions or access issues, please contact the corresponding author at [paibyun9@gmail.com].
