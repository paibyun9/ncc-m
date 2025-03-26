# Neolithic Cultural Complex (NCC) Spatial Analysis

This repository contains the Python code and dataset used for the spatial statistical analysis of Neolithic megalithic sites, as described in the *Nature* submission titled "Spatial Analysis of Neolithic Cultural Complex in Europe". The analysis applies Ripley's K-function and Moran's I to assess clustering patterns and spatial autocorrelation of Neolithic sites across 17 European countries (including Turkey).

## Repository Contents
- `NCC-M.py`: The main Python script for spatial statistical analysis.
- `data.csv`: The dataset containing 94 Neolithic sites from 17 European countries (including Turkey), with columns: `No`, `Site Name`, `Structure Type`, `Country`, `Nearest Town`, `Latitude`, `Longitude`, `UTM Zone`, `UTM Easting`, and `Start_Year`. See the **Dataset Description** section below for details.
- `requirements.txt`: A file listing the required Python library versions.

## Dataset Description
The `data.csv` file contains data for 94 Neolithic sites across 17 European countries (Bulgaria, Czech Republic, Denmark, France, Italy, Belgium, England, Germany, Greece, Ireland, Malta, Portugal, Scotland, Spain, Sweden, Turkey, Ukraine), dated to the Early Neolithic expansion into Europe (6600–4000 BC ± 200 years). The dataset was systematically filtered to exclude non-European countries (Egypt, India, South Korea, Sudan, Syria, Israel) and to ensure exactly 44 sites for K-A (Neolithic sites, stone circles) and 50 sites for K-B (Dolmens, Menhirs). The columns in the dataset are as follows:
- `No`: Unique identifier for each site.
- `Site Name`: Name of the archaeological site.
- `Structure Type`: Type of structure (e.g., neolithicsite, stonecircle, dolmen, menhir).
- `Country`: Country where the site is located.
- `Nearest Town`: Nearest modern town or city to the site.
- `Latitude`: Latitude of the site in degrees (filtered to 35–70°N).
- `Longitude`: Longitude of the site in degrees (filtered to -10–30°E).
- `UTM Zone`: UTM zone of the site.
- `UTM Easting`: UTM Easting coordinate in meters.
- `Start_Year`: Estimated start year of the site (BC).

The dataset was preprocessed by converting `Latitude`, `Longitude`, and `UTM Easting` to numeric values, removing rows with missing values, and filtering sites to ensure they fall within the European range (35–70°N, -10–30°E). For more details, refer to Appendix 1 of the manuscript.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/paibyun9/ncc-m.git
   cd ncc-m


2. Ensure Python 3.11.11 (or 3.11.9, 3.12.8) is installed.

3. Install the required libraries:

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
The dataset (data.csv) must match the structure described in Appendix 1.

Contact
For questions or access issues, please contact the corresponding author, Byun, Dae Jung, at paibyun9@example.com.
