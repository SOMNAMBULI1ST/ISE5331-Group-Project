# ISE5331 – Energy Consumption Regression (EV Painting Workshop)

This repository contains the code and data used for the final group project of ISE5331.  
We build a multiple linear regression model to describe the relationship between:

- **Energy Consumption** (target)
- **Current**
- **Load**

using IoT data from an EV factory painting workshop.

## Files

- `Energy consumption.xlsx` – Sampled IoT dataset from the painting workshop.
- `energy_regression.py` – Main analysis script:
  - Loads the dataset from Excel
  - Fits a multiple linear regression model (OLS)
  - Prints regression summary and prediction for a given point
  - Generates visualization figures:
    - `fig_energy_actual_vs_pred.png`
    - `fig_current_vs_energy.png`
    - `fig_load_vs_energy.png`

## Requirements

Python ≥ 3.8 and the following packages:

```bash
pip install numpy pandas statsmodels matplotlib openpyxl
