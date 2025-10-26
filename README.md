## EV Charging Patterns – Business Analytics Project

This repository contains an R-based analysis of EV charging behavior, exploring descriptive statistics, visualization, and predictive modeling (Linear/Lasso Regression, Random Forest, XGBoost) on `ev_charging_patterns.csv`.

### Background & Objectives
- **Background**: EV adoption is accelerating, increasing demand on charging infrastructure. Understanding consumption drivers helps operators plan capacity and optimize pricing and placement.
- **Objectives**:
  - Analyze charging behavior across user types, charger types, time of day, and environmental factors.
  - Model and predict session energy consumption (`energy_kwh`).
  - Compare model families and examine learning/stability behavior.
  - Quantify uncertainty (prediction intervals) and translate it into planning metrics (coverage, FP/FN, Winkler score).
  - Provide decision-support views for energy allocation.

### Project Structure
- `analysis-14-5.R`: main analysis script
- `ev_charging_patterns.csv`: dataset (expected in the project root)
- `install-packages.R`: helper to install required R packages
- `report.Rmd`: R Markdown report to render HTML summary
- `render.R`: convenience script to render the report
- `figures/`: auto-saved figures
- Figures (`*.png`) generated during EDA/modeling

### Dataset
EV charging session records with key fields:
- **Energy/Power/Cost**: Energy Consumed (kWh), Charging Rate (kW), Charging Cost (USD)
- **Temporal**: Charging Start/End Time, derived Time of Day, Day of Week
- **Vehicle/Usage**: Battery Capacity (kWh), State of Charge (start/end, %), Distance since last charge (km), Vehicle Age (years), Vehicle Model
- **Context**: Temperature (°C), Charger Type (Level 1/2/DC Fast), User Type (e.g., Commuter, Casual Driver, Long-Distance Traveler), Location

Data is cleaned (factor encoding, NA handling, sanity checks) and filtered to respect physical constraints (e.g., `energy_kwh <= Battery.Capacity.kWh`, `State.of.Charge.Start <= 100`).

### Requirements
- R (4.1+ recommended)
- Packages: lubridate, ggplot2, gridExtra, moments, corrplot, caret, glmnet, randomForest, xgboost, keras, car, dplyr, tibble, purrr, here, patchwork, viridis

Install them with:

```r
source("install-packages.R")
```

Notes:
- `keras` is optional for this project’s current script sections, but installed for completeness.
- `patchwork` is used if available; otherwise the script falls back to `gridExtra`.

### Running the analysis
1) Ensure `ev_charging_patterns.csv` is in the repository root.
2) Open R/RStudio and run:

```r
source("analysis-14-5.R")
```
### Rendering the HTML report

```r
source("install-packages.R")
source("render.R")
```

The report saves figures into `figures/` and produces an HTML file next to `report.Rmd`.


The script will:
- Clean and prepare data
- Run EDA and correlation analysis
- Train/evaluate models (Linear, Lasso, RF, XGBoost)
- Produce figures and summary metrics

### Methodology (high-level)
- **EDA**: Histograms, boxplots by segment (User Type, Time of Day, Day of Week), charger usage, temperature effects, scatterplot matrix, correlation heatmap.
- **Statistics**: Chi-square tests for categorical associations; ANOVA for mean differences across groups.
- **Preprocessing**: Factorization, scaling (z-score, robust), optional skewness-based transforms, Box–Cox for positive variables.
- **Modeling**: Linear Regression, Lasso/Elastic Net (CV for λ/α), Random Forest (CV for mtry), XGBoost (sampled grid search with early stopping).
- **Diagnostics**: Learning curves and stability (RMSE across random splits), feature importance across models.
- **Uncertainty & Decisions**: Conformal-style prediction intervals (based on residual quantiles), Winkler score, coverage and FP/FN analysis by segment; decision plot overlaying intervals.

### Key Questions addressed
- Which factors most influence session energy consumption?
- How does consumption vary by user type, charger type, and time of day?
- How does temperature relate to energy needs across segments?
- Which model family performs best by RMSE/MAE/R² for this dataset?
- What buffer should operators plan to achieve target coverage levels?

### Results & Reproducibility Notes
- The script prints model metrics and produces plots into `figures/` (e.g., learning curves, stability, feature importance, intervals).
- Results are sensitive to train/validation/test splits; seeds are fixed for comparability.

### Repository tree (abridged)
```
business-analytics-g/
  analysis-14-5.R
  ev_charging_patterns.csv
  install-packages.R
  report.Rmd
  render.R
  figures/
  LICENSE
  DESCRIPTION
  README.md
```

### Reproducibility
The script sets a fixed seed and uses only relative paths. If you prefer a project root locator, install the `here` package which the script will use automatically when available.

### Limitations & Future Work
- Evaluate time-aware or grouped cross-validation if temporal/leakage risks exist.
- Enrich feature set (charger occupancy, pricing, user history, route/weather context).
- Calibrate uncertainty estimates and consider quantile regression or NGBoost.
- Explore SHAP/accumulated profiles for model interpretability beyond global importances.


### License
Released under the MIT License (see `LICENSE`).


