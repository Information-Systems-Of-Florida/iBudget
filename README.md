# Florida iBudget Algorithm Repository

## Overview

This repository documents the new Florida iBudget allocation algorithm developed for the Agency for Persons with Disabilities (APD). This repository contains the analysis of the current algorithm and presents proposed alternative quantitative approaches. The iBudget program is the Medicaid waiver system that provides individual budgets for developmental disability services. The algorithm plays a central role in determining resource distribution for over 36,000 enrollees across Florida.

## Current Algorithm

The current model, designated as **Model 5b**, is a multiple linear regression approach based on fiscal year 2013–2014 claims data. It incorporates variables from the **Questionnaire for Situational Information (QSI)**, as well as age and living arrangement factors. The formula applies weighted values across functional, behavioral, and physical domains, squares the sum, and apportions results according to available funding.

Key characteristics of the current model include:

* Use of 22 independent variables across QSI domains, age, and residential setting.
* Reliance on square-root transformation to stabilize variance.
* Removal of approximately 9.4% of cases as outliers to achieve statistical fit.
* Reported explanatory power of R² ≈ 0.80.

Limitations identified in both technical reviews and statute compliance include outdated data, reliance on actuarial prediction rather than person-centered planning, and exclusion of significant cases due to outlier treatment.

## Compliance Context

The passage of **House Bill 1103 (2025)** mandates a comprehensive review of the iBudget algorithm. Requirements emphasize alignment with person-centered planning, use of current expenditure data, and transparent methodology consistent with disability rights frameworks.

## Identified Weaknesses

Analysis of the current algorithm highlights several deficiencies:

* **Temporal validity issues** due to reliance on decade-old claims data.
* **Negative coefficients** in prior models implying lower funding for higher needs.
* **High outlier exclusion rates** that disproportionately affect individuals with complex needs.
* **Limited construct validity** with exclusion of disability type and certain QSI items.
* **Non-compliance** with statutory requirements for person-centered planning.

## Proposed Alternatives

To address these issues, six categories of alternative approaches are proposed:

1. **Enhanced Linear Regression**

   * Robust regression to handle outliers.
   * Regularized methods (LASSO, Ridge, Elastic Net) for variable selection and multicollinearity.

2. **Machine Learning Ensemble Models**

   * Random Forests with interpretability through feature importance.
   * Gradient Boosting with custom objectives that integrate person-centered factors.

3. **Hybrid Statistical-Clinical Models**

   * Two-stage models combining statistical prediction with clinical or person-centered adjustments.
   * Bayesian hierarchical models for multilevel data structures.

4. **Person-Centered Optimization Approaches**

   * Multi-objective optimization balancing statistical accuracy with individual goals and fairness.
   * Constrained optimization explicitly incorporating equity considerations.

5. **Modern Time-Aware Methods**

   * Dynamic regression models with time effects.
   * Longitudinal mixed-effects models tracking individual trajectories.

6. **Specialized Needs-Based Models**

   * Latent class mixture models to identify subpopulations.
   * Support vector regression for high-dimensional assessment data.

## Implementation Roadmap

The proposed framework recommends a **phased deployment strategy**:

1. Establish a robust baseline model using elastic net regression.
2. Integrate nonlinear methods such as gradient boosting.
3. Incorporate person-centered features through hybrid and optimization models.
4. Validate through cross-validation, holdout testing, and continuous monitoring.
5. Deploy with safeguards for transparency, fairness, and compliance.

## Repository Structure

```
/docs
  ├── ibudget_rule_current_algorithm.pdf
  ├── draft2_proposed_alternatives.pdf
/src
  ├── baseline_models.py
  ├── machine_learning_models.py
  ├── hybrid_models.py
  └── optimization_models.py
/tests
  └── validation_framework.ipynb
/assets
  └── ISF-icon.png
  └── ISF-background.png
  └── report_configuration.tex
/report
  └── validation_framework.ipynb
README.md
```


## Citation

* Florida Administrative Code, iBudget Rules (65G-4.0213–0218).
* Agency for Persons with Disabilities, *Analysis of the Florida iBudget Algorithm: Current Limitations and Proposed Quantitative Alternatives* (2025).



### Git Protocol

After you make your changes:

```bash
git pull
git add .
git commit -m "[description of the change]"
git push
```

To set up Git credentials:

```bash
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

To undo a staged change:

```bash
git reset
```

To discard local changes and sync with the remote:

```bash
git reset --hard HEAD
git push --force
```

### Prerequisites

* Python 3.x
* PyQt5 for GUI applications: `pip install PyQt5`


### License

This project is open-sourced under CC-BY-SA 4.0 International.

### Contact

For queries or suggestions, contact jgutierrez@isf.com or raise an issue in the repository.
