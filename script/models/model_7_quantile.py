"""
model_7_quantile.py
===================
Model 7: Quantile Regression with Multiple Quantile Estimation
WARNING: RESEARCH ONLY - NOT REGULATORY COMPLIANT

CRITICAL WARNING:
This model produces DISTRIBUTIONS rather than single allocations.
It violates F.S. 393.0662 and F.A.C. 65G-4.0214 which require 
deterministic budget amounts. Suitable for research and risk analysis only.

Key features:
- Quantile regression at tau = {0.10, 0.25, 0.50, 0.75, 0.90}
- Median (tau = 0.50) as primary model for single allocation
- Complete robustness to outliers (50% breakdown point)
- Asymmetric loss function (check function)
- Natural prediction intervals from quantile spread
- Square-root transformation of costs (configurable)
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Import base class
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# SINGLE POINT OF CONTROL FOR RANDOM SEED
# ============================================================================
# Change this value to get different random splits, or keep at 42 for reproducibility
# This seed controls:
#   - Train/test split
#   - Cross-validation folds
#   - Any other random operations in the pipeline
# ============================================================================
RANDOM_SEED = 42


class Model7QuantileRegression(BaseiBudgetModel):
    """
    Model 7: Quantile Regression
    
    WARNING: REGULATORY WARNING - NOT COMPLIANT WITH F.S. 393.0662
    
    This model produces a distribution of potential allocations rather than
    a single deterministic amount. While statistically sophisticated, it cannot
    be used for production budget allocation under current Florida law.
    
    Key features:
    - Quantile regression at multiple quantiles
    - Median (tau = 0.50) as primary estimate
    - Complete outlier robustness (100% data inclusion)
    - Natural prediction intervals
    - Square-root transformation (configurable)
    - Research and validation tool only
    """
    
    def __init__(self, use_sqrt_transform: bool = False):
        """
        Initialize Model 7
        
        Args:
            use_sqrt_transform: Use sqrt transformation (True) or original dollars (False)
        """
        super().__init__(model_id=7, model_name="Quantile Regression")
        
        # ============================================================================
        # TRANSFORMATION CONTROL - Applicable to ALL models
        # ============================================================================
        # Set to True to use sqrt transformation (historical baseline)
        # Set to False to fit on original dollar scale (simpler interpretation)
        # ============================================================================
        self.use_sqrt_transform = use_sqrt_transform
        self.transformation = "sqrt" if use_sqrt_transform else "none"
        logger.info(f"Transformation: {self.transformation}")
        
        # Quantile regression specific parameters
        self.quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]  # Multiple quantiles
        self.primary_quantile = 0.50  # Median as primary estimate
        self.models = {}  # Dictionary of models, one per quantile
        
        # Regulatory compliance
        self.regulatory_compliant = "No"  # CRITICAL: Not compliant
        self.regulatory_warning = "Produces distributions, not single allocations. Violates F.S. 393.0662."
        self.deployment_status = "Research Only"
        
        # Performance tracking
        self.quantile_performance = {}  # R2 for each quantile
        self.prediction_intervals = {}  # Width of intervals
        self.quantile_spread = None  # Q90/Q10 ratio
        self.monotonicity_violations = 0  # Quantile crossing issues
        
        logger.info("="*80)
        logger.info("MODEL 7: QUANTILE REGRESSION - RESEARCH ONLY")
        logger.info("WARNING: NOT REGULATORY COMPLIANT")
        logger.info("Produces distributions, not single allocations")
        logger.info("="*80)
    
    def split_data(self, test_size: float = 0.2, random_state: int = RANDOM_SEED) -> None:
        """
        Override split_data to ensure proper train/test split
        CRITICAL: Handles boolean test_size from base class
        Uses global RANDOM_SEED as default
        
        Args:
            test_size: Proportion for test set  
            random_state: Random seed (defaults to global RANDOM_SEED)
        """
        # CRITICAL: Handle boolean test_size (base class sometimes passes True)
        if isinstance(test_size, bool):
            test_size = 0.2 if test_size else 0.0
        
        np.random.seed(random_state)
        n_records = len(self.all_records)
        n_test = int(n_records * test_size)
        
        # Shuffle indices
        indices = np.arange(n_records)
        np.random.shuffle(indices)
        
        # Split indices
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # Create train/test records
        self.test_records = [self.all_records[i] for i in test_indices]
        self.train_records = [self.all_records[i] for i in train_indices]
        
        logger.info(f"Data split: {len(self.train_records)} training, {len(self.test_records)} test")

    def run_complete_pipeline(self, 
                            fiscal_year_start: int = 2023,
                            fiscal_year_end: int = 2024,
                            perform_cv: bool = True,
                            test_size: float = 0.2,
                            n_cv_folds: int = 10) -> Dict[str, Any]:
        """
        Run complete Model 7 pipeline - let base class orchestrate, add warnings
        
        Args:
            fiscal_year_start: Start year for data
            fiscal_year_end: End year for data
            perform_cv: Whether to perform cross-validation
            test_size: Proportion for test set
            n_cv_folds: Number of CV folds
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL 7: QUANTILE REGRESSION PIPELINE")
        logger.info("WARNING: RESEARCH ONLY - NOT REGULATORY COMPLIANT")
        logger.info("="*80)
        
        # Let base class handle everything - it calls our overridden methods
        results = super().run_complete_pipeline(
            fiscal_year_start=fiscal_year_start,
            fiscal_year_end=fiscal_year_end,
            perform_cv=perform_cv,
            test_size=test_size,
            n_cv_folds=n_cv_folds
        )
        
        # Add regulatory warnings
        logger.warning("\n" + "="*80)
        logger.warning("REGULATORY COMPLIANCE ASSESSMENT")
        logger.warning("="*80)
        logger.warning(f"Status: {self.regulatory_compliant}")
        logger.warning(f"Warning: {self.regulatory_warning}")
        logger.warning("\nThis model CANNOT be used for production budget allocation.")
        logger.warning("It violates F.S. 393.0662 by producing distributions rather than")
        logger.warning("single deterministic amounts. Suitable for research only.")
        logger.warning("="*80)
        
        return results

    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix using ONLY robust features from FeatureSelection.txt
        
        Based on mutual information analysis, using consistently important features:
        - RESIDENCETYPE/Living Setting indicators (highest MI: 0.252-0.272)
        - BSum (behavioral summary, MI: 0.113-0.137)
        - LOSRI/OLEVEL (support levels, MI: 0.113-0.131)
        - Top QSI questions (Q26, Q36, Q20, Q27, etc.)
        - Age group indicators
        
        Returns:
            Tuple of (feature_matrix, feature_names) as expected by base class
        """
        logger.info(f"Preparing features from {len(records)} records...")
        
        # Extract features
        data = []
        for record in records:
            features = [
                # Living setting indicators (5 features)
                1 if record.living_setting == 'ILSL' else 0,
                1 if record.living_setting == 'RH1' else 0,
                1 if record.living_setting == 'RH2' else 0,
                1 if record.living_setting == 'RH3' else 0,
                1 if record.living_setting == 'RH4' else 0,
                
                # Age group indicators (2 features)
                1 if record.age_group == 'Age21_30' else 0,
                1 if record.age_group == 'Age31Plus' else 0,
                
                # Summary scores (2 features)
                record.bsum,
                record.fsum,
                
                # Support levels (4 features)
                record.losri,
                record.olevel,
                record.blevel,
                record.flevel,
                
                # Top QSI questions (10 features)
                record.q16, record.q18, record.q20, record.q21, record.q23,
                record.q28, record.q33, record.q34, record.q36, record.q43
            ]
            data.append(features)
        
        X = np.array(data)
        
        # Define feature names
        feature_names = [
            'LivingILSL', 'LivingRH1', 'LivingRH2', 'LivingRH3', 'LivingRH4',
            'Age21_30', 'Age31Plus',
            'BSum', 'FSum',
            'LOSRI', 'OLEVEL', 'BLEVEL', 'FLEVEL',
            'Q16', 'Q18', 'Q20', 'Q21', 'Q23', 'Q28', 'Q33', 'Q34', 'Q36', 'Q43'
        ]
        
        # Store for later use
        self.feature_names = feature_names
        
        logger.info(f"Feature matrix: {X.shape}")
        logger.info(f"Features: {len(feature_names)}")
        
        return X, feature_names
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit quantile regression models at multiple quantiles
        
        CRITICAL: Base class passes y in ORIGINAL DOLLAR SCALE
        This method must transform it internally if configured
        
        Args:
            X: Feature matrix
            y: Target variable in ORIGINAL DOLLAR SCALE (from base class)
        """
        logger.info("\n" + "="*80)
        logger.info("FITTING QUANTILE REGRESSION MODELS")
        logger.info("="*80)
        
        # CRITICAL: Transform y if needed (base class provides original dollars)
        if self.use_sqrt_transform:
            y_fit = np.sqrt(y)
            logger.info(f"Applied sqrt transformation for fitting")
        else:
            y_fit = y
            logger.info(f"Fitting on original dollar scale (no transformation)")
        
        # Fit model at each quantile (on transformed scale)
        for tau in self.quantiles:
            logger.info(f"\nFitting quantile tau = {tau:.2f}...")
            
            # Create quantile regressor
            model = QuantileRegressor(
                quantile=tau,
                alpha=0.0,  # No regularization
                solver='highs'  # Interior-point solver
            )
            
            # Fit model on transformed scale
            model.fit(X, y_fit)
            self.models[tau] = model
            
            # Calculate pseudo-R2 (based on check function)
            y_pred = model.predict(X)
            
            # Check function loss on transformed scale
            residuals = y_fit - y_pred
            check_loss = np.sum(residuals * (tau - (residuals < 0).astype(float)))
            
            # Null model (predict median for all)
            y_median = np.median(y_fit)
            null_residuals = y_fit - y_median
            null_loss = np.sum(null_residuals * (tau - (null_residuals < 0).astype(float)))
            
            pseudo_r2 = 1 - (check_loss / null_loss) if null_loss > 0 else 0
            
            self.quantile_performance[tau] = {
                'pseudo_r2': pseudo_r2,
                'check_loss': check_loss,
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_)
            }
            
            logger.info(f"  Pseudo-R2 = {pseudo_r2:.4f}")
            logger.info(f"  Check function loss = {check_loss:.2f}")
        
        # Set primary model (median)
        self.model = self.models[self.primary_quantile]
        
        logger.info("\n" + "="*80)
        logger.info(f"PRIMARY MODEL: Median Regression (tau = {self.primary_quantile})")
        logger.info("="*80)
        
        # Calculate quantile spread
        q10_loss = self.quantile_performance[0.10]['check_loss']
        q90_loss = self.quantile_performance[0.90]['check_loss']
        self.quantile_spread = q90_loss / max(q10_loss, 1e-6)
        logger.info(f"Quantile spread ratio (Q90/Q10 loss): {self.quantile_spread:.2f}")
    
    def predict(self, X: np.ndarray, quantile: Optional[float] = None) -> np.ndarray:
        """
        Generate predictions at specified quantile or primary quantile
        
        Args:
            X: Feature matrix
            quantile: Which quantile to predict (default: median)
            
        Returns:
            Predictions in original dollar scale (ALWAYS)
        """
        if quantile is None:
            quantile = self.primary_quantile
        
        if quantile not in self.models:
            raise ValueError(f"Model not fitted for quantile {quantile}")
        
        # Predict in transformed space
        y_pred = self.models[quantile].predict(X)
        
        # Back-transform to dollar scale if needed
        if self.use_sqrt_transform:
            y_pred = y_pred ** 2
        
        # Ensure non-negative
        y_pred = np.maximum(y_pred, 0)
        
        return y_pred
    
    def predict_interval(self, X: np.ndarray, lower_q: float = 0.10, 
                        upper_q: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction intervals using quantile predictions
        
        Args:
            X: Feature matrix
            lower_q: Lower quantile (default: 0.10)
            upper_q: Upper quantile (default: 0.90)
            
        Returns:
            Tuple of (median, lower_bound, upper_bound) predictions
        """
        median = self.predict(X, quantile=0.50)
        lower = self.predict(X, quantile=lower_q)
        upper = self.predict(X, quantile=upper_q)
        
        # Check for quantile crossing (monotonicity violations)
        violations = np.sum((lower > median) | (median > upper))
        if violations > 0:
            self.monotonicity_violations = violations
            logger.warning(f"Quantile crossing detected in {violations} predictions")
        
        return median, lower, upper
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, float]:
        """
        Perform k-fold cross-validation on median regression
        
        CRITICAL: Always works with ORIGINAL DOLLAR SCALE for fair comparison
        Transformation happens internally during fitting
        
        Args:
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV metrics
        """
        logger.info(f"\nPerforming {n_splits}-fold cross-validation on median regression...")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = []
        
        # Prepare features and get ORIGINAL SCALE costs
        X_cv, _ = self.prepare_features(self.train_records)
        y_cv_original = np.array([r.total_cost for r in self.train_records])
        
        # Apply transformation if needed
        if self.use_sqrt_transform:
            y_cv_fit = np.sqrt(y_cv_original)
        else:
            y_cv_fit = y_cv_original
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv), 1):
            # Split data
            X_train_fold = X_cv[train_idx]
            X_val_fold = X_cv[val_idx]
            y_train_fold = y_cv_fit[train_idx]
            y_val_original = y_cv_original[val_idx]
            
            # Train median model
            cv_model = QuantileRegressor(
                quantile=self.primary_quantile,
                alpha=0.0,
                solver='highs'
            )
            cv_model.fit(X_train_fold, y_train_fold)
            
            # Predict (in transformed space)
            y_val_pred_transformed = cv_model.predict(X_val_fold)
            
            # Back-transform predictions to original scale
            if self.use_sqrt_transform:
                y_val_pred = y_val_pred_transformed ** 2
            else:
                y_val_pred = y_val_pred_transformed
            
            y_val_pred = np.maximum(y_val_pred, 0)
            
            # Calculate R2 on ORIGINAL scale (critical for fair comparison)
            score = r2_score(y_val_original, y_val_pred)
            cv_scores.append(score)
            
            logger.info(f"  Fold {fold}: R2 = {score:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_min = np.min(cv_scores)
        cv_max = np.max(cv_scores)
        
        logger.info(f"\nCross-validation R2: {cv_mean:.4f} +/- {cv_std:.4f}")
        logger.info(f"CV Range: [{cv_min:.4f}, {cv_max:.4f}]")
        
        # CRITICAL: Store CV results as instance attributes for metrics
        self.cv_mean = cv_mean
        self.cv_std = cv_std
        self.cv_min = cv_min
        self.cv_max = cv_max
        self.cv_scores = cv_scores
        self.n_cv_folds = n_splits
        
        return {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_min': cv_min,
            'cv_max': cv_max,
            'cv_scores': cv_scores
        }
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for quantile regression
        
        Extends base class metrics with quantile-specific measures.
        """
        # Get base metrics (uses median predictions on original dollar scale)
        metrics = super().calculate_metrics()
        
        # Add CV results if available
        if hasattr(self, 'cv_mean'):
            metrics['cv_mean'] = float(self.cv_mean)
            metrics['cv_std'] = float(self.cv_std)
            metrics['cv_min'] = float(self.cv_min)
            metrics['cv_max'] = float(self.cv_max)
            metrics['n_cv_folds'] = int(self.n_cv_folds)
        
        # Add number of features
        if hasattr(self, 'feature_names') and self.feature_names:
            metrics['num_features'] = len(self.feature_names)
        
        # Add quantile-specific metrics
        if self.y_test is not None and self.X_test is not None:
            # Calculate prediction interval width
            median, lower, upper = self.predict_interval(self.X_test)
            interval_widths = upper - lower
            avg_interval_width = np.mean(interval_widths)
            
            metrics['prediction_interval_width'] = float(avg_interval_width)
            metrics['quantile_spread'] = float(self.quantile_spread) if self.quantile_spread else 0
            
            # Monotonicity percentage
            mono_pct = 100.0 * (1 - self.monotonicity_violations / len(self.y_test))
            metrics['quantile_monotonicity'] = float(mono_pct)
            
            # Performance at each quantile
            for tau in self.quantiles:
                if tau in self.quantile_performance:
                    metrics[f'quantile_{int(tau*100)}_pseudo_r2'] = float(self.quantile_performance[tau]['pseudo_r2'])
            
            # Regulatory compliance (CRITICAL)
            metrics['regulatory_compliant'] = self.regulatory_compliant
            metrics['regulatory_warning'] = self.regulatory_warning
            metrics['deployment_status'] = self.deployment_status
        
        return metrics
    
    def generate_latex_commands(self) -> None:
        """
        Generate LaTeX commands including quantile-specific metrics
        CRITICAL: Must override base class method (not a new name!)
        """
        # STEP 1: Call parent FIRST - creates files with 'w' mode
        super().generate_latex_commands()
        
        # STEP 2: Append model-specific commands using 'a' mode
        logger.info(f"Adding Model {self.model_id} specific LaTeX commands...")
        
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append to newcommands (definitions)
        with open(newcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Quantile-Specific Commands\n")
            f.write("% ============================================================================\n")
            
            # Quantile-specific R2 commands
            f.write("\\newcommand{\\ModelSevenQuantileTenRSquared}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenQuantileTwentyFiveRSquared}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenQuantileFiftyRSquared}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenQuantileSeventyFiveRSquared}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenQuantileNinetyRSquared}{\\placeholder}\n")
            
            # Prediction interval commands
            f.write("\\newcommand{\\ModelSevenPredictionIntervalWidth}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenQuantileSpread}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenQuantileMonotonicity}{\\placeholder}\n")
            
            # CRITICAL: Regulatory compliance commands
            f.write("\\newcommand{\\ModelSevenRegulatoryCompliant}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenRegulatoryWarning}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenDeploymentStatus}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenFatalFlaw}{\\placeholder}\n")
            
            # Additional commands
            f.write("\\newcommand{\\ModelSevenTransformation}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenNumFeatures}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenCVFolds}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenCVMin}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelSevenCVMax}{\\placeholder}\n")
        
        # Get metrics for values
        metrics = self.metrics if hasattr(self, 'metrics') else {}
        
        # Append to renewcommands (values)
        with open(renewcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Quantile-Specific Values\n")
            f.write("% ============================================================================\n")
            
            # Quantile R2 values
            if hasattr(self, 'quantile_performance') and self.quantile_performance:
                quantile_labels = {
                    0.10: 'Ten',
                    0.25: 'TwentyFive',
                    0.50: 'Fifty',
                    0.75: 'SeventyFive',
                    0.90: 'Ninety'
                }
                
                for q_val, q_label in quantile_labels.items():
                    if q_val in self.quantile_performance:
                        r2 = self.quantile_performance[q_val].get('pseudo_r2', 0)
                        f.write(f"\\renewcommand{{\\ModelSevenQuantile{q_label}RSquared}}{{{r2:.4f}}}\n")
                    else:
                        f.write(f"\\renewcommand{{\\ModelSevenQuantile{q_label}RSquared}}{{0.0000}}\n")
            else:
                # Defaults if not fitted
                for q_label in ['Ten', 'TwentyFive', 'Fifty', 'SeventyFive', 'Ninety']:
                    f.write(f"\\renewcommand{{\\ModelSevenQuantile{q_label}RSquared}}{{0.0000}}\n")
            
            # Prediction interval width
            if hasattr(self, 'y_test') and self.y_test is not None and hasattr(self, 'X_test'):
                try:
                    median, lower, upper = self.predict_interval(self.X_test)
                    width = np.mean(upper - lower)
                    f.write(f"\\renewcommand{{\\ModelSevenPredictionIntervalWidth}}{{{width:,.0f}}}\n")
                except:
                    f.write(f"\\renewcommand{{\\ModelSevenPredictionIntervalWidth}}{{0}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelSevenPredictionIntervalWidth}}{{0}}\n")
            
            # Quantile spread
            if hasattr(self, 'quantile_spread') and self.quantile_spread:
                f.write(f"\\renewcommand{{\\ModelSevenQuantileSpread}}{{{self.quantile_spread:.2f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelSevenQuantileSpread}}{{0.00}}\n")
            
            # Monotonicity
            if hasattr(self, 'monotonicity_violations') and hasattr(self, 'y_test') and self.y_test is not None:
                mono_pct = 100.0 * (1 - self.monotonicity_violations / len(self.y_test))
                f.write(f"\\renewcommand{{\\ModelSevenQuantileMonotonicity}}{{{mono_pct:.1f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelSevenQuantileMonotonicity}}{{100.0}}\n")
            
            # CRITICAL: Regulatory compliance (ALWAYS these values)
            f.write(f"\\renewcommand{{\\ModelSevenRegulatoryCompliant}}{{No}}\n")
            f.write(f"\\renewcommand{{\\ModelSevenRegulatoryWarning}}{{Produces distributions, not single allocations. Violates F.S. 393.0662.}}\n")
            f.write(f"\\renewcommand{{\\ModelSevenDeploymentStatus}}{{Research Only}}\n")
            f.write(f"\\renewcommand{{\\ModelSevenFatalFlaw}}{{Produces distributions not single amounts}}\n")
            
            # Transformation
            f.write(f"\\renewcommand{{\\ModelSevenTransformation}}{{{self.transformation}}}\n")
            
            # Number of features
            num_features = metrics.get('num_features', len(self.feature_names) if hasattr(self, 'feature_names') else 0)
            f.write(f"\\renewcommand{{\\ModelSevenNumFeatures}}{{{num_features}}}\n")
            
            # CV metrics
            cv_folds = metrics.get('n_cv_folds', 10)
            cv_min = metrics.get('cv_min', 0.0)
            cv_max = metrics.get('cv_max', 0.0)
            f.write(f"\\renewcommand{{\\ModelSevenCVFolds}}{{{cv_folds}}}\n")
            f.write(f"\\renewcommand{{\\ModelSevenCVMin}}{{{cv_min:.4f}}}\n")
            f.write(f"\\renewcommand{{\\ModelSevenCVMax}}{{{cv_max:.4f}}}\n")
        
        logger.info(f"Model {self.model_id} specific commands added successfully")
    
    def plot_diagnostics(self) -> None:
        """Generate quantile regression specific diagnostic plots"""
        if self.y_test is None or self.test_predictions is None:
            logger.warning("No test predictions available for plotting")
            return
        
        # Call base class diagnostics first
        super().plot_diagnostics()
        
        # Generate quantile-specific plots
        self._plot_quantile_diagnostics()
        self._plot_coefficient_comparison()
    
    def _plot_quantile_diagnostics(self) -> None:
        """Generate quantile-specific diagnostic plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model 7: Quantile Regression Diagnostics - RESEARCH ONLY', 
                     fontsize=14, fontweight='bold', color='red')
        
        # 1. Predicted vs Actual (Median)
        ax = axes[0, 0]
        ax.scatter(self.y_test, self.test_predictions, alpha=0.5, s=20)
        min_val = min(self.y_test.min(), self.test_predictions.min())
        max_val = max(self.y_test.max(), self.test_predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Cost ($)')
        ax.set_ylabel('Predicted Cost - Median ($)')
        ax.set_title('Predicted vs Actual (Median)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Fan Chart (Prediction Intervals)
        ax = axes[0, 1]
        median, lower, upper = self.predict_interval(self.X_test)
        sorted_idx = np.argsort(self.y_test)
        ax.fill_between(range(len(self.y_test)), 
                        lower[sorted_idx], 
                        upper[sorted_idx], 
                        alpha=0.3, color='blue', label='80% Interval (Q10-Q90)')
        ax.plot(median[sorted_idx], 'r-', lw=2, label='Median Prediction')
        ax.plot(self.y_test[sorted_idx], 'k--', lw=1, label='Actual')
        ax.set_xlabel('Observation (sorted by actual cost)')
        ax.set_ylabel('Cost ($)')
        ax.set_title('Fan Chart - Prediction Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Residuals vs Fitted (Median)
        ax = axes[0, 2]
        residuals = self.y_test - self.test_predictions
        ax.scatter(self.test_predictions, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Fitted Values ($)')
        ax.set_ylabel('Residuals ($)')
        ax.set_title('Residual Plot (Median)')
        ax.grid(True, alpha=0.3)
        
        # 4. Interval Width Distribution
        ax = axes[1, 0]
        interval_widths = upper - lower
        ax.hist(interval_widths, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(interval_widths), color='r', linestyle='--', 
                   lw=2, label=f'Mean: ${np.mean(interval_widths):,.0f}')
        ax.set_xlabel('Prediction Interval Width ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of 80% Prediction Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Coverage Probability
        ax = axes[1, 1]
        coverage = ((self.y_test >= lower) & (self.y_test <= upper)).astype(float)
        cost_bins = np.percentile(self.y_test, [0, 25, 50, 75, 100])
        bin_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        coverage_by_bin = []
        for i in range(len(cost_bins)-1):
            mask = (self.y_test >= cost_bins[i]) & (self.y_test < cost_bins[i+1])
            coverage_by_bin.append(coverage[mask].mean() * 100)
        ax.bar(bin_labels, coverage_by_bin, alpha=0.7, edgecolor='black')
        ax.axhline(y=80, color='r', linestyle='--', lw=2, label='Expected 80%')
        ax.set_xlabel('Cost Quartile')
        ax.set_ylabel('Coverage (%)')
        ax.set_title('Prediction Interval Coverage by Cost Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Quantile Performance Comparison
        ax = axes[1, 2]
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        pseudo_r2s = [self.quantile_performance[q]['pseudo_r2'] for q in quantiles]
        ax.bar([f'{int(q*100)}%' for q in quantiles], pseudo_r2s, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Pseudo-R2')
        ax.set_title('Performance Across Quantiles')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_file = self.output_dir / "quantile_diagnostics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Quantile diagnostics plot saved to {plot_file}")
    
    def _plot_coefficient_comparison(self) -> None:
        """Plot coefficients across quantiles"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract coefficients for each quantile
        quantile_labels = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90']
        coef_matrix = np.zeros((len(self.feature_names), len(self.quantiles)))
        
        for i, tau in enumerate(self.quantiles):
            coef_matrix[:, i] = self.quantile_performance[tau]['coefficients']
        
        # Create heatmap
        im = ax.imshow(coef_matrix, aspect='auto', cmap='RdBu_r', 
                      vmin=-np.abs(coef_matrix).max(), vmax=np.abs(coef_matrix).max())
        
        # Set ticks and labels
        ax.set_xticks(range(len(quantile_labels)))
        ax.set_xticklabels(quantile_labels)
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels(self.feature_names, fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coefficient Value', rotation=270, labelpad=20)
        
        ax.set_title('Coefficient Variation Across Quantiles', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        # Save
        plot_file = self.output_dir / "quantile_coefficients.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Coefficient comparison plot saved to {plot_file}")


def main():
    """Main execution function"""
    # SET ALL RANDOM SEEDS FOR REPRODUCIBILITY
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("\n" + "="*80)
    print("MODEL 7: QUANTILE REGRESSION EXECUTION")
    print("="*80)
    print(f"Random Seed: {RANDOM_SEED} (for reproducibility)")
    
    # ============================================================================
    # TRANSFORMATION OPTION - Easy to test both!
    # ============================================================================
    USE_SQRT = True  # Change this to False to test original dollar scale
    
    print(f"Transformation: {'sqrt' if USE_SQRT else 'none (original dollars)'}")
    print("="*80)
    
    # Initialize model with transformation option
    model = Model7QuantileRegression(use_sqrt_transform=USE_SQRT)
    
    # Run complete pipeline
    results = model.run_complete_pipeline(
        fiscal_year_start=2024,
        fiscal_year_end=2024,
        perform_cv=True,
        test_size=0.2,
        n_cv_folds=10
    )
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL 7 QUANTILE REGRESSION EXECUTION COMPLETE")
    print("="*80)
    print("\nWARNING: CRITICAL REGULATORY WARNING")
    print("="*80)
    print("Status: NOT COMPLIANT with F.S. 393.0662")
    print("Issue: Produces distributions, not single allocations")
    print("Deployment: RESEARCH ONLY - NOT for production")
    print("="*80)
    
    # Print key results
    metrics = results['metrics']
    print(f"\nMedian Regression Performance:")
    print(f"  Test R2: {metrics.get('r2_test', 0):.4f}")
    print(f"  Test RMSE: ${metrics.get('rmse_test', 0):,.0f}")
    print(f"  Test MAE: ${metrics.get('mae_test', 0):,.0f}")
    print(f"  Test MAPE: {metrics.get('mape_test', 0):.1f}%")
    
    print(f"\nQuantile Performance:")
    print(f"  Q10 Pseudo-R2: {metrics.get('quantile_10_pseudo_r2', 0):.4f}")
    print(f"  Q50 Pseudo-R2: {metrics.get('quantile_50_pseudo_r2', 0):.4f}")
    print(f"  Q90 Pseudo-R2: {metrics.get('quantile_90_pseudo_r2', 0):.4f}")
    
    print(f"\nPrediction Intervals:")
    print(f"  Average Width: ${metrics.get('prediction_interval_width', 0):,.0f}")
    print(f"  Quantile Spread: {metrics.get('quantile_spread', 0):.2f}")
    print(f"  Monotonicity: {metrics.get('quantile_monotonicity', 0):.1f}%")
    
    if 'cv_mean' in metrics:
        print(f"\nCross-Validation:")
        print(f"  Mean R2: {metrics['cv_mean']:.4f} +/- {metrics['cv_std']:.4f}")
        print(f"  Range: [{metrics['cv_min']:.4f}, {metrics['cv_max']:.4f}]")
    
    print(f"\nModel Configuration:")
    print(f"  Features: {metrics.get('num_features', 0)}")
    print(f"  Transformation: {model.transformation}")
    
    # Verify command count
    renewcommands_file = model.output_dir / f"model_{model.model_id}_renewcommands.tex"
    if renewcommands_file.exists():
        with open(renewcommands_file, 'r') as f:
            command_count = len([line for line in f if '\\renewcommand' in line])
        print(f"\nLaTeX Commands Generated: {command_count}")
        if command_count < 95:
            print(f"WARNING: Expected 95-105+ commands, got {command_count}")
        else:
            print(f"SUCCESS: Command count meets requirements")
    
    print("\n" + "="*80)
    print("INSTRUCTIONS FOR REPRODUCIBILITY:")
    print("="*80)
    print(f"To change random seed: Edit RANDOM_SEED = {RANDOM_SEED} at top of file")
    print(f"To test without sqrt: Set USE_SQRT = False in main()")
    print("="*80)
    
    return model


if __name__ == "__main__":
    model = main()