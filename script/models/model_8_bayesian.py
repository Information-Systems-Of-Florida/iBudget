"""
model_8_bayesian.py
===================
Model 8: Bayesian Linear Regression with Conjugate Priors
WARNING: RESEARCH ONLY - NOT REGULATORY COMPLIANT

Produces PROBABILITY DISTRIBUTIONS over allocations, fundamentally
violating F.S. 393.0662 which requires single deterministic amounts.

Key features:
- Bayesian Ridge Regression (conjugate Normal-Inverse-Gamma prior)
- Full posterior distributions with credible intervals
- Uses ONLY robust features from Model 5b (19 features)
- Optional sqrt transformation (test both empirically)
- Uncertainty quantification via posterior variance

CRITICAL: This model CANNOT be deployed. It's for research/comparison only.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# For Bayesian implementation - using sklearn's conjugate prior approach
from sklearn.linear_model import BayesianRidge

# Import base class - CRITICAL
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

import random  # For setting random seed


class Model8Bayesian(BaseiBudgetModel):
    """
    Model 8: Bayesian Linear Regression with Conjugate Priors
    
    WARNING - REGULATORY STATUS: NON-COMPLIANT
    This model produces probability distributions, not single values.
    CANNOT be used in production. Research purposes only.
    
    Key features:
    - Conjugate Normal-Inverse-Gamma prior
    - Full posterior distributions
    - Credible intervals for predictions
    - Uses ONLY robust features from Model 5b (19 features)
    - Optional sqrt transformation (configurable)
    
    Robust Features (19 total):
    - 5 Living Settings: ILSL, RH1, RH2, RH3, RH4 (FH as reference)
    - 2 Age Groups: Age21_30, Age31Plus (Age3_20 as reference)
    - 10 QSI Questions: Q16, Q18, Q20, Q21, Q23, Q28, Q33, Q34, Q36, Q43
    - 2 Summary Scores: BSum, FSum
    """
    
    def __init__(self, use_fy2024_only: bool = True, use_sqrt_transform: bool = False):
        """
        Initialize Model 8 Bayesian Regression
        
        Args:
            use_fy2024_only: If True, use only FY2024 data
            use_sqrt_transform: If True, use sqrt transformation (historical baseline)
                               If False, fit on original dollar scale
        """
        super().__init__(model_id=8, model_name="Bayesian-Regression")
        
        self.use_fy2024_only = use_fy2024_only
        self.fiscal_years_used = "2024" if use_fy2024_only else "2023-2024"
        
        # ============================================================================
        # TRANSFORMATION CONTROL - Test both to see which performs better!
        # ============================================================================
        # Set to True to use sqrt transformation (historical approach)
        # Set to False to fit on original dollar scale (simpler interpretation)
        # ============================================================================
        self.use_sqrt_transform = use_sqrt_transform
        self.transformation = "sqrt" if use_sqrt_transform else "none"
        
        logger.info(f"Transformation: {self.transformation}")
        
        # ============================================================================
        # REGULATORY COMPLIANCE STATUS - CRITICAL
        # ============================================================================
        self.regulatory_compliant = "No"
        self.regulatory_warning = "Produces probability distributions; F.S. 393.0662 requires single deterministic amounts"
        self.deployment_status = "Research Only"
        self.fatal_flaw = "Probability distributions not deterministic"
        self.legal_impossibility = "Cannot comply with F.S. 393.0662 - distributions vs point estimates"
        
        logger.info("WARNING - REGULATORY STATUS: NON-COMPLIANT")
        logger.info("WARNING - FOR RESEARCH PURPOSES ONLY")
        
        # Define robust features from Model 5b validation
        self.use_living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']  # FH is reference
        self.use_age_groups = ['Age21_30', 'Age31Plus']  # Age3_20 is reference
        self.robust_qsi_questions = [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]
        self.use_summary_scores = ['bsum', 'fsum']
        
        # Total features: 5 + 2 + 10 + 2 = 19 features
        self.num_parameters = 20  # 19 features + intercept
        
        # Bayesian-specific attributes
        self.model = None  # BayesianRidge model
        self.scaler = StandardScaler()
        self.alpha_ = None  # Noise precision (inverse variance)
        self.lambda_ = None  # Weights precision
        self.effective_parameters = None
        self.log_marginal_likelihood = None
        
        # Posterior distributions
        self.posterior_mean = None
        self.posterior_std = None
        self.credible_intervals = {}
        
        # Uncertainty metrics
        self.avg_credible_interval_width = None
        self.prediction_uncertainty = None
        
        # Cost estimates (Bayesian regression - HIGHEST complexity)
        # Should be MORE expensive than Model 5 (Ridge) due to:
        # - PhD-level Bayesian statistician required
        # - Complex posterior sampling infrastructure
        # - Extensive consumer education materials  
        # - Continuous model validation
        self.implementation_cost = 490000  # $490,000 (one-time)
        self.annual_operating_cost = 75000  # $75,000 per year
        self.three_year_tco = 490000 + (75000 * 3)  # $715,000 total over 3 years
        
        logger.info(f"Model 8 initialized with {self.num_parameters} parameters (19 features + intercept)")
        logger.info(f"Using robust features from Model 5b validation")
    
    def split_data(self, test_size: float = 0.2, random_state: int = RANDOM_SEED) -> None:
        """
        Override to handle boolean test_size and use global seed
        
        Args:
            test_size: Proportion for test set (or boolean, which we convert)
            random_state: Random seed (defaults to global RANDOM_SEED)
        """
        # Handle boolean test_size issue from base class
        if isinstance(test_size, bool):
            test_size = 0.2 if test_size else 0.0
        
        if not self.all_records:
            raise ValueError("No records loaded. Call load_data() first.")
        
        # Call parent with correct parameters
        super().split_data(test_size=test_size, random_state=random_state)
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features using ONLY robust features from Model 5b validation
        Total of 19 validated features
        
        Returns:
            Tuple of (feature matrix, feature names)
        """
        if not records:
            return np.array([]), []
        
        features_list = []
        feature_names = []
        
        # Build feature names first
        # 1. Living setting dummies (5 features, drop FH as reference)
        for setting in self.use_living_settings:
            feature_names.append(f'Living_{setting}')
        
        # 2. Age group dummies (2 features, drop Age3_20 as reference)
        for age_group in self.use_age_groups:
            feature_names.append(f'Age_{age_group}')
        
        # 3. Robust QSI questions (10 features)
        for q_num in self.robust_qsi_questions:
            feature_names.append(f'Q{q_num}')
        
        # 4. Summary scores (2 features)
        feature_names.append('BSum')
        feature_names.append('FSum')
        
        # Build feature matrix
        for record in records:
            row_features = []
            
            # 1. Living setting dummies (5 features)
            for setting in self.use_living_settings:
                value = 1.0 if record.living_setting == setting else 0.0
                row_features.append(value)
            
            # 2. Age group dummies (2 features)
            for age_group in self.use_age_groups:
                value = 1.0 if record.age_group == age_group else 0.0
                row_features.append(value)
            
            # 3. Robust QSI questions (10 features)
            for q_num in self.robust_qsi_questions:
                value = getattr(record, f'q{q_num}', 0)
                row_features.append(float(value))
            
            # 4. Summary scores (2 features)
            row_features.append(float(record.bsum))
            row_features.append(float(record.fsum))
            
            features_list.append(row_features)
        
        X = np.array(features_list)
        
        logger.info(f"Prepared {X.shape[0]} records with {X.shape[1]} robust features")
        
        return X, feature_names
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Bayesian Ridge regression model
        
        Args:
            X: Feature matrix
            y: Target values (on appropriate scale based on use_sqrt_transform)
        """
        logger.info("Fitting Bayesian Ridge regression model...")
        logger.info(f"Training on {'sqrt-transformed' if self.use_sqrt_transform else 'original dollar'} scale")
        
        # Standardize features for better numerical stability
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize Bayesian Ridge model
        # Uses conjugate Normal-Inverse-Gamma prior
        self.model = BayesianRidge(
            max_iter=300,  # Maximum iterations for convergence (correct sklearn parameter)
            tol=1e-3,      # Convergence tolerance
            alpha_1=1e-6,  # Gamma prior on alpha (noise precision)
            alpha_2=1e-6,  # Gamma prior on alpha
            lambda_1=1e-6, # Gamma prior on lambda (weights precision)
            lambda_2=1e-6, # Gamma prior on lambda
            compute_score=True,  # Compute log marginal likelihood
            fit_intercept=True
        )
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        # Extract Bayesian parameters
        self.alpha_ = self.model.alpha_  # Noise precision
        self.lambda_ = self.model.lambda_  # Weights precision
        
        # Posterior mean (MAP estimate)
        self.posterior_mean = self.model.coef_
        
        # Calculate posterior standard deviations
        if hasattr(self.model, 'sigma_'):
            self.posterior_std = np.sqrt(np.diag(self.model.sigma_))
        else:
            # Approximate using the precision parameter
            self.posterior_std = np.sqrt(1 / (self.lambda_ * np.ones(len(self.model.coef_))))
        
        # Calculate 95% credible intervals for each coefficient
        for i, name in enumerate(self.feature_names):
            mean = self.posterior_mean[i]
            std = self.posterior_std[i] if i < len(self.posterior_std) else 0.1
            self.credible_intervals[name] = {
                'mean': mean,
                'std': std,
                'lower_95': mean - 1.96 * std,
                'upper_95': mean + 1.96 * std
            }
        
        # Calculate effective degrees of freedom (effective number of parameters)
        # For Bayesian models, some parameters are "shrunk" towards zero
        self.effective_parameters = np.sum(self.model.lambda_ / (self.model.lambda_ + self.model.alpha_))
        
        # Log marginal likelihood (model evidence)
        if hasattr(self.model, 'scores_'):
            self.log_marginal_likelihood = self.model.scores_[-1]
        
        logger.info(f"Bayesian model fitted successfully")
        logger.info(f"Alpha (noise precision): {self.alpha_:.4f}")
        logger.info(f"Lambda (weights precision): {self.lambda_:.4f}")
        logger.info(f"Effective parameters: {self.effective_parameters:.1f} (vs {self.num_parameters} total)")
        if self.log_marginal_likelihood:
            logger.info(f"Log marginal likelihood: {self.log_marginal_likelihood:.2f}")
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Make predictions using fitted Bayesian model
        
        Args:
            X: Feature matrix
            return_std: If True, return both mean and std (for uncertainty quantification)
            
        Returns:
            Predictions on ORIGINAL DOLLAR SCALE (always)
            If return_std=True, returns (predictions, std_predictions)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict on appropriate scale
        if return_std:
            y_pred, y_std = self.model.predict(X_scaled, return_std=True)
        else:
            y_pred = self.model.predict(X_scaled)
            y_std = None
        
        # Back-transform to original dollar scale if needed
        if self.use_sqrt_transform:
            y_pred = y_pred ** 2
            if y_std is not None:
                # Transform uncertainty as well (approximate)
                y_std = 2 * np.sqrt(y_pred) * y_std
        
        # Ensure non-negative predictions
        y_pred = np.maximum(y_pred, 0)
        
        if return_std:
            return y_pred, y_std
        else:
            return y_pred
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation with Bayesian regression
        
        Args:
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = []
        
        # Get original-scale costs
        y_original = np.array([r.total_cost for r in self.train_records])
        
        # Apply transformation if needed
        if self.use_sqrt_transform:
            y_fit = np.sqrt(y_original)
        else:
            y_fit = y_original
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            # Split data
            X_cv_train = self.X_train[train_idx]
            X_cv_val = self.X_train[val_idx]
            y_cv_train = y_fit[train_idx]
            y_cv_val_original = y_original[val_idx]
            
            # Fit model on this fold
            scaler = StandardScaler()
            X_cv_train_scaled = scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = scaler.transform(X_cv_val)
            
            cv_model = BayesianRidge(max_iter=300, tol=1e-3, compute_score=False)
            cv_model.fit(X_cv_train_scaled, y_cv_train)
            
            # Predict
            y_cv_pred = cv_model.predict(X_cv_val_scaled)
            
            # Back-transform if needed
            if self.use_sqrt_transform:
                y_cv_pred = y_cv_pred ** 2
            
            # Ensure non-negative
            y_cv_pred = np.maximum(y_cv_pred, 0)
            
            # Score ALWAYS on original scale
            score = r2_score(y_cv_val_original, y_cv_pred)
            cv_scores.append(score)
            
            logger.info(f"  Fold {fold}/{n_splits}: R² = {score:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logger.info(f"Cross-validation R²: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_r2_mean': cv_mean,  # For base class compatibility
            'cv_r2_std': cv_std
        }
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Override to add Bayesian-specific metrics
        """
        # Get base metrics first
        metrics = super().calculate_metrics()
        
        # Add Bayesian-specific metrics
        if self.model is not None:
            # Calculate average credible interval width
            if self.test_predictions is not None:
                # Get predictions with uncertainty
                _, y_test_std = self.predict(self.X_test, return_std=True)
                ci_width = 2 * 1.96 * y_test_std  # 95% CI width
                self.avg_credible_interval_width = np.mean(ci_width)
            
            metrics.update({
                'alpha': self.alpha_,
                'lambda': self.lambda_,
                'effective_parameters': self.effective_parameters,
                'avg_credible_interval_width': self.avg_credible_interval_width,
                'log_marginal_likelihood': self.log_marginal_likelihood,
                'n_robust_features': len(self.feature_names),
                'transformation': self.transformation,
                # Regulatory compliance
                'regulatory_compliant': self.regulatory_compliant,
                'deployment_status': self.deployment_status
            })
        
        return metrics
    
    def plot_bayesian_diagnostics(self) -> None:
        """
        Generate Bayesian-specific diagnostic plots
        """
        logger.info("Generating Bayesian-specific diagnostic plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model 8: Bayesian Regression Diagnostics', fontsize=16, fontweight='bold')
        
        # Panel A: Posterior distributions for top 5 features
        ax = axes[0, 0]
        top_features_idx = np.argsort(np.abs(self.posterior_mean))[-5:]
        for idx in top_features_idx:
            name = self.feature_names[idx]
            mean = self.posterior_mean[idx]
            std = self.posterior_std[idx]
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            y = stats.norm.pdf(x, mean, std)
            ax.plot(x, y, label=name, linewidth=2)
        ax.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Posterior Density', fontsize=11, fontweight='bold')
        ax.set_title('A: Posterior Distributions (Top 5 Features)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        # Panel B: Coefficient credible intervals
        ax = axes[0, 1]
        sorted_idx = np.argsort(np.abs(self.posterior_mean))[-10:]
        y_pos = np.arange(len(sorted_idx))
        for i, idx in enumerate(sorted_idx):
            mean = self.posterior_mean[idx]
            std = self.posterior_std[idx]
            ax.errorbar([mean], [i], xerr=[1.96*std], fmt='o', capsize=5, capthick=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in sorted_idx], fontsize=9)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Coefficient Value (95% CI)', fontsize=11, fontweight='bold')
        ax.set_title('B: Coefficient Credible Intervals (Top 10)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Panel C: Predictive uncertainty
        ax = axes[0, 2]
        y_pred_mean, y_pred_std = self.predict(self.X_test, return_std=True)
        sorted_idx = np.argsort(self.y_test)
        ax.plot(self.y_test[sorted_idx], y_pred_mean[sorted_idx], 'o', alpha=0.5, label='Mean prediction')
        ax.fill_between(
            self.y_test[sorted_idx],
            (y_pred_mean - 1.96*y_pred_std)[sorted_idx],
            (y_pred_mean + 1.96*y_pred_std)[sorted_idx],
            alpha=0.3, label='95% Credible Interval'
        )
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('Actual Cost ($)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Cost ($)', fontsize=11, fontweight='bold')
        ax.set_title('C: Predictive Uncertainty (Test Set)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        # Panel D: Uncertainty calibration
        ax = axes[1, 0]
        # Check if actual values fall within predicted intervals
        y_pred_mean, y_pred_std = self.predict(self.X_test, return_std=True)
        z_scores = np.abs((self.y_test - y_pred_mean) / (y_pred_std + 1e-6))
        coverage = np.mean(z_scores <= 1.96)  # Should be ~0.95 for well-calibrated
        ax.hist(z_scores, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=1.96, color='red', linestyle='--', linewidth=2, label=f'95% threshold (coverage: {coverage:.2%})')
        ax.set_xlabel('|Z-score| = |Actual - Predicted| / Std', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('D: Uncertainty Calibration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        # Panel E: Effective parameters visualization
        ax = axes[1, 1]
        shrinkage = self.lambda_ / (self.lambda_ + self.alpha_)
        ax.bar(['Total\nParameters', 'Effective\nParameters'], 
               [self.num_parameters-1, self.effective_parameters],  # -1 to exclude intercept
               color=['lightblue', 'darkblue'], edgecolor='black', linewidth=2)
        ax.set_ylabel('Number of Parameters', fontsize=11, fontweight='bold')
        ax.set_title(f'E: Parameter Shrinkage (Factor: {shrinkage:.3f})', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        for i, v in enumerate([self.num_parameters-1, self.effective_parameters]):
            ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')
        
        # Panel F: Regulatory compliance warning
        ax = axes[1, 2]
        ax.axis('off')
        warning_text = (
            "WARNING - REGULATORY NON-COMPLIANCE\n\n"
            "Model 8 produces PROBABILITY\n"
            "DISTRIBUTIONS over allocations,\n"
            "not single deterministic values.\n\n"
            "Status: RESEARCH ONLY\n\n"
            "Cannot comply with:\n"
            "- F.S. 393.0662\n"
            "- F.A.C. 65G-4.0214\n"
            "- Appeals process requirements\n\n"
            "Recommendation: REJECT for production"
        )
        ax.text(0.5, 0.5, warning_text, 
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.2, pad=1),
                fontweight='bold')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'bayesian_diagnostic_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Bayesian diagnostics saved to {plot_path}")
    
    def generate_latex_commands(self) -> None:
        """
        Generate LaTeX commands including Bayesian-specific metrics
        CRITICAL: Calls super() FIRST, then appends model-specific commands
        """
        # STEP 1: Call parent method FIRST - creates files with 'w' mode
        super().generate_latex_commands()
        
        # STEP 2: Now append Model 8 specific commands using 'a' mode
        logger.info(f"Adding Model {self.model_id} specific LaTeX commands...")
        
        model_word = 'Eight'
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append to newcommands (definitions)
        with open(newcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Bayesian Specific Commands\n")
            f.write("% ============================================================================\n")
            f.write(f"\\newcommand{{\\Model{model_word}Alpha}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Lambda}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}EffectiveParams}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}AvgCredibleWidth}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}LogMarginalLikelihood}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}NRobustFeatures}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Transformation}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}NumFeatures}}{{\\placeholder}}\n")
            
            # Regulatory compliance commands (CRITICAL)
            f.write("\n% Regulatory Compliance Status\n")
            f.write(f"\\newcommand{{\\Model{model_word}RegulatoryCompliant}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}RegulatoryWarning}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}DeploymentStatus}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}FatalFlaw}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}LegalImpossibility}}{{\\placeholder}}\n")
            
            # Cost commands
            f.write("\n% Implementation and Operating Costs\n")
            f.write(f"\\newcommand{{\\Model{model_word}ImplementationCost}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}AnnualCost}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}ThreeYearTCO}}{{\\placeholder}}\n")
        
        # Append to renewcommands (values)
        with open(renewcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Bayesian Specific Values\n")
            f.write("% ============================================================================\n")
            
            if self.model is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}Alpha}}{{{self.alpha_:.4f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Lambda}}{{{self.lambda_:.4f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}NRobustFeatures}}{{{len(self.feature_names)}}}\n")
                
                if self.effective_parameters is not None:
                    f.write(f"\\renewcommand{{\\Model{model_word}EffectiveParams}}{{{self.effective_parameters:.1f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}EffectiveParams}}{{0.0}}\n")
                
                if self.avg_credible_interval_width is not None:
                    f.write(f"\\renewcommand{{\\Model{model_word}AvgCredibleWidth}}{{{self.avg_credible_interval_width:.0f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}AvgCredibleWidth}}{{0}}\n")
                
                if self.log_marginal_likelihood is not None:
                    f.write(f"\\renewcommand{{\\Model{model_word}LogMarginalLikelihood}}{{{self.log_marginal_likelihood:.1f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}LogMarginalLikelihood}}{{0.0}}\n")
            else:
                # Provide defaults if model not fitted
                f.write(f"\\renewcommand{{\\Model{model_word}Alpha}}{{0.0000}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Lambda}}{{0.0000}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}NRobustFeatures}}{{19}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}EffectiveParams}}{{0.0}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}AvgCredibleWidth}}{{0}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}LogMarginalLikelihood}}{{0.0}}\n")
            
            # Transformation
            f.write(f"\\renewcommand{{\\Model{model_word}Transformation}}{{{self.transformation}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}NumFeatures}}{{{self.num_parameters}}}\n")
            
            # Regulatory compliance values (CRITICAL)
            f.write("\n% Regulatory Compliance Status\n")
            f.write(f"\\renewcommand{{\\Model{model_word}RegulatoryCompliant}}{{{self.regulatory_compliant}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}RegulatoryWarning}}{{{self.regulatory_warning}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}DeploymentStatus}}{{{self.deployment_status}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}FatalFlaw}}{{{self.fatal_flaw}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}LegalImpossibility}}{{{self.legal_impossibility}}}\n")
            
            # Costs (formatted with commas)
            f.write("\n% Implementation and Operating Costs\n")
            f.write(f"\\renewcommand{{\\Model{model_word}ImplementationCost}}{{\\${self.implementation_cost:,}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}AnnualCost}}{{\\${self.annual_operating_cost:,}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}ThreeYearTCO}}{{\\${self.three_year_tco:,}}}\n")
        
        logger.info(f"Model {self.model_id} specific commands added successfully")


def main():
    """Main execution with comprehensive output"""
    # ============================================================================
    # SET ALL RANDOM SEEDS FOR REPRODUCIBILITY
    # ============================================================================
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("\n" + "="*80)
    print("MODEL 8: BAYESIAN LINEAR REGRESSION")
    print("WARNING - RESEARCH ONLY - NOT REGULATORY COMPLIANT")
    print("="*80)
    print(f"\nRandom Seed: {RANDOM_SEED} (for reproducibility)")
    
    # ============================================================================
    # TRANSFORMATION OPTION - Easy to test both!
    # ============================================================================
    USE_SQRT = True  # Change this to False to test original dollar scale
    
    print(f"Transformation: {'sqrt' if USE_SQRT else 'none (original dollars)'}")
    print(f"Regulatory Status: NON-COMPLIANT (distributions vs point estimates)")
    print(f"Purpose: Research and comparison only\n")
    
    # Initialize model
    model = Model8Bayesian(use_fy2024_only=True, use_sqrt_transform=USE_SQRT)
    
    # Display feature specification
    print("\n" + "="*80)
    print("ROBUST FEATURE SPECIFICATION (Model 5b Validated)")
    print("="*80)
    print(f"  Living Settings ({len(model.use_living_settings)}): {', '.join(model.use_living_settings)}")
    print(f"  Age Groups ({len(model.use_age_groups)}): {', '.join(model.use_age_groups)}")
    print(f"  QSI Questions ({len(model.robust_qsi_questions)}): {', '.join(map(str, model.robust_qsi_questions))}")
    print(f"  Summary Scores ({len(model.use_summary_scores)}): {', '.join(model.use_summary_scores)}")
    print(f"\n  Total Features: {len(model.use_living_settings) + len(model.use_age_groups) + len(model.robust_qsi_questions) + len(model.use_summary_scores)}")
    
    # Run complete pipeline
    print("\n" + "="*80)
    print("RUNNING COMPLETE PIPELINE")
    print("="*80)
    
    try:
        # Load data
        print("\n[Loading data...]")
        model.all_records = model.load_data(fiscal_year_start=2023, fiscal_year_end=2024)
        print(f"[OK] Loaded {len(model.all_records)} records from FY2023-2024")
        
        # Split data
        print("\n[Splitting data...]")
        model.split_data(test_size=0.2, random_state=RANDOM_SEED)
        print(f"[OK] Training: {len(model.train_records)} records")
        print(f"[OK] Test: {len(model.test_records)} records")
        
        # Prepare features
        print("\n[Preparing features...]")
        model.X_train, model.feature_names = model.prepare_features(model.train_records)
        model.X_test, _ = model.prepare_features(model.test_records)
        print(f"[OK] Feature matrix shape: {model.X_train.shape}")
        
        # Prepare targets with appropriate transformation
        print(f"\n[Preparing targets ({model.transformation} transformation)...]")
        y_train_original = np.array([r.total_cost for r in model.train_records])
        y_test_original = np.array([r.total_cost for r in model.test_records])
        
        if model.use_sqrt_transform:
            y_train_fit = np.sqrt(y_train_original)
            y_test_fit = np.sqrt(y_test_original)
            print("[OK] Applied sqrt transformation for fitting")
        else:
            y_train_fit = y_train_original
            y_test_fit = y_test_original
            print("[OK] Using original dollar scale for fitting")
        
        # Fit model
        print("\n[Fitting Bayesian Ridge model...]")
        model.fit(model.X_train, y_train_fit)
        print(f"[OK] Model fitted successfully")
        print(f"  - Alpha (noise precision): {model.alpha_:.4f}")
        print(f"  - Lambda (weights precision): {model.lambda_:.4f}")
        print(f"  - Effective parameters: {model.effective_parameters:.1f} / {model.num_parameters}")
        
        # Make predictions (always on original dollar scale)
        print("\n[Making predictions...]")
        model.train_predictions = model.predict(model.X_train)
        model.test_predictions = model.predict(model.X_test)
        
        # Set y to original scale for metrics (CRITICAL)
        model.y_train = y_train_original
        model.y_test = y_test_original
        print("[OK] Predictions completed (on original dollar scale)")
        
        # Calculate metrics
        print("\n[Calculating metrics...]")
        model.metrics = model.calculate_metrics()
        
        print(f"\n  TRAINING METRICS:")
        print(f"    R² = {model.metrics.get('r2_train', 0):.4f}")
        print(f"    RMSE = ${model.metrics.get('rmse_train', 0):,.0f}")
        print(f"    MAE = ${model.metrics.get('mae_train', 0):,.0f}")
        
        print(f"\n  TEST METRICS:")
        print(f"    R² = {model.metrics.get('r2_test', 0):.4f}")
        print(f"    RMSE = ${model.metrics.get('rmse_test', 0):,.0f}")
        print(f"    MAE = ${model.metrics.get('mae_test', 0):,.0f}")
        print(f"    MAPE = {model.metrics.get('mape_test', 0):.2f}%")
        
        print(f"\n  BAYESIAN METRICS:")
        print(f"    Avg Credible Interval Width: ${model.avg_credible_interval_width:,.0f}")
        print(f"    Log Marginal Likelihood: {model.log_marginal_likelihood:.2f}")
        
        # Perform cross-validation
        print("\n[Performing 10-fold cross-validation...]")
        cv_results = model.perform_cross_validation(n_splits=10)
        model.metrics.update(cv_results)
        print(f"[OK] CV R-squared = {cv_results['cv_mean']:.4f} +/- {cv_results['cv_std']:.4f}")
        
        # Calculate additional analyses
        print("\n[Calculating subgroup metrics...]")
        model.calculate_subgroup_metrics()
        print("[OK] Subgroup metrics calculated")
        
        print("\n[Calculating variance metrics...]")
        model.calculate_variance_metrics()
        print("[OK] Variance metrics calculated")
        
        print("\n[Calculating population scenarios...]")
        model.calculate_population_scenarios()
        print("[OK] Population scenarios calculated")
        
        # Generate plots
        print("\n[Generating diagnostic plots...]")
        model.plot_diagnostics()
        print("[OK] Standard diagnostic plots generated")
        
        print("\n[Generating Bayesian-specific plots...]")
        model.plot_bayesian_diagnostics()
        print("[OK] Bayesian diagnostic plots generated")
        
        # Save results
        print("\n[Saving results...]")
        model.save_results()
        print("[OK] Results saved to models/model_8/")
        
        # Generate LaTeX commands
        print("\n[Generating LaTeX commands...]")
        model.generate_latex_commands()
        print("[OK] LaTeX commands generated")
        
        # Verify command count
        print("\n[Verifying LaTeX command generation...]")
        renewcommands_file = model.output_dir / f"model_{model.model_id}_renewcommands.tex"
        if renewcommands_file.exists():
            with open(renewcommands_file, 'r') as f:
                command_count = sum(1 for line in f if '\\renewcommand' in line)
            print(f"[OK] Generated {command_count} LaTeX commands")
            if command_count >= 80:
                print("  [PASS] Command count meets minimum requirement (80+)")
            else:
                print(f"  [WARN] Command count below minimum (expected 80+, got {command_count})")
        
        print("\n" + "="*80)
        print("[SUCCESS] MODEL 8 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n[REMINDER] This model is RESEARCH ONLY")
        print(f"[REMINDER] NOT regulatory compliant - produces distributions not point estimates")
        print(f"\n[TIP] To change random seed, edit RANDOM_SEED = {RANDOM_SEED} at top of file")
        print(f"[TIP] To test different transformation, edit USE_SQRT = {USE_SQRT} in main()\n")
        
        return model
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # DON'T use logging.basicConfig() - let base_model handle logging
    # This ensures ALL logs go to both console AND file
    
    model = main()