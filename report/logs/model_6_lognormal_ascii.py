"""
model_6_lognormal.py
====================
Model 6: Log-Normal GLM with Flexible Transformation
Uses Duan's smearing estimator for retransformation bias correction
No outlier removal - uses all available data

ENHANCEMENTS (Critical Addendum v4.0):
- Rule 4: Single point random seed control
- Rule 7: Flexible transformation (log-sqrt vs log only)
- Rule 8: Proper logging configuration (no basicConfig)
- Rule 5: Complete LaTeX command generation with super() pattern
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# CRITICAL: Import random for seed control
import random

# Import base class
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# SINGLE POINT OF CONTROL FOR RANDOM SEED (Rule 4)
# ============================================================================
# Change this value to get different random splits, or keep at 42 for reproducibility
# This seed controls:
#   - Train/test split
#   - Cross-validation folds
#   - Any other random operations in the pipeline
# ============================================================================
RANDOM_SEED = 42


class Model6LogNormal(BaseiBudgetModel):
    """
    Model 6: Log-Normal GLM with Flexible Transformation
    
    Key features:
    - No outlier removal (uses all data)
    - Flexible transformation: log(sqrt(Y)) OR log(Y)
    - Normal distribution on log scale (log-normal on original scale)
    - Duan's smearing estimator for retransformation
    - Multiplicative effects interpretation
    - Built-in heteroscedasticity handling
    """
    
    def __init__(self, use_sqrt_transform: bool = False):
        """
        Initialize Model 6
        
        Args:
            use_sqrt_transform: Use log(sqrt(Y)) if True, log(Y) if False
        """
        super().__init__(model_id=6, model_name="Log-Normal-GLM")
        
        # ============================================================================
        # TRANSFORMATION CONTROL - Rule 7
        # ============================================================================
        # Set to True to use log(sqrt(Y)) transformation (traditional approach)
        # Set to False to use log(Y) directly (simpler interpretation)
        # ============================================================================
        self.use_sqrt_transform = use_sqrt_transform
        self.transformation = "log-sqrt" if use_sqrt_transform else "log"
        logger.info(f"Transformation mode: {self.transformation}")
        
        # Log-Normal specific attributes
        self.ols_model = None  # OLS on log-transformed target
        self.smearing_factor = None  # Duan's smearing factor
        self.sigma_log = None  # Standard deviation on log scale
        self.r2_log_scale = None  # R^2 on log scale
        self.skewness_original = None  # Skewness of original residuals
        self.skewness_log = None  # Skewness of log-scale residuals
        self.heteroscedasticity_pval = None  # Breusch-Pagan test p-value
        self.aic = None
        self.bic = None
        self.coefficients = {}
        self.num_parameters = 0
        
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
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare robust feature set (22 features total)
        Uses ONLY validated features from Model 5b analysis
        
        Features:
        - 5 Living Settings (FH as reference)
        - 2 Age Groups (Age3_20 as reference)
        - 10 QSI Questions (highest MI scores)
        - 2 Summary Scores (BSum, FSum)
        - 3 Primary Diagnosis indicators
        
        Args:
            records: List of ConsumerRecord objects
            
        Returns:
            Feature matrix and feature names
        """
        feature_list = []
        
        for record in records:
            features = []
            
            # 1. Living Setting (5 dummy variables, FH as reference)
            living = record.living_setting if record.living_setting else 'FH'
            features.append(1 if living == 'ILSL' else 0)
            features.append(1 if living == 'RH1' else 0)
            features.append(1 if living == 'RH2' else 0)
            features.append(1 if living == 'RH3' else 0)
            features.append(1 if living == 'RH4' else 0)
            
            # 2. Age Group (2 dummy variables, Age3_20 as reference)
            age_group = record.age_group if record.age_group else 'Age3_20'
            features.append(1 if age_group == 'Age21_30' else 0)
            features.append(1 if age_group == 'Age31Plus' else 0)
            
            # 3. QSI Questions (10 robust features)
            for q_num in [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]:
                q_val = getattr(record, f'q{q_num}', 0)
                if isinstance(q_val, bool):
                    q_val = 1 if q_val else 0
                features.append(float(q_val) if q_val is not None else 0)
            
            # 4. Summary Scores (2 features)
            features.append(float(record.bsum) if record.bsum is not None else 0)
            features.append(float(record.fsum) if record.fsum is not None else 0)
            
            # 5. Primary Disability (3 indicators)
            primary_diag = record.primary_diagnosis if record.primary_diagnosis else ''
            features.append(1 if 'intellectual' in primary_diag.lower() else 0)
            features.append(1 if 'autism' in primary_diag.lower() else 0)
            features.append(1 if 'cerebral' in primary_diag.lower() else 0)
            
            feature_list.append(features)
        
        # Generate feature names
        if not self.feature_names:
            feature_names = [
                'living_ILSL', 'living_RH1', 'living_RH2', 'living_RH3', 'living_RH4',
                'age_21_30', 'age_31_plus',
                'q16', 'q18', 'q20', 'q21', 'q23', 'q28', 'q33', 'q34', 'q36', 'q43',
                'bsum', 'fsum',
                'diag_intellectual', 'diag_autism', 'diag_cerebral'
            ]
            self.feature_names = feature_names
            self.num_parameters = len(feature_names) + 1
            logger.info(f"Prepared feature structure with {len(feature_names)} features")
        
        return np.array(feature_list), self.feature_names
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit Log-Normal GLM using OLS on log-transformed target
        Supports both log(sqrt(Y)) and log(Y) transformations
        
        Args:
            X_train: Training feature matrix
            y_train: Training target values (costs)
        """
        logger.info(f"Fitting Log-Normal GLM with {self.transformation} transformation...")
        
        # Transform target based on mode
        if self.use_sqrt_transform:
            # log(sqrt(Y)) = 0.5 * log(Y)
            y_sqrt = np.sqrt(y_train + 1e-10)
            y_transformed = np.log(y_sqrt)
        else:
            # log(Y) directly
            y_transformed = np.log(y_train + 1e-10)
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X_train)
        
        try:
            # Fit OLS on log-transformed target
            self.ols_model = sm.OLS(y_transformed, X_with_const).fit()
            self.model = self.ols_model  # Store for base class compatibility
            
            # Extract model metrics
            self.r2_log_scale = self.ols_model.rsquared
            self.sigma_log = np.sqrt(self.ols_model.scale)
            self.aic = self.ols_model.aic
            self.bic = self.ols_model.bic
            
            # Calculate Duan's smearing factor for retransformation
            residuals_log = self.ols_model.resid
            self.smearing_factor = np.mean(np.exp(residuals_log))
            
            # Calculate skewness metrics
            y_pred_log = self.ols_model.predict(X_with_const)
            residuals_original = y_train - np.exp(y_pred_log) * self.smearing_factor
            self.skewness_original = stats.skew(residuals_original)
            self.skewness_log = stats.skew(residuals_log)
            
            # Breusch-Pagan test for heteroscedasticity (on log scale)
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_test = het_breuschpagan(residuals_log, X_with_const)
            self.heteroscedasticity_pval = bp_test[1]
            
            # Extract coefficients with multiplicative effects
            params = self.ols_model.params
            pvalues = self.ols_model.pvalues
            
            for i, (param, pval) in enumerate(zip(params, pvalues)):
                if i == 0:
                    name = 'const'
                else:
                    name = self.feature_names[i-1]
                
                # Calculate multiplicative effect: (exp(beta) - 1) * 100%
                mult_effect = (np.exp(param) - 1) * 100
                
                self.coefficients[name] = {
                    'coefficient': float(param),
                    'p_value': float(pval),
                    'multiplicative_effect': float(mult_effect)
                }
            
            logger.info(f"Model fitted successfully")
            logger.info(f"  R^2 (log scale): {self.r2_log_scale:.4f}")
            logger.info(f"  Sigma (log): {self.sigma_log:.4f}")
            logger.info(f"  Smearing factor: {self.smearing_factor:.4f}")
            logger.info(f"  Skewness reduction: {((self.skewness_original - self.skewness_log) / abs(self.skewness_original) * 100):.1f}%")
            
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with proper back-transformation
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted costs (original scale)
        """
        if self.ols_model is None:
            raise ValueError("Model not fitted yet")
        
        # Add constant
        X_with_const = sm.add_constant(X)
        
        # Predict on log-transformed scale
        y_pred_log = self.ols_model.predict(X_with_const)
        
        # Back-transform with Duan's smearing estimator
        if self.use_sqrt_transform:
            # log(sqrt(Y)) -> sqrt(Y) -> Y
            # sqrt(Y) = exp(log_pred) * smearing_factor
            # Y = [exp(log_pred) * smearing_factor]^2
            sqrt_pred = np.exp(y_pred_log) * self.smearing_factor
            predictions = sqrt_pred ** 2
        else:
            # log(Y) -> Y
            # Y = exp(log_pred) * smearing_factor
            predictions = np.exp(y_pred_log) * self.smearing_factor
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Override to add Log-Normal specific metrics"""
        # Calculate base metrics
        metrics = super().calculate_metrics()
        
        # Add Log-Normal specific metrics
        if self.ols_model is not None:
            metrics.update({
                'r2_log_scale': self.r2_log_scale,
                'sigma_log': self.sigma_log,
                'smearing_factor': self.smearing_factor,
                'skewness_original': self.skewness_original,
                'skewness_log': self.skewness_log,
                'skewness_reduction_pct': ((self.skewness_original - self.skewness_log) / 
                                          abs(self.skewness_original) * 100) if self.skewness_original != 0 else 0,
                'heteroscedasticity_pval': self.heteroscedasticity_pval,
                'aic': self.aic,
                'bic': self.bic,
                'num_parameters': self.num_parameters,
                'transformation': self.transformation
            })
        
        return metrics
    
    def generate_latex_commands(self) -> None:
        """
        Override to add Log-Normal specific LaTeX commands
        CRITICAL (Rule 5): Must call super() FIRST, then append with 'a' mode
        """
        # STEP 1: Call parent FIRST - creates files with 'w' mode (fresh start)
        super().generate_latex_commands()
        
        # STEP 2: Now append model-specific commands using 'a' mode
        logger.info(f"Adding Model {self.model_id} specific LaTeX commands...")
        
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append to newcommands (definitions)
        with open(newcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Specific Commands\n")
            f.write("% ============================================================================\n")
            f.write("\\newcommand{\\ModelSixRSquaredLogScale}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSigma}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSmearingFactor}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSkewnessReduction}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixHeteroscedasticityTest}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSmearingBias}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixAIC}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixBIC}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixNRobustFeatures}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixTransformation}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixDispersion}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixLinkFunction}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixDistribution}{\\WarningRunPipeline}\n")
        
        # Append to renewcommands (values)
        with open(renewcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Specific Values\n")
            f.write("% ============================================================================\n")
            
            if self.ols_model is not None:
                # Core log-normal metrics
                f.write(f"\\renewcommand{{\\ModelSixRSquaredLogScale}}{{{self.r2_log_scale:.4f}}}\n")
                f.write(f"\\renewcommand{{\\ModelSixSigma}}{{{self.sigma_log:.4f}}}\n")
                f.write(f"\\renewcommand{{\\ModelSixSmearingFactor}}{{{self.smearing_factor:.4f}}}\n")
                
                # Skewness reduction
                skew_reduction = ((self.skewness_original - self.skewness_log) / 
                                 abs(self.skewness_original) * 100) if self.skewness_original != 0 else 0
                f.write(f"\\renewcommand{{\\ModelSixSkewnessReduction}}{{{skew_reduction:.1f}}}\n")
                
                # Heteroscedasticity test
                f.write(f"\\renewcommand{{\\ModelSixHeteroscedasticityTest}}{{{self.heteroscedasticity_pval:.4f}}}\n")
                
                # Retransformation bias as percent
                smearing_bias = (self.smearing_factor - 1) * 100
                f.write(f"\\renewcommand{{\\ModelSixSmearingBias}}{{{smearing_bias:.2f}}}\n")
                
                # Information criteria
                f.write(f"\\renewcommand{{\\ModelSixAIC}}{{{self.aic:,.0f}}}\n")
                f.write(f"\\renewcommand{{\\ModelSixBIC}}{{{self.bic:,.0f}}}\n")
                
                # Transformation type
                f.write(f"\\renewcommand{{\\ModelSixTransformation}}{{{self.transformation}}}\n")
                
                # GLM-specific (for compatibility with GLM chapter structure)
                f.write(f"\\renewcommand{{\\ModelSixDispersion}}{{{self.sigma_log**2:.4f}}}\n")
                f.write("\\renewcommand{\\ModelSixLinkFunction}{log}\n")
                f.write("\\renewcommand{\\ModelSixDistribution}{Gaussian}\n")
            else:
                # Provide defaults if model not fitted
                f.write("\\renewcommand{\\ModelSixRSquaredLogScale}{0.0000}\n")
                f.write("\\renewcommand{\\ModelSixSigma}{0.0000}\n")
                f.write("\\renewcommand{\\ModelSixSmearingFactor}{1.0000}\n")
                f.write("\\renewcommand{\\ModelSixSkewnessReduction}{0.0}\n")
                f.write("\\renewcommand{\\ModelSixHeteroscedasticityTest}{1.0000}\n")
                f.write("\\renewcommand{\\ModelSixSmearingBias}{0.00}\n")
                f.write("\\renewcommand{\\ModelSixAIC}{0}\n")
                f.write("\\renewcommand{\\ModelSixBIC}{0}\n")
                f.write("\\renewcommand{\\ModelSixNRobustFeatures}{0}\n")
                f.write("\\renewcommand{\\ModelSixTransformation}{none}\n")
                f.write("\\renewcommand{\\ModelSixDispersion}{0.0000}\n")
                f.write("\\renewcommand{\\ModelSixLinkFunction}{none}\n")
                f.write("\\renewcommand{\\ModelSixDistribution}{none}\n")
        
        logger.info(f"Model {self.model_id} specific commands added successfully")
    
    def save_results(self) -> None:
        """Override to save Log-Normal specific results"""
        # Save base results
        super().save_results()
        
        # Save coefficients
        coef_file = self.output_dir / "coefficients.json"
        with open(coef_file, 'w') as f:
            json.dump(self.coefficients, f, indent=2, default=str)
        
        # Save model summary
        if self.ols_model is not None:
            summary_file = self.output_dir / "model_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(str(self.ols_model.summary()))
        
        # Save significant coefficients (p < 0.05) with multiplicative effects
        significant_coef = {
            name: data for name, data in self.coefficients.items()
            if data['p_value'] < 0.05
        }
        sig_file = self.output_dir / "significant_coefficients.json"
        with open(sig_file, 'w') as f:
            json.dump(significant_coef, f, indent=2, default=str)
        
        # Save multiplicative effects table
        effects_file = self.output_dir / "multiplicative_effects.txt"
        with open(effects_file, 'w') as f:
            f.write("Multiplicative Effects (Percentage Change in Cost)\n")
            f.write("="*60 + "\n\n")
            sorted_coef = sorted(
                [(k, v) for k, v in self.coefficients.items() if v['p_value'] < 0.05],
                key=lambda x: abs(x[1]['multiplicative_effect']),
                reverse=True
            )
            for name, data in sorted_coef:
                f.write(f"{name:30s}: {data['multiplicative_effect']:+7.2f}% "
                       f"(p={data['p_value']:.4f})\n")
    
    def plot_diagnostics(self) -> None:
        """Generate comprehensive diagnostic plots"""
        if self.ols_model is None or self.test_predictions is None:
            logger.warning("Model not fitted or predictions not available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Get log-scale predictions and residuals
        X_test_const = sm.add_constant(self.X_test)
        y_pred_log = self.ols_model.predict(X_test_const)
        
        # Calculate residuals on log scale
        if self.use_sqrt_transform:
            y_test_log = np.log(np.sqrt(self.y_test + 1e-10))
        else:
            y_test_log = np.log(self.y_test + 1e-10)
        residuals_log = y_test_log - y_pred_log
        
        # 1. Predicted vs Actual (original scale)
        ax = axes[0, 0]
        ax.scatter(self.y_test, self.test_predictions, alpha=0.5, s=10)
        max_val = max(self.y_test.max(), self.test_predictions.max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        ax.set_xlabel('Actual Cost ($)')
        ax.set_ylabel('Predicted Cost ($)')
        ax.set_title('Predicted vs Actual (Original Scale)')
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals on log scale
        ax = axes[0, 1]
        ax.scatter(y_pred_log, residuals_log, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Fitted Values (log scale)')
        ax.set_ylabel('Residuals (log scale)')
        ax.set_title('Residual Plot (Log Scale)')
        ax.grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        ax = axes[0, 2]
        stats.probplot(residuals_log, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Log-Scale Residuals)')
        ax.grid(True, alpha=0.3)
        
        # 4. Residual distribution
        ax = axes[1, 0]
        ax.hist(residuals_log, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('Residuals (log scale)')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 5. Retransformation bias check
        ax = axes[1, 1]
        naive_pred = np.exp(y_pred_log)
        if self.use_sqrt_transform:
            naive_pred = naive_pred ** 2
        corrected_pred = self.test_predictions
        ax.scatter(naive_pred, corrected_pred, alpha=0.5, s=10)
        max_val = max(naive_pred.max(), corrected_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        ax.set_xlabel('Naive exp() Prediction')
        ax.set_ylabel('Smearing-Corrected Prediction')
        ax.set_title('Smearing Correction Impact')
        ax.grid(True, alpha=0.3)
        
        # 6. Performance by cost quartile
        ax = axes[1, 2]
        quartiles = pd.qcut(self.y_test, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_errors = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            mask = quartiles == q
            errors = np.abs(self.y_test[mask] - self.test_predictions[mask])
            quartile_errors.append(errors)
        
        ax.boxplot(quartile_errors, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        ax.set_xlabel('Cost Quartile')
        ax.set_ylabel('Absolute Error ($)')
        ax.set_title('Error Distribution by Quartile')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "diagnostic_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plots saved to {plot_file}")


def main():
    """
    Main execution function
    CRITICAL (Rule 8): NO logging.basicConfig() call here!
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    # ============================================================================
    # SET ALL RANDOM SEEDS FOR REPRODUCIBILITY (Rule 4)
    # ============================================================================
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("\n" + "="*80)
    print("MODEL 6: LOG-NORMAL GLM")
    print("="*80)
    print(f"\n[*] Random Seed: {RANDOM_SEED} (for reproducibility)")
    
    # ============================================================================
    # TRANSFORMATION OPTION - Easy to test both! (Rule 7)
    # ============================================================================
    USE_SQRT = True  # Change to False to test log(Y) directly
    
    print(f"[*] Transformation: {'log(sqrt(Y))' if USE_SQRT else 'log(Y)'}")
    print("    (Change USE_SQRT in main() to test alternative)")
    
    # Initialize model with transformation option
    model = Model6LogNormal(use_sqrt_transform=USE_SQRT)
    
    # Run complete pipeline
    print("\n[*] Running complete pipeline...")
    results = model.run_complete_pipeline(
        fiscal_year_start=2024,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    print("\n" + "="*80)
    print("MODEL 6 EXECUTION COMPLETE")
    print("="*80)
    
    print("\n[Configuration]")
    print(f"  - Transformation: {model.transformation}")
    print(f"  - Data Utilization: 100% (no outlier removal)")
    print(f"  - Features: {len(model.feature_names)}")
    print(f"  - Random Seed: {RANDOM_SEED}")
    
    print("\n[Key Metrics]")
    print(f"  - Test R2: {model.metrics.get('r2_test', 0):.4f}")
    print(f"  - R2 (log scale): {model.r2_log_scale:.4f}")
    print(f"  - RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    print(f"  - CV R2: {model.metrics.get('cv_r2_mean', 0):.4f} +/- {model.metrics.get('cv_r2_std', 0):.4f}")
    
    print("\n[Log-Normal Specific]")
    print(f"  - Smearing Factor: {model.smearing_factor:.4f}")
    print(f"  - Smearing Bias: {(model.smearing_factor - 1) * 100:+.2f}%")
    print(f"  - Sigma (log): {model.sigma_log:.4f}")
    print(f"  - Skewness Reduction: {((model.skewness_original - model.skewness_log) / abs(model.skewness_original) * 100):.1f}%")
    print(f"  - Heteroscedasticity p-value: {model.heteroscedasticity_pval:.4f}")
    
    print("\n[Model Selection Criteria]")
    print(f"  - AIC: {model.aic:,.0f}")
    print(f"  - BIC: {model.bic:,.0f}")
    
    print("\n[Key Advantages]")
    print("  - No outlier removal (100% data retention)")
    print("  - Natural handling of right-skewed costs")
    print("  - Multiplicative effects interpretation")
    print("  - Built-in heteroscedasticity handling")
    print("  - Duan's smearing reduces retransformation bias")
    
    print("\n[Files Generated]")
    for file in sorted(model.output_dir.glob("*")):
        print(f"  - {file.name}")
    
    # ============================================================================
    # COMMAND COUNT VERIFICATION (Critical Addendum)
    # ============================================================================
    renewcommands_file = model.output_dir / f"model_{model.model_id}_renewcommands.tex"
    if renewcommands_file.exists():
        with open(renewcommands_file, 'r') as f:
            command_count = sum(1 for line in f if '\\renewcommand' in line)
        print(f"\n[LaTeX Commands] {command_count} commands generated")
        if command_count >= 85:
            print("   + Exceeds minimum requirement of 80+ commands")
        else:
            print(f"   ! Below target of 85+ commands (missing {85 - command_count})")
    
    print("\n[Tips]")
    print(f"  - To change random seed: edit RANDOM_SEED = {RANDOM_SEED} at top of file")
    print(f"  - To test log(Y) directly: set USE_SQRT = False in main()")
    print("  - Compare both transformations empirically to choose the best!")
    
    print("="*80 + "\n")
    
    return model


if __name__ == "__main__":
    # CRITICAL (Rule 8): Do NOT use logging.basicConfig()
    # Let base_model handle all logging configuration
    model = main()
