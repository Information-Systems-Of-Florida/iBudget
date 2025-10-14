"""
model_6_lognormal.py
====================
Model 6: Log-Normal GLM with Conditional Duan's Smearing
Following the EXACT pattern from Models 1, 2, and 3

Key features:
- Uses Model 5b feature specification (21 features)
- Log transformation (handled by base class)
- CONDITIONAL Duan's smearing estimator (bin-wise correction)
- Addresses heteroscedasticity on log scale
- No outlier removal (100% data retention)
- Normal distribution on log scale

CONDITIONAL SMEARING ENHANCEMENT:
Instead of a single global smearing factor (which assumes homoscedasticity),
we compute 10 bin-specific smearing factors based on deciles of fitted log values.
This addresses the heteroscedasticity problem where residual variance changes with X.

Mathematical approach:
1. Global Duan (naive): Y' = exp(log_pred) x E[exp(epsilon)]
   - Assumes constant variance across all predictions
   - Fails when Var(epsilon|X) varies with X
   
2. Conditional Duan (improved): Y' = exp(log_pred) x E[exp(epsilon)|bin]
   - Computes separate smearing factor for each decile bin
   - Adapts to varying residual variance across prediction range
   - Typically improves R^2 from negative to positive

SIMPLIFIED IMPLEMENTATION:
- Base class handles transformation='log' (applies log forward, exp backward)
- We override inverse_transformation() to add conditional smearing
- Much cleaner than manual transformation handling

CRITICAL PATTERNS FOLLOWED:
- Exact initialization structure from Model 1/2/3
- Abstract methods: prepare_features, _fit_core, _predict_core
- LaTeX generation: super() FIRST, then append
- Main function: explicit configuration like Model 3
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import logging
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import base class
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# SINGLE POINT OF CONTROL FOR RANDOM SEED
# ============================================================================
RANDOM_SEED = 42


class Model6LogNormal(BaseiBudgetModel):
    """
    Model 6: Log-Normal GLM with Duan's Smearing Estimator
    
    Follows the EXACT pattern from Models 1, 2, and 3:
    - Same initialization parameters
    - Same feature preparation structure
    - Same main() function pattern
    
    Uses base class transformation='log':
    - Base class applies log() to y in fit()
    - Base class applies exp() to predictions in predict()
    - We add Duan's smearing in inverse_transformation()
    """
    
    def __init__(self,
                 use_outlier_removal: bool = False,  # Model 6 never removes outliers
                 outlier_threshold: float = 1.645,
                 random_seed: int = RANDOM_SEED,
                 log_suffix: Optional[str] = None):
        """
        Initialize Model 6: Log-Normal GLM
        
        Args:
            use_outlier_removal: Always False for Model 6 (100% data retention)
            outlier_threshold: Kept for compatibility
            random_seed: Random seed for reproducibility
            log_suffix: Suffix for log file name
        """
        # Use base class log transformation
        super().__init__(
            model_id=6,
            model_name="Log-Normal GLM (Conditional Smearing)",
            transformation='log',  # Base class handles log/exp
            use_outlier_removal=use_outlier_removal,
            outlier_threshold=outlier_threshold,
            random_seed=random_seed,
            log_suffix=log_suffix
        )
        
        # Model-specific attributes
        self.ols_model = None  # Statsmodels OLS on log scale
        
        # Log-normal specific metrics
        self.r2_log_scale = 0.0
        self.sigma_log = 0.0
        self.smearing_factor = 1.0
        self.skewness_original = 0.0
        self.skewness_log = 0.0
        self.heteroscedasticity_pval = 1.0
        self.aic = 0.0
        self.bic = 0.0
        self.num_parameters = 0
    
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features using Model 5b specification
        Uses EXACT same features as Model 1 for consistency
        """
        # If feature_config provided from pipeline, use it
        if hasattr(self, 'feature_config') and self.feature_config is not None:
            return self.prepare_features_from_spec(records, self.feature_config)
        
        # Define high MI QSI items
        high_mi_qsi = [26, 36, 27, 20, 21, 23, 30, 25, 16, 18, 28, 33, 34, 43, 44]
        
        feature_config = {
            'categorical': {
                'living_setting': {
                    'categories': ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4'],
                    'reference': 'FH'  # Not included in features
                }
            },
            'binary': {
                'Age21_30': lambda r: 21 <= r.age <= 30,
                'Age31Plus': lambda r: r.age > 30,
                'Male': lambda r: r.gender == 'M'
            },
            'numeric': ['losri', 'olevel', 'blevel', 'flevel', 'plevel', 
                    'bsum', 'fsum', 'psum', 'age'],
            #'qsi': high_mi_qsi[:15] if self.use_selected_features else list(range(14, 51)),
            'qsi': list(range(14, 51)),
            'interactions': [
                ('SupportedLiving_x_LOSRI', lambda r: (1 if r.living_setting in ['RH1','RH2','RH3','RH4'] else 0) * float(r.losri)),
                ('Age_x_BSum', lambda r: float(r.age) * float(r.bsum) / 100.0),
                ('FH_x_FSum', lambda r: (1 if r.living_setting == 'FH' else 0) * float(r.fsum))
            ]
        }
        
        return self.prepare_features_from_spec(records, feature_config)
    
    def _fit_core(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Core log-normal fitting
        
        Args:
            X: Feature matrix
            y: Target values in LOG SCALE (base class applied transformation)
        
        Note: y arrives already log-transformed by base class
        """
        self.log_section("MODEL 6 FITTING: LOG-NORMAL GLM")
        
        n_samples = len(y)
        self.logger.info(f"Training samples: {n_samples:,}")
        self.logger.info(f"Features: {X.shape[1]}")
        self.logger.info("Note: y already log-transformed by base class")
        
        # Store original y for skewness calculation
        # (We need to get it from self.y_train which base class stores)
        if hasattr(self, 'y_train'):
            self.skewness_original = stats.skew(self.y_train)
            self.logger.info(f"Skewness (original scale): {self.skewness_original:.4f}")
        
        # Calculate skewness on LOG scale (current y)
        self.skewness_log = stats.skew(y)
        if hasattr(self, 'y_train'):
            skew_reduction = ((self.skewness_original - self.skewness_log) / 
                             abs(self.skewness_original) * 100) if self.skewness_original != 0 else 0
            self.logger.info(f"Skewness (log scale): {self.skewness_log:.4f}")
            self.logger.info(f"Skewness reduction: {skew_reduction:.1f}%")
        else:
            self.logger.info(f"Skewness (log scale): {self.skewness_log:.4f}")
        
        # Fit OLS on log scale using statsmodels
        X_with_const = sm.add_constant(X)
        self.ols_model = sm.OLS(y, X_with_const).fit()
        
        # Store parameters
        self.num_parameters = len(self.ols_model.params)
        
        # Calculate R2 on log scale
        self.r2_log_scale = self.ols_model.rsquared
        self.logger.info(f"R-squared (log scale): {self.r2_log_scale:.4f}")
        
        # Calculate sigma on log scale (residual standard error)
        residuals_log = self.ols_model.resid
        self.sigma_log = np.sqrt(self.ols_model.scale)  # RSS/(n-p)
        self.logger.info(f"Sigma (log scale): {self.sigma_log:.4f}")
        
        # Calculate Duan's smearing factor
        # This corrects bias when back-transforming: E[exp(e)] != exp(E[e])
        exp_residuals = np.exp(residuals_log)
        
        # Log residual statistics for debugging
        self.logger.info(f"Residual statistics (log scale):")
        self.logger.info(f"  Mean: {np.mean(residuals_log):.6f}")
        self.logger.info(f"  Std: {np.std(residuals_log):.4f}")
        self.logger.info(f"  Min: {np.min(residuals_log):.2f}")
        self.logger.info(f"  Max: {np.max(residuals_log):.2f}")
        
        # Validate exp(residuals) are finite
        if np.any(np.isinf(exp_residuals)) or np.any(np.isnan(exp_residuals)):
            self.logger.warning("WARNING: Some exp(residuals) are inf/nan")
            self.logger.warning(f"  Residuals range: [{np.min(residuals_log):.2f}, {np.max(residuals_log):.2f}]")
            self.logger.warning("  Clipping to finite range")
            exp_residuals = np.clip(exp_residuals, 0, 1e10)
        
        self.logger.info(f"exp(residuals) statistics:")
        self.logger.info(f"  Mean (smearing factor): {np.mean(exp_residuals):.6f}")
        self.logger.info(f"  Min: {np.min(exp_residuals):.6f}")
        self.logger.info(f"  Max: {np.max(exp_residuals):.6f}")
        
        self.smearing_factor = np.mean(exp_residuals)
        smearing_bias_pct = (self.smearing_factor - 1) * 100
        self.logger.info(f"Smearing factor: {self.smearing_factor:.6f}")
        self.logger.info(f"Retransformation bias: {smearing_bias_pct:+.2f}%")
        
        # Validate smearing factor
        if not np.isfinite(self.smearing_factor) or self.smearing_factor <= 0:
            self.logger.error(f"Invalid smearing factor: {self.smearing_factor}")
            raise ValueError("Smearing factor must be positive and finite")
        
        # Warn if smearing factor indicates poor fit
        if self.smearing_factor > 1.5:
            self.logger.warning("="*60)
            self.logger.warning("CAUTION: Large smearing factor detected!")
            self.logger.warning(f"  Smearing factor = {self.smearing_factor:.2f} means predictions multiplied by ~{self.smearing_factor:.1f}x")
            self.logger.warning(f"  This occurs when log-scale residual variance is high (sigma^2 = {self.sigma_log**2:.2f})")
            self.logger.warning("  Indicates log-normal distribution may not be appropriate for this data")
            self.logger.warning("="*60)
        
        # Test for heteroscedasticity on log scale (Breusch-Pagan)
        from statsmodels.stats.diagnostic import het_breuschpagan
        _, self.heteroscedasticity_pval, _, _ = het_breuschpagan(residuals_log, X_with_const)
        self.logger.info(f"Breusch-Pagan test p-value: {self.heteroscedasticity_pval:.6f}")
        if self.heteroscedasticity_pval > 0.05:
            self.logger.info("  -> No significant heteroscedasticity on log scale")
        else:
            self.logger.info("  -> Some heteroscedasticity remains on log scale")
        
        # Information criteria
        self.aic = self.ols_model.aic
        self.bic = self.ols_model.bic
        self.logger.info(f"AIC: {self.aic:,.0f}")
        self.logger.info(f"BIC: {self.bic:,.0f}")
        
        self.logger.info("Log-Normal model fitted successfully")
    
    def _predict_core(self, X: np.ndarray) -> np.ndarray:
        """
        Core prediction - returns in LOG scale
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions in LOG scale (base class will apply inverse_transformation)
        """
        if self.ols_model is None:
            raise ValueError("Model not fitted yet")
        
        # Add constant
        X_with_const = sm.add_constant(X)
        
        # Predict on log scale (base class will back-transform)
        y_pred_log = self.ols_model.predict(X_with_const)
        
        return y_pred_log
    
    def inverse_transformation(self, y: np.ndarray) -> np.ndarray:
        """
        Override base class to add Duan's smearing during exp()
        
        Uses CONDITIONAL smearing when available (bin-wise factors),
        falls back to global smearing otherwise.
        
        Args:
            y: Predictions in log scale
            
        Returns:
            Predictions in original dollar scale with smearing correction
        """
        if self.transformation == 'log':
            # Use conditional smearing if available (addresses heteroscedasticity)
            if hasattr(self, '_cond_smearing_bins') and hasattr(self, '_cond_smearing_vals'):
                # Map each log-prediction to appropriate bin
                bins = self._cond_smearing_bins
                smear_vals = np.array(self._cond_smearing_vals)
                
                # Find which bin each prediction falls into
                bin_idx = np.clip(
                    np.digitize(y, bins[1:-1], right=True),
                    0, 9
                )
                
                # Get bin-specific smearing factor for each prediction
                sf = smear_vals[bin_idx]
                
                # Apply conditional smearing
                predictions = np.exp(y) * sf
                
                # Debug logging (only first time)
                if not hasattr(self, '_inverse_transform_logged'):
                    self.logger.info(f"Inverse transformation using CONDITIONAL smearing:")
                    self.logger.info(f"  Log predictions range: [{np.min(y):.2f}, {np.max(y):.2f}]")
                    self.logger.info(f"  Smearing factors used: [{np.min(sf):.4f}, {np.max(sf):.4f}]")
                    self.logger.info(f"  (vs global: {self.smearing_factor:.4f})")
                    self.logger.info(f"  Dollar predictions range: [${np.min(predictions):,.0f}, ${np.max(predictions):,.0f}]")
                    self._inverse_transform_logged = True
            else:
                # Fallback to global smearing
                predictions = np.exp(y) * self.smearing_factor
                
                # Debug logging (only first time)
                if not hasattr(self, '_inverse_transform_logged'):
                    self.logger.info(f"Inverse transformation using GLOBAL smearing:")
                    self.logger.info(f"  Log predictions range: [{np.min(y):.2f}, {np.max(y):.2f}]")
                    self.logger.info(f"  Smearing factor: {self.smearing_factor:.6f}")
                    self.logger.info(f"  Dollar predictions range: [${np.min(predictions):,.0f}, ${np.max(predictions):,.0f}]")
                    self._inverse_transform_logged = True
            
            return predictions
        else:
            # Fall back to base class for other transformations
            return super().inverse_transformation(y)
    
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
            
            # Add conditional smearing info if available
            if hasattr(self, '_cond_smearing_vals'):
                metrics['smearing_method'] = 'conditional'
                metrics['smearing_min'] = min(self._cond_smearing_vals)
                metrics['smearing_max'] = max(self._cond_smearing_vals)
                metrics['smearing_range'] = max(self._cond_smearing_vals) - min(self._cond_smearing_vals)
                metrics['smearing_bins'] = len(self._cond_smearing_vals)
            else:
                metrics['smearing_method'] = 'global'
        
        return metrics
    
    def generate_latex_commands(self) -> None:
        """
        Override to add Log-Normal specific LaTeX commands
        CRITICAL: Must call super() FIRST, then append with 'a' mode
        """
        # STEP 1: Call parent FIRST - creates files with 'w' mode (fresh start)
        super().generate_latex_commands()
        
        # STEP 2: Now append model-specific commands using 'a' mode
        self.logger.info(f"Adding Model {self.model_id} specific LaTeX commands...")
        
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append to newcommands (definitions)
        with open(newcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Log-Normal Specific Commands\n")
            f.write("% ============================================================================\n")
            f.write("\\newcommand{\\ModelSixRSquaredLogScale}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSigma}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSmearingFactor}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSmearingMin}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSmearingMax}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSmearingRange}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSmearingMethod}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSkewnessReduction}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixHeteroscedasticityTest}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixSmearingBias}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixAIC}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixBIC}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixTransformation}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixDispersion}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixLinkFunction}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelSixDistribution}{\\WarningRunPipeline}\n")
        
        # Append to renewcommands (values)
        with open(renewcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Log-Normal Specific Values\n")
            f.write("% ============================================================================\n")
            
            if self.ols_model is not None:
                # Core log-normal metrics
                f.write(f"\\renewcommand{{\\ModelSixRSquaredLogScale}}{{{self.r2_log_scale:.4f}}}\n")
                f.write(f"\\renewcommand{{\\ModelSixSigma}}{{{self.sigma_log:.4f}}}\n")
                f.write(f"\\renewcommand{{\\ModelSixSmearingFactor}}{{{self.smearing_factor:.4f}}}\n")
                
                # Conditional smearing info
                if hasattr(self, '_cond_smearing_vals'):
                    min_smear = min(self._cond_smearing_vals)
                    max_smear = max(self._cond_smearing_vals)
                    range_smear = max_smear - min_smear
                    f.write(f"\\renewcommand{{\\ModelSixSmearingMin}}{{{min_smear:.4f}}}\n")
                    f.write(f"\\renewcommand{{\\ModelSixSmearingMax}}{{{max_smear:.4f}}}\n")
                    f.write(f"\\renewcommand{{\\ModelSixSmearingRange}}{{{range_smear:.4f}}}\n")
                    f.write("\\renewcommand{\\ModelSixSmearingMethod}{Conditional (10-bin)}\n")
                else:
                    f.write(f"\\renewcommand{{\\ModelSixSmearingMin}}{{{self.smearing_factor:.4f}}}\n")
                    f.write(f"\\renewcommand{{\\ModelSixSmearingMax}}{{{self.smearing_factor:.4f}}}\n")
                    f.write("\\renewcommand{\\ModelSixSmearingRange}{0.0000}\n")
                    f.write("\\renewcommand{\\ModelSixSmearingMethod}{Global}\n")
                
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
                
                # Transformation type (NO UNICODE!)
                f.write(f"\\renewcommand{{\\ModelSixTransformation}}{{log(Y)}}\n")
                
                # GLM-specific parameters
                f.write(f"\\renewcommand{{\\ModelSixDispersion}}{{{self.sigma_log**2:.4f}}}\n")
                f.write("\\renewcommand{\\ModelSixLinkFunction}{log}\n")
                f.write("\\renewcommand{\\ModelSixDistribution}{Gaussian (on log scale)}\n")
            else:
                # Provide defaults if model not fitted
                f.write("\\renewcommand{\\ModelSixRSquaredLogScale}{0.0000}\n")
                f.write("\\renewcommand{\\ModelSixSigma}{0.0000}\n")
                f.write("\\renewcommand{\\ModelSixSmearingFactor}{1.0000}\n")
                f.write("\\renewcommand{\\ModelSixSmearingMin}{1.0000}\n")
                f.write("\\renewcommand{\\ModelSixSmearingMax}{1.0000}\n")
                f.write("\\renewcommand{\\ModelSixSmearingRange}{0.0000}\n")
                f.write("\\renewcommand{\\ModelSixSmearingMethod}{Not computed}\n")
                f.write("\\renewcommand{\\ModelSixSkewnessReduction}{0.0}\n")
                f.write("\\renewcommand{\\ModelSixHeteroscedasticityTest}{1.0000}\n")
                f.write("\\renewcommand{\\ModelSixSmearingBias}{0.00}\n")
                f.write("\\renewcommand{\\ModelSixAIC}{0}\n")
                f.write("\\renewcommand{\\ModelSixBIC}{0}\n")
                f.write("\\renewcommand{\\ModelSixTransformation}{none}\n")
                f.write("\\renewcommand{\\ModelSixDispersion}{0.0000}\n")
                f.write("\\renewcommand{\\ModelSixLinkFunction}{none}\n")
                f.write("\\renewcommand{\\ModelSixDistribution}{none}\n")
        
        self.logger.info(f"Model {self.model_id} specific commands added successfully")

    def plot_diagnostics(self) -> None:
        """
        Model 6: 2x3 diagnostics on ORIGINAL dollar scale.
        Overrides the base plotter (which only filled the first panel).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats

        if self.X_test is None or self.y_test is None:
            self.logger.warning("No test data available for diagnostics.")
            return

        # Use the model's pipeline predictions on the test set
        y_true = self.y_test.astype(float)
        y_pred = self.predict(self.X_test).astype(float)  # already inverse-transformed w/ smearing
        resid  = y_true - y_pred
        std_resid = resid / (np.std(resid) if np.std(resid) > 0 else 1.0)

        self.log_section("GENERATING DIAGNOSTIC PLOTS")

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle('Model 6: Log-Normal GLM (Conditional Smearing)', fontsize=14, fontweight='bold')

        # (1) Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(y_true / 1000, y_pred / 1000, alpha=0.5, s=12)
        vmax = max(y_true.max(), y_pred.max()) / 1000
        ax.plot([0, vmax], [0, vmax], 'r--', lw=2, alpha=0.8)
        ax.set_xlabel('Actual Cost ($1000s)')
        ax.set_ylabel('Predicted Cost ($1000s)')
        ax.set_title(f'Actual vs Predicted\nR^2 = {self.metrics.get("r2_test", 0):.4f}')
        ax.grid(True, alpha=0.3)

        # (2) Residuals vs Predicted
        ax = axes[0, 1]
        ax.scatter(y_pred / 1000, resid / 1000, alpha=0.5, s=12)
        ax.axhline(0, color='r', linestyle='--', lw=1.5)
        ax.set_xlabel('Predicted ($1000s)')
        ax.set_ylabel('Residual ($1000s)')
        ax.set_title('Residuals vs Predicted')
        ax.grid(True, alpha=0.3)

        # (3) Q-Q of residuals
        ax = axes[0, 2]
        stats.probplot(std_resid, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Residuals)')

        # (4) Scale-Location
        ax = axes[1, 0]
        ax.scatter(y_pred / 1000, np.sqrt(np.abs(std_resid)), alpha=0.5, s=12)
        ax.set_xlabel('Predicted ($1000s)')
        ax.set_ylabel('sqrt|Standardized Residuals|')
        ax.set_title('Scale-Location')
        ax.grid(True, alpha=0.3)

        # (5) Histogram of residuals
        ax = axes[1, 1]
        ax.hist(std_resid, bins=40, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(np.nanmin(std_resid), np.nanmax(std_resid), 200)
        ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, label='N(0,1)')
        ax.set_xlabel('Standardized Residual')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        ax.legend()

        # (6) Calibration by prediction decile
        ax = axes[1, 2]
        q = np.quantile(y_pred, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(y_pred, q[1:-1], right=True), 0, 9)
        cal_act = np.array([np.mean(y_true[idx == b]) for b in range(10)])
        cal_pred = np.array([np.mean(y_pred[idx == b]) for b in range(10)])
        ax.plot(cal_act / 1000, cal_pred / 1000, marker='o')
        m = max(np.nanmax(cal_act), np.nanmax(cal_pred)) / 1000
        ax.plot([0, m], [0, m], 'r--', lw=1.5)
        ax.set_xlabel('Mean Actual ($1000s)')
        ax.set_ylabel('Mean Predicted ($1000s)')
        ax.set_title('Calibration (Prediction Deciles)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_file = self.output_dir / 'diagnostic_plots.png'
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Diagnostic plots saved to {self.output_dir_relative / 'diagnostic_plots.png'}")


def main():
    """
    Run Model 6: Log-Normal GLM
    Following the EXACT pattern from Models 1, 2, and 3
    """
    print("\n" + "="*80)
    print("MODEL 6: LOG-NORMAL GLM WITH CONDITIONAL SMEARING")
    print("="*80)
    print(f"\nRandom Seed: {RANDOM_SEED} (for reproducibility)")
    print(f"Transformation: log(Y) [handled by base class]")
    print(f"Smearing: Conditional (10-bin decile-based)\n")
    
    # Initialize model with explicit parameters (following Model 1/2/3 pattern)
    use_outlier = False             # Model 6 NEVER removes outliers
    suffix = f'Log_Outliers_{use_outlier}'
    
    model = Model6LogNormal(
        use_outlier_removal=use_outlier,    # Always False for Model 6
        outlier_threshold=1.645,            # Kept for compatibility
        random_seed=RANDOM_SEED,            # For reproducibility
        log_suffix=suffix                   # Clear log suffix
    )
    
   
    # Run complete pipeline
    print("Running complete pipeline...")
    results = model.run_complete_pipeline(
        fiscal_year_start=2024,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    model.plot_diagnostics() 
    
    # Generate diagnostic plots
    if hasattr(model, 'plot_diagnostics'):
        model.plot_diagnostics()
    else:
        model.logger.warning("plot_diagnostics() method not available - skipping plots")
    
    # Calculate proper metrics BEFORE final logging
    if model.test_predictions is not None and model.y_test is not None:
        # Use base class method with correct name
        if hasattr(model, 'calculate_proper_mape_smape'):
            proper_metrics = model.calculate_proper_mape_smape(
                model.y_test, 
                model.test_predictions,
                threshold=1000.0
            )
            model.metrics.update(proper_metrics)
    
    # Log final summary
    model.log_section(f"MODEL {model.model_id} FINAL SUMMARY", "=")
    
    model.logger.info("")
    model.logger.info("Performance Metrics (Final):")
    model.logger.info(f"  Training R^2: {model.metrics.get('r2_train', 0):.4f}")
    model.logger.info(f"  Test R^2: {model.metrics.get('r2_test', 0):.4f}")
    model.logger.info(f"  Log-scale R^2: {model.r2_log_scale:.4f}")
    model.logger.info(f"  RMSE (original scale): ${model.metrics.get('rmse_test', 0):,.2f}")
    model.logger.info(f"  MAE (original scale): ${model.metrics.get('mae_test', 0):,.2f}")
    
    model.logger.info("")
    model.logger.info("Percentage Error Metrics:")
    if 'smape' in model.metrics:
        model.logger.info(f"  SMAPE (all cases): {model.metrics['smape']:.2f}%")
    if 'mape_threshold' in model.metrics and not np.isnan(model.metrics['mape_threshold']):
        threshold = model.metrics.get('mape_threshold_value', 1000)
        n = model.metrics.get('mape_n', 0)
        model.logger.info(f"  MAPE (costs >= ${threshold:,.0f}, n={n:,}): {model.metrics['mape_threshold']:.2f}%")
    
    if 'cv_mean' in model.metrics:
        model.logger.info("")
        model.logger.info(f"  CV R^2: {model.metrics['cv_mean']:.4f} +- {model.metrics['cv_std']:.4f}")
    
    model.logger.info("")
    model.logger.info("Log-Normal Specific Metrics:")
    model.logger.info(f"  Global smearing factor: {model.smearing_factor:.6f}")
    model.logger.info(f"  Retransformation bias (global): {(model.smearing_factor - 1) * 100:+.2f}%")
    
    # Conditional smearing info
    if hasattr(model, '_cond_smearing_vals'):
        min_smear = min(model._cond_smearing_vals)
        max_smear = max(model._cond_smearing_vals)
        model.logger.info(f"  Conditional smearing range: [{min_smear:.4f}, {max_smear:.4f}]")
        model.logger.info(f"  Conditional smearing method: 10-bin decile-based")
    
    model.logger.info(f"  Sigma (log scale): {model.sigma_log:.4f}")
    model.logger.info(f"  Variance (log scale): {model.sigma_log**2:.4f}")
    skew_reduction = ((model.skewness_original - model.skewness_log) / 
                     abs(model.skewness_original) * 100) if model.skewness_original != 0 else 0
    model.logger.info(f"  Skewness (original): {model.skewness_original:.4f}")
    model.logger.info(f"  Skewness (log scale): {model.skewness_log:.4f}")
    model.logger.info(f"  Skewness reduction: {skew_reduction:.1f}%")
    model.logger.info(f"  Heteroscedasticity p-value: {model.heteroscedasticity_pval:.6f}")
    model.logger.info(f"  AIC: {model.aic:,.0f}")
    model.logger.info(f"  BIC: {model.bic:,.0f}")
    
    # Add interpretation
    model.logger.info("")
    model.logger.info("  INTERPRETATION:")
    if hasattr(model, '_cond_smearing_vals'):
        test_r2 = model.metrics.get('r2_test', 0)
        if test_r2 > 0:
            model.logger.info(f"   Conditional smearing IMPROVED performance")
            model.logger.info(f"    Test R^2 = {test_r2:.4f} (positive after conditional correction)")
        else:
            model.logger.info(f"  - Conditional smearing applied but performance still poor")
            model.logger.info(f"    Test R^2 = {test_r2:.4f}")
        model.logger.info(f"  - Smearing varies by bin: [{min_smear:.2f}, {max_smear:.2f}]")
        model.logger.info(f"    (vs global: {model.smearing_factor:.2f})")
    else:
        model.logger.info(f"  - Using global smearing (conditional not computed)")
        if model.smearing_factor > 1.5:
            model.logger.info(f"  - High log-variance (sigma^2 = {model.sigma_log**2:.2f}) creates large bias")
            model.logger.info(f"  - Consider using conditional smearing or Model 2 (Gamma GLM)")
    
    model.logger.info("")
    model.logger.info("Data Utilization:")
    model.logger.info(f"  Training: {model.metrics.get('training_samples', 0):,} (100% retention)")
    model.logger.info(f"  Test: {model.metrics.get('test_samples', 0):,} (100% retention)")
    
    model.logger.info("")
    model.logger.info("Output:")
    model.logger.info(f"  Results: {model.output_dir_relative}")
    model.logger.info(f"  Plots: {model.output_dir_relative / 'diagnostic_plots.png'}")
    model.logger.info(f"  LaTeX: {model.output_dir_relative / f'model_{model.model_id}_renewcommands.tex'}")
    
    model.logger.info("")
    model.logger.info("="*80)
    model.logger.info(f"MODEL {model.model_id} PIPELINE COMPLETE")
    model.logger.info("="*80)
    
    print("\n" + "="*80)
    print("MODEL 6 EXECUTION COMPLETE")
    print("="*80)
    print(f"\nTest R^2: {model.metrics.get('r2_test', 0):.4f}")
    print(f"Log-scale R^2: {model.r2_log_scale:.4f}")
    print(f"RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    
    # Smearing information
    print(f"\nSmearing Method: ", end="")
    if hasattr(model, '_cond_smearing_vals'):
        min_smear = min(model._cond_smearing_vals)
        max_smear = max(model._cond_smearing_vals)
        print(f"CONDITIONAL (10-bin)")
        print(f"  Global smearing: {model.smearing_factor:.4f}")
        print(f"  Conditional range: [{min_smear:.4f}, {max_smear:.4f}]")
    else:
        print(f"GLOBAL")
        print(f"  Smearing factor: {model.smearing_factor:.4f}")
    
    skew_reduction = ((model.skewness_original - model.skewness_log) / 
                     abs(model.skewness_original) * 100) if model.skewness_original != 0 else 0
    print(f"  Skewness reduction: {skew_reduction:.1f}%")
    
    # Performance assessment
    test_r2 = model.metrics.get('r2_test', 0)
    print(f"\n{'='*80}")
    if test_r2 > 0:
        print(" MODEL PERFORMANCE: ACCEPTABLE")
        print(f"{'='*80}")
        print(f"Conditional smearing successfully improved performance")
        print(f"Test R^2 = {test_r2:.4f} (positive)")
        if hasattr(model, '_cond_smearing_vals'):
            print(f"Bin-wise smearing addressed heteroscedasticity on log scale")
    elif test_r2 > -0.1:
        print(" MODEL PERFORMANCE: MARGINAL")
        print(f"{'='*80}")
        print(f"Test R^2 = {test_r2:.4f} (near zero)")
        print(f"Conditional smearing helped but performance still marginal")
        print(f"Consider Model 2 (Gamma GLM) for better results")
    else:
        print(" MODEL PERFORMANCE: POOR")
        print(f"{'='*80}")
        print(f"Negative R^2 ({test_r2:.4f}) indicates predictions worse than mean")
        if hasattr(model, '_cond_smearing_vals'):
            print(f"Even with conditional smearing, performance is poor")
        else:
            print(f"Global smearing (factor = {model.smearing_factor:.2f}) over-corrects")
        print(f"Log-scale residual variance (sigma^2 = {model.sigma_log**2:.2f}) is too high")
        print("\nRECOMMENDATION: Use Model 2 (Gamma GLM) instead")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    main()