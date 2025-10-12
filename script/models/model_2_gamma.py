"""
model_2_gamma_final.py
======================
Model 2: Generalized Linear Model with Gamma Distribution
Final implementation with comprehensive feature selection and transformation options

Key improvements:
- All features with MI > 0.05 from feature selection analysis
- Optional sqrt transformation (like Model 1)
- Optional outlier removal (like Model 1)
- Removed use_fy2024_only flag
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import logging
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import base class
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

class Model2GLMGamma(BaseiBudgetModel):
    """
    Model 2: GLM with Gamma distribution and log link
    
    Key characteristics:
    - Models right-skewed costs with or without transformation
    - Log link ensures positive predictions
    - Quadratic variance function (Var ~ mu^2)
    - Optional outlier removal
    - Comprehensive feature selection based on MI analysis
    """
    
    def __init__(self, 
                 use_sqrt_transform: bool = False,
                 use_outlier_removal: bool = False,
                 outlier_threshold: float = 1.645,
                 use_selected_features: bool = True,
                 random_seed: int = 42,
                 log_suffix: Optional[str] = None
                 ):
        """
        Initialize Model 2 GLM-Gamma
        
        Args:
            use_sqrt_transform: Apply sqrt transformation (like Model 1)
            use_outlier_removal: Remove outliers using studentized residuals
            outlier_threshold: Threshold for outlier removal (1.645 = ~10%)
            use_selected_features: Use MI-based feature selection
            output_dir: Directory for outputs
            random_seed: Random seed for reproducibility
        """
        # ****************************************************************************
        # WARNING : The flag 'use_sqrt_transform' here is for research only. 
        # A Gamma model already assumes E[Y]>0 and Var(Y)=mu^(2*phi). 
        # Applying sqrt(Y) then Gamma on sqrt(Y) breaks the distributional assumption.
        # I leave the capability to tranform  only for didactical purposes
        # ****************************************************************************
        # Determine transformation
        transformation = 'sqrt' if use_sqrt_transform else 'none'
        
        # Initialize base class with Model 1-like options
        super().__init__(
            model_id=2,
            model_name="GLM-Gamma",
            use_outlier_removal=use_outlier_removal,
            outlier_threshold=outlier_threshold,
            transformation='none',
            random_seed=random_seed,
            log_suffix = log_suffix
        )
        
        # Model-specific configuration
        self.use_selected_features = use_selected_features
        
        # GLM-specific attributes
        self.glm_model = None
        self.dispersion = None
        self.deviance = None
        self.aic = None
        self.bic = None
        self.null_deviance = None
        self.deviance_r2 = None
        self.mcfadden_r2 = None
        
        # Comprehensive feature selection based on MI > 0.05 analysis
        # From FeatureSelection.txt and model logs
        self.high_mi_qsi = [
            26, 36, 27, 20, 21, 23, 30, 25,  # Top 8
            16, 18, 28, 33, 34, 43, 44,      # Next 7
            15, 19, 22, 24, 29, 35, 37, 38,  # Additional with MI > 0.05
            39, 40, 41, 42, 45, 46, 47       # More QSI items
        ]
        
        # Coefficient storage
        self.coefficients = {}
        
        # Additional metrics
        self.glm_diagnostics = {}
        
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Simplified feature preparation using generic method
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
            'qsi': high_mi_qsi[:15] if self.use_selected_features else list(range(14, 51)),
            'interactions': [
                ('SupportedLiving_x_LOSRI', lambda r: (1 if r.living_setting in ['RH1','RH2','RH3','RH4'] else 0) * float(r.losri)),
                ('Age_x_BSum', lambda r: float(r.age) * float(r.bsum) / 100.0),
                ('FH_x_FSum', lambda r: (1 if r.living_setting == 'FH' else 0) * float(r.fsum))
            ]
        }
        
        return self.prepare_features_from_spec(records, feature_config)

    
    def _fit_core(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit GLM with Gamma distribution and log link
        
        Note: y may be in transformed scale if use_sqrt_transform=True
        
        Args:
            X: Feature matrix (possibly with outliers removed)
            y: Target values (possibly sqrt transformed)
        """
        self.log_section("FITTING GLM-GAMMA MODEL")
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        
        # For Gamma, we need positive values
        # If sqrt transformed, we're already positive
        # If not transformed, ensure minimum positive value
        min_value = 1.0 if self.transformation == 'none' else 0.05
        y_adjusted = np.maximum(y, min_value)
        
        if (y <= 0).any():
            n_adjusted = (y <= 0).sum()
            self.logger.warning(f"Adjusted {n_adjusted} non-positive values to {min_value}")
        
        try:
            # Initialize GLM with Gamma family and log link
            glm = sm.GLM(
                y_adjusted,
                X_with_const,
                family=Gamma(link=Log())
            )
            
            # Fit model with increased iterations for convergence
            self.glm_model = glm.fit(maxiter=200, disp=0)  # Do **not** use 'scale'. Let Gamma estimate dispersion
            self.model = self.glm_model  # Store for base class compatibility
            
            # Core model attributes
            self.dispersion = float(self.glm_model.scale)
            self.deviance = float(self.glm_model.deviance)
            self.aic = float(self.glm_model.aic)
            self.num_parameters = len(self.glm_model.params)

            # BIC is not on GLMResults; compute manually
            n = int(X_with_const.shape[0])
            k = int(len(self.glm_model.params))   # includes intercept
            self.bic = float(n * np.log(self.deviance / max(n, 1)) + k * np.log(max(n, 1)))            
            
            # Calculate null model for pseudo-R^2
            null_model = sm.GLM(
                y_adjusted,
                np.ones((len(y_adjusted), 1)),
                family=Gamma(link=Log())
            ).fit(disp=0)
            
            self.null_deviance = float(null_model.deviance)
            
            # Calculate pseudo-R^2 measures
            # Null model with intercept only
            null_X = np.ones((len(y_adjusted), 1))
            null_model = sm.GLM(y_adjusted, null_X, family=Gamma(link=Log())).fit(disp=0)
            self.null_deviance = float(null_model.deviance)

            # Pseudo-R^2 metrics
            eps = 1e-12
            self.deviance_r2 = float(1.0 - (self.deviance / max(self.null_deviance, eps)))
            #self.mcfadden_r2 = float(1.0 - (self.glm_model.llf / max(null_model.llf, -eps)))
            if null_model.llf == 0:
                self.mcfadden_r2 = np.nan
            else:
                self.mcfadden_r2 = 1.0 - (self.glm_model.llf / null_model.llf)

            # Store coefficients with statistics
            coef_names = ['const'] + self.feature_names
            for i, name in enumerate(coef_names):
                self.coefficients[name] = {
                    'value': float(self.glm_model.params[i]),
                    'std_error': float(self.glm_model.bse[i]),
                    'z_value': float(self.glm_model.tvalues[i]),
                    'p_value': float(self.glm_model.pvalues[i]),
                    'exp_value': float(np.exp(self.glm_model.params[i])),
                    'pct_effect': (np.exp(self.glm_model.params[i]) - 1) * 100
                }
            
            # Log summary
            self.logger.info(f"GLM-Gamma model fitted successfully")
            self.logger.info(f"  Features: {len(self.feature_names)}")
            self.logger.info(f"  Transformation: {self.transformation}")
            self.logger.info(f"  Outliers removed: {self.use_outlier_removal}")
            if self.use_outlier_removal and hasattr(self, 'outlier_diagnostics'):
                if 'n_removed' in self.outlier_diagnostics:
                    self.logger.info(f"  Outliers removed: {self.outlier_diagnostics['n_removed']}")
            self.logger.info(f"  Dispersion parameter: {self.dispersion:.4f}")
            self.logger.info(f"  Deviance: {self.deviance:.2f}")
            self.logger.info(f"  AIC: {self.aic:.2f}, BIC: {self.bic:.2f}")
            self.logger.info(f"  Deviance R^2: {self.deviance_r2:.4f}")
            self.logger.info(f"  McFadden R^2: {self.mcfadden_r2:.4f}")
            
            # Prepare feature list for base class logging method
            feature_list = []
            for name, coef_data in self.coefficients.items():
                if name != 'const':  # Skip intercept
                    feature_list.append({
                        'name': name,
                        'coefficient': coef_data['value'],
                        'std_error': coef_data['std_error'],
                        'p_value': coef_data['p_value'],
                        'effect_pct': coef_data['pct_effect'],
                        'z_value': coef_data.get('z_value', None)
                    })
            
            # Use base class method to log ALL features
            self.log_feature_importance(feature_list, model_type="GLM-Gamma")
                
        except Exception as e:
            self.logger.error(f"Error fitting GLM-Gamma: {str(e)}")
            raise
    
    def _predict_core(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using GLM model
        
        Returns predictions in the same scale as training
        (sqrt scale if transformed, dollar scale if not)
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions in training scale
        """
        if self.glm_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Add constant
        X_with_const = sm.add_constant(X)
        
        # GLM predictions are in the scale of the training data
        predictions = self.glm_model.predict(X_with_const)
        
        return predictions

    def predict_original(self, X: np.ndarray) -> np.ndarray:
        """
        Override base-class hook.
        Model 2 (Gamma + log link) already returns predictions in original dollar scale
        inside _predict_core(), so we just call it directly.

        This prevents the base class from applying inverse_transformation() again
        during cross-validation or evaluation.
        """
        y_pred = self._predict_core(X)
        return np.maximum(0.0, y_pred)
    
    def calculate_metrics_with_proper_mape(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          mape_threshold: float = 1000.0) -> Dict[str, float]:
        """
        Calculate metrics with proper MAPE handling
        
        Args:
            y_true: True values
            y_pred: Predicted values
            mape_threshold: Minimum value for MAPE calculation
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Standard metrics
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # MAPE with threshold to avoid division by small numbers
        mask = y_true > mape_threshold
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['mape'] = mape
            metrics['mape_n'] = mask.sum()
        else:
            metrics['mape'] = np.nan
            metrics['mape_n'] = 0
        
        # Symmetric MAPE (more stable)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
        metrics['smape'] = smape
        
        # Median APE (robust to outliers)
        ape_values = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100 if mask.sum() > 0 else [np.nan]
        metrics['median_ape'] = np.median(ape_values)
        
        return metrics
    
    def generate_diagnostic_plots(self) -> None:
        """
        Generate comprehensive diagnostic plots for GLM-Gamma
        """
        if self.glm_model is None or self.X_test is None:
            self.logger.warning("Model not fitted or no test data for plots")
            return
        
        self.log_section("GENERATING DIAGNOSTIC PLOTS")
        
        # Get predictions
        test_predictions = self.predict(self.X_test)  # These are already inverse transformed if needed
        
        # Create figure with 6 subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model 2: GLM-Gamma Diagnostic Plots', fontsize=14, fontweight='bold')
        
        # 1. Predicted vs Actual
        ax = axes[0, 0]
        ax.scatter(self.y_test, test_predictions, alpha=0.5, s=10)
        
        # Add perfect prediction line
        min_val = min(self.y_test.min(), test_predictions.min())
        max_val = max(self.y_test.max(), test_predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7)
        
        ax.set_xlabel('Actual Cost ($)')
        ax.set_ylabel('Predicted Cost ($)')
        ax.set_title('Predicted vs Actual')
        
        # Calculate metrics
        metrics = self.calculate_metrics_with_proper_mape(self.y_test, test_predictions)
        
        # Add metrics annotation
        text = f"R^2 = {metrics['r2']:.4f}\nRMSE = ${metrics['rmse']:,.0f}\n"
        if not np.isnan(metrics['mape']):
            text += f"MAPE = {metrics['mape']:.1f}% (n={metrics['mape_n']})\n"
        text += f"SMAPE = {metrics['smape']:.1f}%"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residual Plot
        ax = axes[0, 1]
        residuals = self.y_test - test_predictions
        ax.scatter(test_predictions, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Fitted Values ($)')
        ax.set_ylabel('Residuals ($)')
        ax.set_title('Residual Plot')
        
        # 3. Q-Q Plot
        ax = axes[0, 2]
        standardized_residuals = residuals / np.std(residuals)
        stats.probplot(standardized_residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        
        # 4. Scale-Location Plot
        ax = axes[1, 0]
        sqrt_abs_std_resid = np.sqrt(np.abs(standardized_residuals))
        ax.scatter(test_predictions, sqrt_abs_std_resid, alpha=0.5, s=10)
        ax.set_xlabel('Fitted Values ($)')
        ax.set_ylabel('sqrt|Standardized Residuals|')
        ax.set_title('Scale-Location Plot')
        
        # 5. Histogram of Residuals
        ax = axes[1, 1]
        ax.hist(standardized_residuals, bins=30, density=True, alpha=0.7, 
                color='blue', edgecolor='black')
        x = np.linspace(standardized_residuals.min(), standardized_residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, label='N(0,1)')
        ax.set_xlabel('Standardized Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Histogram of Residuals')
        ax.legend()
        
        # 6. Feature Importance (Top 10)
        ax = axes[1, 2]
        
        # Get top 10 features by absolute coefficient value
        feature_importance = [
            (name, abs(coef['value']), coef['p_value'])
            for name, coef in self.coefficients.items()
            if name != 'const'
        ]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance[:10]
        
        names = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        colors = ['green' if f[2] < 0.05 else 'gray' for f in top_features]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('|Coefficient|')
        ax.set_title('Top 10 Features')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "diagnostic_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Diagnostic plots saved to {plot_file}")
    
    def generate_latex_commands(self) -> None:
        """
        Generate LaTeX commands for Model 2
        Minimal approach using existing base class functionality
        """
        # STEP 1: Let base class do all the standard work
        super().generate_latex_commands()
        
        # STEP 2: Append Model 2's unique GLM metrics to both files
        if self.glm_model is None:
            return  # Nothing to add if model not fitted
        
        # Append to newcommands file (placeholders)
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        with open(newcommands_file, 'a') as f:
            f.write("\n% Model 2 GLM-Specific Command Placeholders\n")
            f.write("\\newcommand{\\ModelTwoDispersion}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelTwoDeviance}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelTwoNullDeviance}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelTwoDevianceRSquared}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelTwoMcFaddenRSquared}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelTwoAIC}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelTwoBIC}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelTwoDistribution}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelTwoLinkFunction}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelTwoVarianceFunction}{\\WarningRunPipeline}\n")
            f.write("\\newcommand{\\ModelTwoParameters}{\\WarningRunPipeline}\n")

        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        with open(renewcommands_file, 'a') as f:
            f.write("\n% GLM-Specific Values\n")
            f.write("\\renewcommand{\\ModelTwoDistribution}{Gamma}\n")
            f.write("\\renewcommand{\\ModelTwoLinkFunction}{Log}\n")
            f.write(f"\\renewcommand{{\\ModelTwoDispersion}}{{{self.dispersion:.4f}}}\n")
            f.write(f"\\renewcommand{{\\ModelTwoNullDeviance}}{{{self.null_deviance:.2f}}}\n")
            f.write(f"\\renewcommand{{\\ModelTwoDeviance}}{{{self.deviance:.2f}}}\n")
            f.write(f"\\renewcommand{{\\ModelTwoDevianceRSquared}}{{{self.deviance_r2:.4f}}}\n")
            f.write(f"\\renewcommand{{\\ModelTwoMcFaddenRSquared}}{{{self.mcfadden_r2:.4f}}}\n")
            f.write(f"\\renewcommand{{\\ModelTwoAIC}}{{{self.aic:.1f}}}\n")
            f.write(f"\\renewcommand{{\\ModelTwoBIC}}{{{self.bic:.1f}}}\n")
            f.write(f"\\renewcommand{{\\ModelTwoParameters}}{{{len(self.glm_model.params)}}}\n")
            f.write(f"\\renewcommand{{\\ModelTwoVarianceFunction}}{{Quadratic}}\n")
        
        self.logger.info("Model 2 GLM-specific commands appended to both files")

def main():
    """
    Run Model 2 GLM-Gamma implementation
    """
    logger.info("="*80)
    logger.info("MODEL 2: GLM WITH GAMMA DISTRIBUTION (FINAL)")
    logger.info("="*80)
    
    # Initialize model with Model 1-like options
    # ****************************************************************************
    # WARNING : The flag 'use_sqrt_transform' here is for research only. 
    # A Gamma model already assumes E[Y]>0 and Var(Y)=mu^(2*phi) 
    # Applying sqrt(Y) then Gamma on sqrt(Y) breaks the distributional assumption.
    # I leave the capability to tranform  only for didactical purposes
    # ****************************************************************************
    use_sqrt = False                 # Didactic purpose. Should ALWAYS be False for Gamma
    use_outlier = True
    suffix = 'Sqrt_' + str(use_sqrt) + '_Outliers_' + str(use_outlier)
    model = Model2GLMGamma(
        use_sqrt_transform=False,    # Didactic purpose. Should ALWAYS be False for Gamma
        use_outlier_removal=True,      
        outlier_threshold=1.645,     # ~10% outliers (Model 5b default) 
        use_selected_features=False, # Use MI-based feature selection
        random_seed=42,              # For reproducibility
        log_suffix=suffix
    )
    
    # Run complete pipeline
    results = model.run_complete_pipeline(
        fiscal_year_start=2024,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    # Generate diagnostic plots
    model.generate_diagnostic_plots()
    
    # Print summary
    model.logger.info("=" * 80)
    # Log final summary to BOTH file and console
    model.log_section("MODEL 2 FINAL SUMMARY", "=")
    
    # Build summary for logging
    summary_lines = []
    summary_lines.append("Configuration:")
    summary_lines.append(f"   Distribution: Gamma")
    summary_lines.append(f"   Link Function: Log")
    summary_lines.append(f"   Transformation: {model.transformation}")
    summary_lines.append(f"   Outlier Removal: {model.use_outlier_removal}")
    
    if model.use_outlier_removal and hasattr(model, 'outlier_diagnostics'):
        if model.outlier_diagnostics:
            summary_lines.append(f"   Outliers Removed: {model.outlier_diagnostics.get('n_removed', 0)} "
                               f"({model.outlier_diagnostics.get('pct_removed', 0):.1f}%)")
    
    summary_lines.append(f"   Number of Features: {len(model.feature_names)}")
    
    summary_lines.append("Feature Categories:")
    summary_lines.append(f"   Living Settings: 5")
    summary_lines.append(f"   Age Groups: 2")
    summary_lines.append(f"   Support Levels: 5")
    summary_lines.append(f"   Clinical Scores: 3")
    summary_lines.append(f"   Demographics: 2")
    summary_lines.append(f"   QSI Items: 15+")
    summary_lines.append(f"   Interactions: 3")
    
    summary_lines.append("Data Summary:")
    summary_lines.append(f"   Total Records: {len(model.all_records)}")
    summary_lines.append(f"   Training Records: {len(model.train_records)}")
    summary_lines.append(f"   Test Records: {len(model.test_records)}")
    
    summary_lines.append("Model Performance:")
    if model.metrics:
        summary_lines.append(f"   Training R^2: {model.metrics.get('r2_train', 0):.4f}")
        summary_lines.append(f"   Test R^2: {model.metrics.get('r2_test', 0):.4f}")
        summary_lines.append(f"   RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
        summary_lines.append(f"   MAE: ${model.metrics.get('mae_test', 0):,.2f}")
        summary_lines.append(f"   CV R^2 (mean +- std): {model.metrics.get('cv_mean', 0):.4f} +- {model.metrics.get('cv_std', 0):.4f}")
    
    summary_lines.append("GLM-Specific Metrics:")
    if model.glm_model:
        summary_lines.append(f"   Dispersion Parameter: {model.dispersion:.3f}")
        summary_lines.append(f"   Deviance R^2: {model.deviance_r2:.4f}")
        summary_lines.append(f"   McFadden R^2: {model.mcfadden_r2:.4f}")
        summary_lines.append(f"   AIC: {model.aic:.1f}")
        
        n = len(model.y_train) if hasattr(model, 'y_train') and model.y_train is not None else 1
        k = model.num_parameters
        correct_bic = -2 * model.glm_model.llf + k * np.log(n)
        summary_lines.append(f"   BIC (corrected): {correct_bic:.1f}")
    
    summary_lines.append("Top 5 Features (by coefficient magnitude):")
    if model.coefficients:
        sig_features = [
            (name, coef['value'], coef['pct_effect'])
            for name, coef in model.coefficients.items()
            if coef['p_value'] < 0.05 and name != 'const'
        ]
        sig_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for i, (name, coef, pct) in enumerate(sig_features[:5], 1):
            summary_lines.append(f"  {i}. {name}: Beta={coef:.4f} ({pct:+.1f}% effect)")
    
    # Write everything to log file
    for line in summary_lines:
        model.logger.info(line)
    
    model.logger.info("")
    model.logger.info("=" * 80)
    model.logger.info("Model 2 pipeline complete!")
    model.logger.info(f"Results saved to: {model.output_dir_relative}")
    model.logger.info("=" * 80)
    
    # Also print a brief summary to console for immediate visibility
    print("\n" + "="*80)
    print("MODEL 2 COMPLETE - See log file for detailed summary")
    print(f"Test R^2: {model.metrics.get('r2_test', 0):.4f}")
    print(f"RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    print(f"Results saved to: {model.output_dir_relative}")
    print("="*80)

if __name__ == "__main__":
    main()