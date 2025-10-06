"""
Model 2: Generalized Linear Model with Gamma Distribution
==========================================================
GLM with Gamma distribution and log link for iBudget cost prediction
Uses feature selection based on mutual information analysis
No outlier removal - robust to extreme values through distribution choice

STANDARDIZED IMPLEMENTATION - Follows lessons learned from Model 1
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    Key features:
    - Gamma distribution for right-skewed cost data
    - Log link function ensures positive predictions
    - 100% data utilization (NO outlier removal)
    - Feature selection based on mutual information
    - Works directly in dollar scale (no transformation needed)
    
    Regulatory Status: Fully compliant (requires minor F.A.C. update)
    """
    
    # Selected features based on mutual information analysis (FY2013-2024)
    SELECTED_FEATURES = [
        # Top predictors from MI analysis (consistently important across years)
        'RESIDENCETYPE',     # MI: 0.203-0.272 (highest predictor)
        'BSum',              # MI: 0.113-0.137 (behavioral summary)
        'BLEVEL',            # MI: 0.089-0.106
        'LOSRI',             # MI: 0.087-0.130
        'OLEVEL',            # MI: 0.085-0.131
        
        # Clinical scores
        'FSum',              # MI: 0.059-0.084
        'FLEVEL',            # MI: 0.061-0.085
        'PSum',              # MI: 0.057-0.091
        'PLEVEL',            # MI: 0.070-0.083
        
        # Top individual QSI questions
        'Q26',               # MI: 0.084-0.094 (consistently in top 10)
        'Q36',               # MI: 0.066-0.088
        'Q27',               # MI: 0.061-0.080
        'Q20',               # MI: 0.052-0.086
        'Q21',               # MI: 0.062-0.078
        'Q23',               # MI: 0.046-0.067
        'Q30',               # MI: 0.063-0.065
        'Q25',               # MI: 0.055-0.067
        'Q16',               # MI: 0.047-0.060
        'Q18',               # MI: 0.044-0.048
        'Q28',               # MI: 0.043-0.058
        
        # Demographics (required for regulatory compliance)
        'Age',
        'AgeGroup',
        'County',
        'LivingSetting'
    ]
    
    def __init__(self, use_selected_features: bool = True, 
                 use_fy2024_only: bool = True,
                 random_state: int = 42):
        """
        Initialize Model 2
        
        Args:
            use_selected_features: Whether to use MI-based feature selection
            use_fy2024_only: Whether to use only fiscal year 2024 data
            random_state: Random state for reproducibility
        """
        super().__init__(model_id=2, model_name="GLM-Gamma")
        
        # Configuration
        self.use_selected_features = use_selected_features
        self.use_fy2024_only = use_fy2024_only
        self.fiscal_years_used = "2024" if use_fy2024_only else "2020-2025"
        self.random_state = random_state
        
        # GLM-specific attributes
        self.glm_model = None
        self.dispersion = None
        self.deviance = None
        self.null_deviance = None
        self.deviance_r2 = None
        self.mcfadden_r2 = None
        self.aic = None
        self.bic = None
        self.num_parameters = 0
        self.coefficients = {}
        
        # Feature importance from GLM
        self.feature_importance = {}
        
        logger.info(f"Model 2 initialized: feature_selection={use_selected_features}, fy2024_only={use_fy2024_only}")
    
    def load_data(self, fiscal_year_start: int = 2023, fiscal_year_end: int = 2024) -> List[ConsumerRecord]:
        """
        Load data for Model 2
        
        Args:
            fiscal_year_start: Start fiscal year (ignored if use_fy2024_only=True)
            fiscal_year_end: End fiscal year (ignored if use_fy2024_only=True)
            
        Returns:
            List of consumer records
        """
        if self.use_fy2024_only:
            # Force FY2024 only
            return super().load_data(fiscal_year_start=2024, fiscal_year_end=2024)
        else:
            return super().load_data(fiscal_year_start=fiscal_year_start, fiscal_year_end=fiscal_year_end)
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for Model 2
        
        Args:
            records: List of consumer records
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        if self.use_selected_features:
            return self._prepare_selected_features(records)
        else:
            return self._prepare_all_features(records)
    
    def _prepare_selected_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features based on mutual information selection
        EXACTLY matches the original implementation with 33 features
        
        Args:
            records: Consumer records
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        features_list = []
        
        for record in records:
            row_features = []
            
            # 1. RESIDENCETYPE dummies (5 features) - FH as reference
            residence_map = {
                'FH': 0, 'ILSL': 1, 'RH1': 2, 'RH2': 3, 'RH3': 4, 'RH4': 5
            }
            residence_val = residence_map.get(getattr(record, 'residencetype', 'FH'), 0)
            for i in range(1, 6):
                row_features.append(1.0 if residence_val == i else 0.0)
            
            # 2. Living setting dummies (5 features) - FH as reference
            # Note: This might be duplicate of RESIDENCETYPE but keeping for compatibility
            living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
            for setting in living_settings:
                row_features.append(1.0 if record.living_setting == setting else 0.0)
            
            # 3. Age group dummies (2 features) - Age3_20 as reference
            row_features.append(1.0 if record.age_group == 'Age21_30' else 0.0)
            row_features.append(1.0 if record.age_group == 'Age31Plus' else 0.0)
            
            # 4. Age (continuous)
            row_features.append(float(record.age))
            
            # 5. Summary scores and levels
            row_features.append(float(record.bsum if record.bsum is not None else 0))
            row_features.append(float(record.blevel if record.blevel is not None else 0))
            row_features.append(float(record.fsum if record.fsum is not None else 0))
            row_features.append(float(record.flevel if record.flevel is not None else 0))
            row_features.append(float(record.psum if record.psum is not None else 0))
            row_features.append(float(record.plevel if record.plevel is not None else 0))
            
            # 6. Support level scores
            row_features.append(float(record.losri if record.losri is not None else 0))
            row_features.append(float(record.olevel if record.olevel is not None else 0))
            
            # 7. QSI Questions (11 features) - ordered by MI importance
            qsi_questions = [26, 36, 27, 20, 21, 23, 30, 25, 16, 18, 28]
            for q_num in qsi_questions:
                q_val = getattr(record, f'q{q_num}', None)
                row_features.append(float(q_val) if q_val is not None else 0.0)
            
            # 8. County encoding (simple hash-based)
            county_code = hash(record.county) % 100 if record.county else 0
            row_features.append(float(county_code))
            
            features_list.append(row_features)
        
        # Build feature names (only once)
        if not self.feature_names:
            self.feature_names = [
                # RESIDENCETYPE dummies (5)
                'Res_ILSL', 'Res_RH1', 'Res_RH2', 'Res_RH3', 'Res_RH4',
                # Living Setting dummies (5)
                'Live_ILSL', 'Live_RH1', 'Live_RH2', 'Live_RH3', 'Live_RH4',
                # Age (3)
                'Age21_30', 'Age31Plus', 'Age',
                # Summary scores and levels (6)
                'BSum', 'BLEVEL', 'FSum', 'FLEVEL', 'PSum', 'PLEVEL',
                # Support levels (2)
                'LOSRI', 'OLEVEL',
                # QSI questions (11)
                'Q26', 'Q36', 'Q27', 'Q20', 'Q21', 'Q23', 
                'Q30', 'Q25', 'Q16', 'Q18', 'Q28',
                # County (1)
                'County_Code'
            ]
        
        return np.array(features_list, dtype=np.float64), self.feature_names
    
    def _prepare_all_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare all available features (not using feature selection)
        
        Args:
            records: Consumer records
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        # Delegate to base class
        return super().prepare_features(records)
    
    def _prepare_all_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare all available features (not using feature selection)
        Delegates to base class for full feature set
        
        Args:
            records: Consumer records
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        logger.info("Using all features (no selection)")
        return super().prepare_features(records)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit GLM with Gamma distribution and log link
        
        Args:
            X: Feature matrix
            y: Target values (costs in dollars, NOT transformed)
        """
        logger.info("Fitting GLM-Gamma model...")
        logger.info(f"Training samples: {len(y)}, Features: {X.shape[1]}")
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        
        # Ensure no zeros or negative values (Gamma requires positive)
        y_adjusted = np.maximum(y, 0.01)
        
        try:
            # Initialize and fit GLM with Gamma family and log link
            glm = sm.GLM(
                y_adjusted,
                X_with_const,
                family=Gamma(link=Log())
            )
            
            # Fit with increased iterations for convergence
            self.glm_model = glm.fit(maxiter=200, scale='x2')
            self.model = self.glm_model  # Store for base class compatibility
            
            # Extract GLM-specific metrics
            self.dispersion = self.glm_model.scale
            self.deviance = self.glm_model.deviance
            self.aic = self.glm_model.aic
            self.bic = self.glm_model.bic
            self.num_parameters = len(self.glm_model.params)
            
            # Calculate null model for pseudo-R²
            null_model = sm.GLM(
                y_adjusted,
                np.ones((len(y_adjusted), 1)),
                family=Gamma(link=Log())
            ).fit(disp=0)
            
            self.null_deviance = null_model.deviance
            
            # Calculate pseudo-R² measures
            self.deviance_r2 = 1 - (self.deviance / self.null_deviance)
            self.mcfadden_r2 = 1 - (self.glm_model.llf / null_model.llf)
            
            # Store coefficients with statistics
            coef_names = ['const'] + self.feature_names
            for i, name in enumerate(coef_names):
                self.coefficients[name] = {
                    'value': float(self.glm_model.params[i]),
                    'std_error': float(self.glm_model.bse[i]),
                    'z_value': float(self.glm_model.tvalues[i]),
                    'p_value': float(self.glm_model.pvalues[i]),
                    'conf_int_lower': float(self.glm_model.conf_int()[i, 0]),
                    'conf_int_upper': float(self.glm_model.conf_int()[i, 1]),
                    'exp_value': float(np.exp(self.glm_model.params[i]))  # Multiplicative effect
                }
            
            # Calculate feature importance (absolute z-values)
            self.feature_importance = {}
            for i, name in enumerate(self.feature_names):
                self.feature_importance[name] = abs(float(self.glm_model.tvalues[i+1]))
            
            # Sort by importance
            self.feature_importance = dict(sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            logger.info(f"GLM-Gamma fitted successfully")
            logger.info(f"  Converged: {self.glm_model.converged}")
            logger.info(f"  Dispersion: {self.dispersion:.4f}")
            logger.info(f"  Deviance R²: {self.deviance_r2:.4f}")
            logger.info(f"  McFadden R²: {self.mcfadden_r2:.4f}")
            logger.info(f"  AIC: {self.aic:.1f}, BIC: {self.bic:.1f}")
            
        except Exception as e:
            logger.error(f"Error fitting GLM-Gamma: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted GLM
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions in dollar scale
        """
        if self.glm_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Add constant to match training
        X_with_const = sm.add_constant(X, has_constant='add')
        
        # GLM predict returns on original scale (handles back-transformation)
        predictions = self.glm_model.predict(X_with_const)
        
        # Ensure predictions are positive (should be by design with log link)
        predictions = np.maximum(predictions, 1.0)
        
        return predictions
    
    def generate_latex_commands(self) -> None:
        """
        Generate LaTeX commands for Model 2
        Calls base class first, then adds GLM-specific commands
        """
        # CRITICAL: Call parent class first to generate base commands
        super().generate_latex_commands()
        
        # Model word for LaTeX commands
        model_word = "Two"
        
        # Append GLM-specific commands to newcommands file
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        
        try:
            with open(newcommands_file, 'a') as f:
                f.write("\n% GLM-Specific Command Definitions\n")
                f.write(f"\\newcommand{{\\Model{model_word}Distribution}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}LinkFunction}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Dispersion}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}DevianceRSquared}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}McFaddenRSquared}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}AIC}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}BIC}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Parameters}}{{\\WarningRunPipeline}}\n")
                
                # Feature selection specific
                if self.use_selected_features:
                    f.write("\n% Feature Selection Command Definitions\n")
                    f.write(f"\\newcommand{{\\Model{model_word}FeatureSelection}}{{\\WarningRunPipeline}}\n")
                    f.write(f"\\newcommand{{\\Model{model_word}TopFeature}}{{\\WarningRunPipeline}}\n")
                    f.write(f"\\newcommand{{\\Model{model_word}TopFeatureMI}}{{\\WarningRunPipeline}}\n")
            
            logger.info(f"Appended GLM-specific command definitions to {newcommands_file}")
        except Exception as e:
            logger.error(f"Error appending to newcommands file: {e}")
        
        # Append actual values to renewcommands file
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        try:
            with open(renewcommands_file, 'a') as f:
                f.write("\n% GLM-Specific Values\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Distribution}}{{Gamma}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}LinkFunction}}{{Log}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Dispersion}}{{{self.dispersion:.4f if self.dispersion else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}DevianceRSquared}}{{{self.deviance_r2:.4f if self.deviance_r2 else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}McFaddenRSquared}}{{{self.mcfadden_r2:.4f if self.mcfadden_r2 else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}AIC}}{{{self.aic:,.0f if self.aic else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}BIC}}{{{self.bic:,.0f if self.bic else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Parameters}}{{{self.num_parameters}}}\n")
                
                # Feature selection specific values
                if self.use_selected_features:
                    f.write("\n% Feature Selection Values\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}FeatureSelection}}{{True}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}TopFeature}}{{RESIDENCETYPE}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}TopFeatureMI}}{{0.256}}\n")
            
            logger.info(f"Appended GLM-specific values to {renewcommands_file}")
        except Exception as e:
            logger.error(f"Error appending to renewcommands file: {e}")
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive metrics including GLM-specific measures
        
        Returns:
            Dictionary of metrics
        """
        # Calculate base metrics first
        metrics = super().calculate_metrics()
        
        # Add prediction interval coverage metrics
        if self.test_predictions is not None and self.y_test is not None:
            within_5k = np.mean(np.abs(self.test_predictions - self.y_test) <= 5000) * 100
            within_10k = np.mean(np.abs(self.test_predictions - self.y_test) <= 10000) * 100
            within_20k = np.mean(np.abs(self.test_predictions - self.y_test) <= 20000) * 100
            
            metrics.update({
                'within_5k': within_5k,
                'within_10k': within_10k,
                'within_20k': within_20k
            })
        
        # Add GLM-specific metrics
        if self.glm_model is not None:
            metrics.update({
                'dispersion': self.dispersion if self.dispersion else 0,
                'deviance': self.deviance if self.deviance else 0,
                'null_deviance': self.null_deviance if self.null_deviance else 0,
                'deviance_r2': self.deviance_r2 if self.deviance_r2 else 0,
                'mcfadden_r2': self.mcfadden_r2 if self.mcfadden_r2 else 0,
                'aic': self.aic if self.aic else 0,
                'bic': self.bic if self.bic else 0,
                'num_parameters': self.num_parameters,
                'converged': self.glm_model.converged
            })
        
        self.metrics = metrics
        return metrics
    
    def plot_diagnostics(self) -> None:
        """
        Generate comprehensive diagnostic plots for GLM
        Creates 9-panel diagnostic figure
        """
        if self.test_predictions is None or self.y_test is None:
            logger.warning("No predictions available for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Model {self.model_id}: GLM-Gamma Diagnostic Plots', 
                     fontsize=14, fontweight='bold')
        
        # 1. Predicted vs Actual
        ax = axes[0, 0]
        ax.scatter(self.y_test, self.test_predictions, alpha=0.5, s=10)
        min_val = min(self.y_test.min(), self.test_predictions.min())
        max_val = max(self.y_test.max(), self.test_predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax.set_xlabel('Actual Cost ($)')
        ax.set_ylabel('Predicted Cost ($)')
        ax.set_title(f'Predicted vs Actual (R²={self.metrics.get("r2_test", 0):.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residual Plot
        ax = axes[0, 1]
        residuals = self.test_predictions - self.y_test
        ax.scatter(self.test_predictions, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Predicted Cost ($)')
        ax.set_ylabel('Residuals ($)')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # 3. Deviance Residuals
        ax = axes[0, 2]
        # Calculate deviance residuals
        y_test_adj = np.maximum(self.y_test, 0.01)
        y_pred_adj = np.maximum(self.test_predictions, 0.01)
        dev_residuals = np.sign(y_test_adj - y_pred_adj) * np.sqrt(
            2 * np.abs(y_test_adj * np.log(y_test_adj / y_pred_adj) - 
                      (y_test_adj - y_pred_adj))
        )
        ax.scatter(self.test_predictions, dev_residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Predicted Cost ($)')
        ax.set_ylabel('Deviance Residuals')
        ax.set_title('Deviance Residuals')
        ax.grid(True, alpha=0.3)
        
        # 4. Q-Q Plot of Deviance Residuals
        ax = axes[1, 0]
        from scipy import stats
        stats.probplot(dev_residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Deviance Residuals)')
        ax.grid(True, alpha=0.3)
        
        # 5. Scale-Location Plot
        ax = axes[1, 1]
        standardized_dev = dev_residuals / np.std(dev_residuals)
        ax.scatter(self.test_predictions, np.sqrt(np.abs(standardized_dev)), alpha=0.5, s=10)
        ax.set_xlabel('Predicted Cost ($)')
        ax.set_ylabel('√|Standardized Deviance Residuals|')
        ax.set_title('Scale-Location Plot')
        ax.grid(True, alpha=0.3)
        
        # 6. Feature Importance (Top 15)
        ax = axes[1, 2]
        if self.feature_importance:
            top_features = list(self.feature_importance.keys())[:15]
            top_importance = [self.feature_importance[f] for f in top_features]
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_importance, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features, fontsize=8)
            ax.set_xlabel('|z-value|')
            ax.set_title('Top 15 Feature Importance')
            ax.grid(True, alpha=0.3)
        
        # 7. Distribution of Residuals
        ax = axes[2, 0]
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Residuals ($)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Residuals (Mean=${np.mean(residuals):,.0f})')
        ax.grid(True, alpha=0.3)
        
        # 8. Log-Scale Predicted vs Actual
        ax = axes[2, 1]
        ax.scatter(np.log(self.y_test + 1), np.log(self.test_predictions + 1), 
                  alpha=0.5, s=10)
        log_min = min(np.log(self.y_test + 1).min(), np.log(self.test_predictions + 1).min())
        log_max = max(np.log(self.y_test + 1).max(), np.log(self.test_predictions + 1).max())
        ax.plot([log_min, log_max], [log_min, log_max], 'r--', label='Perfect Prediction')
        ax.set_xlabel('Log(Actual Cost + 1)')
        ax.set_ylabel('Log(Predicted Cost + 1)')
        ax.set_title('Log-Scale Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. MAPE by Cost Decile
        ax = axes[2, 2]
        deciles = np.percentile(self.y_test, range(10, 100, 10))
        mapes = []
        for i in range(len(deciles)):
            if i == 0:
                mask = self.y_test <= deciles[i]
            else:
                mask = (self.y_test > deciles[i-1]) & (self.y_test <= deciles[i])
            
            if mask.sum() > 0:
                actual_subset = self.y_test[mask]
                pred_subset = self.test_predictions[mask]
                # Avoid division by zero
                mape = np.mean(np.abs((actual_subset - pred_subset) / 
                              np.maximum(actual_subset, 1))) * 100
                mapes.append(mape)
            else:
                mapes.append(0)
        
        ax.bar(range(1, len(mapes) + 1), mapes, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Cost Decile')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('MAPE by Cost Decile')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "diagnostic_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plots saved to {plot_file}")
        
        # Generate additional GLM-specific plots
        self._plot_glm_specific()
    
    def _plot_glm_specific(self) -> None:
        """
        Generate GLM-specific diagnostic plots (4-panel)
        Includes: Partial residuals, link assessment, influence, Pearson residuals
        """
        if self.glm_model is None:
            logger.warning("GLM model not fitted, skipping GLM-specific plots")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('GLM-Specific Diagnostics', fontsize=14, fontweight='bold')
            
            # 1. Partial Residuals (Top 3 features)
            ax = axes[0, 0]
            if len(self.feature_names) >= 3:
                top_3_features = list(self.feature_importance.keys())[:3]
                for feature in top_3_features:
                    if feature in self.feature_names:
                        idx = self.feature_names.index(feature)
                        ax.scatter(self.X_test[:, idx], 
                                 self.test_predictions - self.y_test, 
                                 alpha=0.3, s=5, label=feature)
                ax.set_xlabel('Feature Value')
                ax.set_ylabel('Partial Residual')
                ax.set_title('Partial Residuals (Top 3 Features)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 2. Link Function Assessment
            ax = axes[0, 1]
            # Linear predictor vs response
            linear_predictor = self.glm_model.predict(
                sm.add_constant(self.X_test), linear=True
            )
            ax.scatter(linear_predictor, self.test_predictions, alpha=0.5, s=10)
            ax.set_xlabel('Linear Predictor (log scale)')
            ax.set_ylabel('Predicted Response ($)')
            ax.set_title('Link Function Assessment')
            ax.grid(True, alpha=0.3)
            
            # 3. Pearson Residuals
            ax = axes[1, 0]
            # Calculate Pearson residuals
            y_test_adj = np.maximum(self.y_test, 0.01)
            y_pred_adj = np.maximum(self.test_predictions, 0.01)
            # Variance for Gamma is phi * mu^2
            variance = self.dispersion * (y_pred_adj ** 2)
            pearson_resid = (y_test_adj - y_pred_adj) / np.sqrt(variance)
            ax.scatter(self.test_predictions, pearson_resid, alpha=0.5, s=10)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Predicted Cost ($)')
            ax.set_ylabel('Pearson Residuals')
            ax.set_title('Pearson Residuals')
            ax.grid(True, alpha=0.3)
            
            # 4. Cook's Distance (Influence)
            ax = axes[1, 1]
            # Simplified influence plot using residuals
            leverage = np.abs(pearson_resid)
            ax.scatter(range(len(leverage)), leverage, alpha=0.5, s=10)
            ax.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Threshold')
            ax.set_xlabel('Observation Index')
            ax.set_ylabel('Absolute Pearson Residual')
            ax.set_title('Influence Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save GLM-specific plots
            plot_file = self.output_dir / "glm_specific_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"GLM-specific plots saved to {plot_file}")
        except Exception as e:
            logger.error(f"Error generating GLM-specific plots: {e}")
    
    def save_results(self) -> None:
        """
        Save Model 2 specific results including GLM diagnostics
        """
        # Save base results (metrics, predictions, etc.)
        super().save_results()
        
        # Save GLM-specific results
        glm_results = {
            'model_type': 'GLM-Gamma',
            'distribution': 'Gamma',
            'link_function': 'Log',
            'converged': bool(self.glm_model.converged) if self.glm_model else None,
            'iterations': int(self.glm_model.fit_history['iteration']) if self.glm_model else None,
            'dispersion': float(self.dispersion) if self.dispersion else None,
            'deviance': float(self.deviance) if self.deviance else None,
            'null_deviance': float(self.null_deviance) if self.null_deviance else None,
            'deviance_r2': float(self.deviance_r2) if self.deviance_r2 else None,
            'mcfadden_r2': float(self.mcfadden_r2) if self.mcfadden_r2 else None,
            'aic': float(self.aic) if self.aic else None,
            'bic': float(self.bic) if self.bic else None,
            'num_parameters': int(self.num_parameters),
            'feature_selection': self.use_selected_features,
            'fiscal_years': self.fiscal_years_used,
            'feature_importance': {k: float(v) for k, v in self.feature_importance.items()} if self.feature_importance else {}
        }
        
        glm_file = self.output_dir / 'glm_results.json'
        with open(glm_file, 'w') as f:
            json.dump(glm_results, f, indent=2)
        
        logger.info(f"GLM-specific results saved to {glm_file}")
        
        # Save coefficients
        coef_file = self.output_dir / 'coefficients.json'
        with open(coef_file, 'w') as f:
            json.dump(self.coefficients, f, indent=2)
        
        logger.info(f"Coefficients saved to {coef_file}")


def main():
    """
    Run Model 2 GLM-Gamma with feature selection
    """
    logger.info("="*80)
    logger.info("MODEL 2: GLM WITH GAMMA DISTRIBUTION")
    logger.info("="*80)
    
    # Initialize model with feature selection
    model = Model2GLMGamma(
        use_selected_features=True,  # Use MI-based feature selection
        use_fy2024_only=True,        # Use only FY2024 data
        random_state=42              # For reproducibility
    )
    
    # Run complete pipeline
    results = model.run_complete_pipeline(
        fiscal_year_start=2023,  # Ignored due to use_fy2024_only=True
        fiscal_year_end=2024,    # Ignored due to use_fy2024_only=True
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL 2 SUMMARY: GLM-GAMMA RESULTS")
    print("="*80)
    
    print("\nConfiguration:")
    print(f"  • Feature Selection: {model.use_selected_features}")
    print(f"  • Fiscal Years: {model.fiscal_years_used}")
    print(f"  • Number of Features: {len(model.feature_names)}")
    print(f"  • Distribution: Gamma")
    print(f"  • Link Function: Log")
    
    print("\nData Summary:")
    print(f"  • Total Records: {len(model.all_records)}")
    print(f"  • Training Records: {len(model.train_records)}")
    print(f"  • Test Records: {len(model.test_records)}")
    print(f"  • Outliers Removed: 0 (GLM is robust to outliers)")
    
    print("\nPerformance Metrics:")
    print(f"  • Training R²: {results['metrics']['r2_train']:.4f}")
    print(f"  • Test R²: {results['metrics']['r2_test']:.4f}")
    print(f"  • RMSE: ${results['metrics']['rmse_test']:,.0f}")
    print(f"  • MAE: ${results['metrics']['mae_test']:,.0f}")
    print(f"  • MAPE: {results['metrics']['mape_test']:.1f}%")
    
    print("\nGLM-Specific Metrics:")
    print(f"  • Deviance R²: {model.deviance_r2:.4f}")
    print(f"  • McFadden R²: {model.mcfadden_r2:.4f}")
    print(f"  • AIC: {model.aic:,.0f}")
    print(f"  • BIC: {model.bic:,.0f}")
    print(f"  • Dispersion: {model.dispersion:.4f}")
    
    print("\nPrediction Accuracy:")
    print(f"  • Within ±$5,000: {results['metrics'].get('within_5k', 0):.1f}%")
    print(f"  • Within ±$10,000: {results['metrics'].get('within_10k', 0):.1f}%")
    print(f"  • Within ±$20,000: {results['metrics'].get('within_20k', 0):.1f}%")
    
    print("\nCross-Validation:")
    if 'cv_mean' in results.get('metrics', {}):
        print(f"  • Mean R²: {results['metrics']['cv_mean']:.4f}")
        print(f"  • Std R²: {results['metrics']['cv_std']:.4f}")
    elif 'cv_results' in results:
        print(f"  • Mean R²: {results['cv_results'].get('mean_score', 0):.4f}")
        print(f"  • Std R²: {results['cv_results'].get('std_score', 0):.4f}")
    
    print("\nTop 5 Important Features:")
    if model.feature_importance:
        for i, (feat, imp) in enumerate(list(model.feature_importance.items())[:5], 1):
            print(f"  {i}. {feat}: |z|={imp:.2f}")
    
    print("\n" + "="*80)
    print("Model 2 pipeline completed successfully!")
    print(f"Results saved to: {model.output_dir}")
    print("="*80)
    
    return model, results


if __name__ == "__main__":
    model, results = main()