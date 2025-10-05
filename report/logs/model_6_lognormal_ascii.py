"""
model_6_lognormal.py
====================
Model 6: Log-Normal GLM with log(sqrt(Y)) transformation
Uses Duan's smearing estimator for retransformation bias correction
No outlier removal - uses all available data
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

# Import base class
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

class Model6LogNormal(BaseiBudgetModel):
    """
    Model 6: Log-Normal GLM with log(sqrt(Y)) transformation
    
    Key features:
    - No outlier removal (uses all data)
    - Log transformation of square root costs
    - Normal distribution on log scale (log-normal on original scale)
    - Duan's smearing estimator for retransformation
    - Multiplicative effects interpretation
    - Built-in heteroscedasticity handling
    """
    
    def __init__(self, use_fy2024_only: bool = True):
        """Initialize Model 6"""
        super().__init__(model_id=6, model_name="Log-Normal-GLM")
        self.use_fy2024_only = use_fy2024_only
        self.fiscal_years_used = "2024" if use_fy2024_only else "2023-2024"
        
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
        
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Override split_data to ensure proper train/test split
        CRITICAL: Handles boolean test_size from base class
        
        Args:
            test_size: Proportion for test set  
            random_state: Random seed
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
        
        # Split records
        self.train_records = [self.all_records[i] for i in train_indices]
        self.test_records = [self.all_records[i] for i in test_indices]
        
        # Prepare features and targets
        self.X_train, self.feature_names = self.prepare_features(self.train_records)
        self.X_test, _ = self.prepare_features(self.test_records)
        
        # Target: original costs (will be transformed in fit())
        self.y_train = np.array([r.total_cost for r in self.train_records])
        self.y_test = np.array([r.total_cost for r in self.test_records])
        
        logger.info(f"Data split: {len(self.train_records)} training, {len(self.test_records)} test")
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix using ONLY robust features from FeatureSelection.txt
        
        Robust features identified:
        - Living Setting (RESIDENCETYPE)
        - Age Groups
        - BSum, FSum (summary scores)
        - Q16, Q18, Q20, Q21, Q23, Q28, Q33, Q34, Q36, Q43
        - Primary Disability indicators
        """
        feature_list = []
        
        # Track features to match Model 2/4 pattern (22 features total)
        feature_names = []
        
        for record in records:
            features = []
            
            # 1. Living Setting (5 dummies, FH as reference)
            features.append(1 if record.living_setting == 'ILSL' else 0)
            features.append(1 if record.living_setting == 'RH1' else 0)
            features.append(1 if record.living_setting == 'RH2' else 0)
            features.append(1 if record.living_setting == 'RH3' else 0)
            features.append(1 if record.living_setting == 'RH4' else 0)
            
            # 2. Age Groups (2 dummies, Age3_20 as reference)
            features.append(1 if record.age_group == 'Age21_30' else 0)
            features.append(1 if record.age_group == 'Age31Plus' else 0)
            
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
        Fit Log-Normal GLM using OLS on log(sqrt(Y))
        
        Args:
            X_train: Training feature matrix
            y_train: Training target values (costs)
        """
        logger.info("Fitting Log-Normal GLM...")
        
        # Transform target: log(sqrt(Y))
        # Add small constant to handle zeros
        y_sqrt = np.sqrt(y_train + 1e-10)
        y_log_sqrt = np.log(y_sqrt)
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X_train)
        
        try:
            # Fit OLS on log-transformed target
            self.ols_model = sm.OLS(y_log_sqrt, X_with_const).fit()
            self.model = self.ols_model  # Store for base class compatibility
            
            # Extract model metrics
            self.r2_log_scale = self.ols_model.rsquared
            self.sigma_log = np.sqrt(self.ols_model.scale)
            self.aic = self.ols_model.aic
            self.bic = self.ols_model.bic
            
            # Calculate Duan's smearing factor for retransformation
            residuals_log = self.ols_model.resid
            self.smearing_factor = np.mean(np.exp(residuals_log))
            
            # Calculate skewness measures
            # Original scale residuals (for comparison)
            y_train_pred_original = self.predict(X_train)
            residuals_original = y_train - y_train_pred_original
            self.skewness_original = stats.skew(residuals_original)
            self.skewness_log = stats.skew(residuals_log)
            
            # Breusch-Pagan test for heteroscedasticity on log scale
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_test = het_breuschpagan(residuals_log, X_with_const)
            self.heteroscedasticity_pval = bp_test[1]  # p-value
            
            # Store coefficients with statistics
            coef_names = ['const'] + self.feature_names
            self.coefficients = {}
            
            # Get confidence intervals (returns numpy array)
            conf_int = self.ols_model.conf_int()
            
            for i, name in enumerate(coef_names):
                self.coefficients[name] = {
                    'estimate': float(self.ols_model.params[i]),
                    'std_error': float(self.ols_model.bse[i]),
                    't_value': float(self.ols_model.tvalues[i]),
                    'p_value': float(self.ols_model.pvalues[i]),
                    'conf_lower': float(conf_int[i, 0]),
                    'conf_upper': float(conf_int[i, 1]),
                    'multiplicative_effect': float(np.exp(self.ols_model.params[i]) - 1) * 100  # Percent change
                }
            
            logger.info(f"Model fitted successfully")
            logger.info(f"R^2 (log scale): {self.r2_log_scale:.4f}")
            logger.info(f"Sigma (log scale): {self.sigma_log:.4f}")
            logger.info(f"Smearing factor: {self.smearing_factor:.4f}")
            logger.info(f"Skewness reduction: {self.skewness_original:.4f} ? {self.skewness_log:.4f}")
            logger.info(f"Heteroscedasticity test p-value: {self.heteroscedasticity_pval:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict costs with retransformation and bias correction
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted costs (original scale)
        """
        if self.ols_model is None:
            raise ValueError("Model not fitted yet")
        
        # Add constant
        X_with_const = sm.add_constant(X)
        
        # Predict on log(sqrt(Y)) scale
        log_sqrt_pred = self.ols_model.predict(X_with_const)
        
        # Retransform with Duan's smearing estimator
        # sqrt(Y) = exp(log_pred) * smearing_factor
        # Y = [exp(log_pred) * smearing_factor]^2
        sqrt_pred = np.exp(log_sqrt_pred) * self.smearing_factor
        predictions = sqrt_pred ** 2
        
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
                'num_parameters': self.num_parameters
            })
        
        return metrics
    
    def generate_latex_commands(self) -> None:
        """Override to add Log-Normal specific LaTeX commands"""
        # Generate base commands
        super().generate_latex_commands()
        
        # Model word for LaTeX commands
        model_word = "Six"
        
        # Add Log-Normal specific commands to newcommands file
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        
        with open(newcommands_file, 'a') as f:
            f.write("\n% Log-Normal GLM Specific Commands\n")
            f.write(f"\\newcommand{{\\ModelSixRSquaredLogScale}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\ModelSixSigma}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\ModelSixSmearingFactor}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\ModelSixSkewnessReduction}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\ModelSixHeteroscedasticityTest}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\ModelSixSmearingBias}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\ModelSixAIC}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\ModelSixBIC}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\ModelSixNRobustFeatures}}{{\\WarningRunPipeline}}\n")
        
        # Add Log-Normal specific commands to renewcommands file  
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        with open(renewcommands_file, 'a') as f:
            f.write("\n% Log-Normal GLM Specific Metrics\n")
            if self.ols_model is not None:
                f.write(f"\\renewcommand{{\\ModelSixRSquaredLogScale}}{{{self.r2_log_scale:.4f}}}\n")
                f.write(f"\\renewcommand{{\\ModelSixSigma}}{{{self.sigma_log:.4f}}}\n")
                f.write(f"\\renewcommand{{\\ModelSixSmearingFactor}}{{{self.smearing_factor:.4f}}}\n")
                
                skew_reduction = ((self.skewness_original - self.skewness_log) / 
                                 abs(self.skewness_original) * 100) if self.skewness_original != 0 else 0
                f.write(f"\\renewcommand{{\\ModelSixSkewnessReduction}}{{{skew_reduction:.1f}}}\n")
                f.write(f"\\renewcommand{{\\ModelSixHeteroscedasticityTest}}{{{self.heteroscedasticity_pval:.4f}}}\n")
                
                # Retransformation bias as percent
                smearing_bias = (self.smearing_factor - 1) * 100
                f.write(f"\\renewcommand{{\\ModelSixSmearingBias}}{{{smearing_bias:.2f}}}\n")
                
                f.write(f"\\renewcommand{{\\ModelSixAIC}}{{{self.aic:,.0f}}}\n")
                f.write(f"\\renewcommand{{\\ModelSixBIC}}{{{self.bic:,.0f}}}\n")
                f.write(f"\\renewcommand{{\\ModelSixNRobustFeatures}}{{{len(self.feature_names)}}}\n")
    
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
    
    def plot_lognormal_diagnostics(self) -> None:
        """Generate Log-Normal specific diagnostic plots"""
        if self.ols_model is None or self.test_predictions is None:
            logger.warning("Model not fitted or predictions not available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Get log-scale predictions and residuals
        X_test_const = sm.add_constant(self.X_test)
        log_sqrt_pred = self.ols_model.predict(X_test_const)
        y_test_log_sqrt = np.log(np.sqrt(self.y_test + 1e-10))
        residuals_log = y_test_log_sqrt - log_sqrt_pred
        
        # 1. Predicted vs Actual (original scale)
        axes[0, 0].scatter(self.y_test, self.test_predictions, alpha=0.5, s=10)
        max_val = max(self.y_test.max(), self.test_predictions.max())
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Actual Cost ($)')
        axes[0, 0].set_ylabel('Predicted Cost ($)')
        axes[0, 0].set_title('Predicted vs Actual (Original Scale)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals on log scale
        axes[0, 1].scatter(log_sqrt_pred, residuals_log, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Fitted log(?Y)')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Log-Scale Residuals vs Fitted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot of log-scale residuals
        stats.probplot(residuals_log, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot (Log Scale)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Histogram of log-scale residuals
        axes[1, 0].hist(residuals_log, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals (Log Scale)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Residual Distribution (Skewness: {self.skewness_log:.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Retransformation bias analysis
        # Compare naive retransformation vs smeared
        sqrt_pred_naive = np.exp(log_sqrt_pred)
        pred_naive = sqrt_pred_naive ** 2
        axes[1, 1].scatter(self.y_test, pred_naive, alpha=0.3, s=10, label='Naive')
        axes[1, 1].scatter(self.y_test, self.test_predictions, alpha=0.3, s=10, label='Smeared')
        axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        axes[1, 1].set_xlabel('Actual Cost ($)')
        axes[1, 1].set_ylabel('Predicted Cost ($)')
        axes[1, 1].set_title('Retransformation Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance by cost quartile
        quartiles = pd.qcut(self.y_test, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_errors = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            mask = quartiles == q
            errors = np.abs(self.y_test[mask] - self.test_predictions[mask])
            quartile_errors.append(errors)
        
        axes[1, 2].boxplot(quartile_errors, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        axes[1, 2].set_xlabel('Cost Quartile')
        axes[1, 2].set_ylabel('Absolute Error ($)')
        axes[1, 2].set_title('Error Distribution by Quartile')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "diagnostic_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Log-Normal diagnostic plots saved to {plot_file}")
    
    def plot_multiplicative_effects(self) -> None:
        """Generate multiplicative effects visualization"""
        if not self.coefficients:
            logger.warning("No coefficients available for plotting")
            return
        
        # Get significant coefficients
        sig_coef = {
            name: data for name, data in self.coefficients.items()
            if data['p_value'] < 0.05 and name != 'const'
        }
        
        if not sig_coef:
            logger.warning("No significant coefficients to plot")
            return
        
        # Sort by absolute effect
        sorted_items = sorted(
            sig_coef.items(),
            key=lambda x: abs(x[1]['multiplicative_effect']),
            reverse=True
        )[:15]  # Top 15
        
        names = [item[0] for item in sorted_items]
        effects = [item[1]['multiplicative_effect'] for item in sorted_items]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['green' if e > 0 else 'red' for e in effects]
        ax.barh(range(len(names)), effects, color=colors, alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Percentage Change in Cost (%)')
        ax.set_title('Top 15 Features by Multiplicative Effect\n(Log-Normal GLM)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_file = self.output_dir / "multiplicative_effects.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Multiplicative effects plot saved to {plot_file}")
    
    def run_complete_pipeline(self, 
                            fiscal_year_start: int = 2019,
                            fiscal_year_end: int = 2021,
                            perform_cv: bool = True) -> Dict[str, Any]:
        """Override to add Log-Normal specific diagnostics"""
        # Run base pipeline
        results = super().run_complete_pipeline(fiscal_year_start, fiscal_year_end, perform_cv)
        
        # Add Log-Normal specific diagnostics
        logger.info("Generating Log-Normal specific diagnostics...")
        self.plot_lognormal_diagnostics()
        self.plot_multiplicative_effects()
        
        # Add Log-Normal info to results
        results['lognormal_info'] = {
            'r2_log_scale': self.r2_log_scale if self.r2_log_scale else 0,
            'sigma_log': self.sigma_log if self.sigma_log else 0,
            'smearing_factor': self.smearing_factor if self.smearing_factor else 0,
            'skewness_reduction_pct': ((self.skewness_original - self.skewness_log) / 
                                       abs(self.skewness_original) * 100) if self.skewness_original and self.skewness_original != 0 else 0,
            'heteroscedasticity_pval': self.heteroscedasticity_pval if self.heteroscedasticity_pval else 0,
            'aic': self.aic if self.aic else 0,
            'bic': self.bic if self.bic else 0,
            'num_parameters': self.num_parameters if self.num_parameters else 0,
            'coefficients': self.coefficients if self.coefficients else {}
        }
        
        return results


def main():
    """Main execution function"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Initialize model
    model = Model6LogNormal()
    
    # Run pipeline
    results = model.run_complete_pipeline(
        fiscal_year_start=2023,
        fiscal_year_end=2024,
        perform_cv=True
    )
    
    print("\n" + "="*80)
    print("MODEL 6 LOG-NORMAL GLM EXECUTION COMPLETE")
    print("="*80)
    print("\nKey Advantages:")
    print("  ? No outlier removal (100% data retention)")
    print("  ? Natural handling of right-skewed costs via log transformation")
    print("  ? Multiplicative effects interpretation")
    print("  ? Built-in heteroscedasticity handling")
    print("  ? Duan's smearing estimator reduces retransformation bias")
    print("  ? Reduced residual skewness on log scale")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()
