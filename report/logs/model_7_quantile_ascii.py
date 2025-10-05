"""
model_7_quantile.py
===================
Model 7: Quantile Regression with Multiple Quantile Estimation
?? RESEARCH ONLY - NOT REGULATORY COMPLIANT ??

CRITICAL WARNING:
This model produces DISTRIBUTIONS rather than single allocations.
It violates F.S. 393.0662 and F.A.C. 65G-4.0214 which require 
deterministic budget amounts. Suitable for research and risk analysis only.

Key features:
- Quantile regression at ? = {0.10, 0.25, 0.50, 0.75, 0.90}
- Median (? = 0.50) as primary model for single allocation
- Complete robustness to outliers (50% breakdown point)
- Asymmetric loss function (check function)
- Natural prediction intervals from quantile spread
- Square-root transformation of costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold, train_test_split
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

class Model7QuantileRegression(BaseiBudgetModel):
    """
    Model 7: Quantile Regression
    
    ?? REGULATORY WARNING: NOT COMPLIANT WITH F.S. 393.0662 ??
    
    This model produces a distribution of potential allocations rather than
    a single deterministic amount. While statistically sophisticated, it cannot
    be used for production budget allocation under current Florida law.
    
    Key features:
    - Quantile regression at multiple quantiles
    - Median (? = 0.50) as primary estimate
    - Complete outlier robustness (100% data inclusion)
    - Natural prediction intervals
    - Square-root transformation like Model 1
    - Research and validation tool only
    """
    
    def __init__(self):
        """Initialize Model 7"""
        super().__init__(model_id=7, model_name="Quantile Regression")
        
        # Quantile regression specific parameters
        self.quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]  # Multiple quantiles
        self.primary_quantile = 0.50  # Median as primary estimate
        self.models = {}  # Dictionary of models, one per quantile
        self.transformation = "sqrt"  # Square-root like Model 1
        
        # Regulatory compliance
        self.regulatory_compliant = "No"  # CRITICAL: Not compliant
        self.regulatory_warning = "Produces distributions, not single allocations. Violates F.S. 393.0662."
        
        # Performance tracking
        self.quantile_performance = {}  # R^2 for each quantile
        self.prediction_intervals = {}  # Width of intervals
        self.quantile_spread = None  # Q90/Q10 ratio
        self.monotonicity_violations = 0  # Quantile crossing issues
        
        logger.info("="*80)
        logger.info("MODEL 7: QUANTILE REGRESSION - RESEARCH ONLY")
        logger.info("??  NOT REGULATORY COMPLIANT ??")
        logger.info("Produces distributions, not single allocations")
        logger.info("="*80)
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List, List]:
        """
        Override to handle boolean conversion for all records
        
        Args:
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_records, test_records)
        """
        # Convert boolean fields to integers for all records
        for record in self.all_records:
            record.late_entry = int(record.late_entry) if isinstance(record.late_entry, bool) else record.late_entry
            record.early_exit = int(record.early_exit) if isinstance(record.early_exit, bool) else record.early_exit
            record.has_multiple_qsi = int(record.has_multiple_qsi) if isinstance(record.has_multiple_qsi, bool) else record.has_multiple_qsi
            record.usable = int(record.usable) if isinstance(record.usable, bool) else record.usable
        
        # Use parent's split method
        return super().split_data(test_size, random_state)
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix using ONLY robust features from FeatureSelection.txt
        
        Based on mutual information analysis, using consistently important features:
        - RESIDENCETYPE/Living Setting indicators (highest MI: 0.252-0.272)
        - BSum (behavioral summary, MI: 0.113-0.137)
        - LOSRI/OLEVEL (support levels, MI: 0.113-0.131)
        - Top QSI questions (Q26, Q36, Q20, Q27, etc.)
        - Age group indicators
        
        Returns square-root transformed costs as target.
        """
        if not records:
            logger.warning("No records provided to prepare_features")
            return np.array([]), np.array([])
        
        # Build feature matrix
        X_list = []
        y_list = []
        
        for record in records:
            features = []
            
            # Living Setting (5 indicators, FH as reference)
            features.append(1 if record.living_setting == 'ILSL' else 0)
            features.append(1 if record.living_setting == 'RH1' else 0)
            features.append(1 if record.living_setting == 'RH2' else 0)
            features.append(1 if record.living_setting == 'RH3' else 0)
            features.append(1 if record.living_setting == 'RH4' else 0)
            
            # Age Group (2 indicators, Age3_20 as reference)
            features.append(1 if record.age_group == 'Age21_30' else 0)
            features.append(1 if record.age_group == 'Age31Plus' else 0)
            
            # Summary Scores (highly predictive)
            features.append(record.bsum)  # Behavioral sum
            features.append(record.fsum)  # Functional sum
            
            # Support Levels
            features.append(getattr(record, 'losri', 0))
            features.append(getattr(record, 'olevel', 0))
            features.append(getattr(record, 'blevel', 0))
            features.append(getattr(record, 'flevel', 0))
            
            # Top Individual QSI Questions (based on MI analysis)
            features.append(record.q16)
            features.append(record.q18)
            features.append(record.q20)
            features.append(record.q21)
            features.append(record.q23)
            features.append(record.q26)  # High MI score
            features.append(record.q27)
            features.append(record.q28)
            features.append(getattr(record, 'q33', 0))
            features.append(getattr(record, 'q34', 0))
            features.append(record.q36)  # High MI score
            features.append(getattr(record, 'q43', 0))
            
            X_list.append(features)
            y_list.append(record.total_cost)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Store feature names for interpretability
        self.feature_names = [
            'ILSL', 'RH1', 'RH2', 'RH3', 'RH4',  # Living settings
            'Age21_30', 'Age31Plus',  # Age groups
            'BSum', 'FSum',  # Summary scores
            'LOSRI', 'OLEVEL', 'BLEVEL', 'FLEVEL',  # Levels
            'Q16', 'Q18', 'Q20', 'Q21', 'Q23', 'Q26', 'Q27', 'Q28',  # QSI questions
            'Q33', 'Q34', 'Q36', 'Q43'
        ]
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Using {len(self.feature_names)} robust features")
        
        if len(y) > 0:
            logger.info(f"Cost range: ${y.min():.2f} to ${y.max():.2f}")
            
            # Apply square-root transformation to target
            y_transformed = np.sqrt(y)
            logger.info(f"Transformed target range: {y_transformed.min():.2f} to {y_transformed.max():.2f}")
        else:
            logger.warning("No valid costs found")
            y_transformed = np.array([])
        
        return X, y_transformed
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit quantile regression models at multiple quantiles
        
        Uses sklearn.linear_model.QuantileRegressor with highs-ds solver
        for robust optimization of the asymmetric check function.
        
        Args:
            X_train: Training features
            y_train: Square-root transformed costs
        """
        logger.info("="*80)
        logger.info("FITTING QUANTILE REGRESSION MODELS")
        logger.info("="*80)
        
        # Fit separate model for each quantile
        for tau in self.quantiles:
            logger.info(f"\nFitting quantile ? = {tau:.2f}...")
            
            model = QuantileRegressor(
                quantile=tau,
                alpha=0.0,  # No regularization
                solver='highs-ds',  # Robust solver for linear programming
                solver_options={'max_iter': 10000}
            )
            
            model.fit(X_train, y_train)
            self.models[tau] = model
            
            # Calculate in-sample performance
            y_pred = model.predict(X_train)
            
            # Pseudo R^2 for quantile regression (different from OLS R^2)
            # Based on ratio of check function values
            residuals = y_train - y_pred
            check_loss = np.sum(np.where(residuals >= 0, 
                                        tau * residuals, 
                                        (tau - 1) * residuals))
            
            # Null model: predict ?-th quantile
            q_null = np.quantile(y_train, tau)
            null_residuals = y_train - q_null
            null_loss = np.sum(np.where(null_residuals >= 0,
                                       tau * null_residuals,
                                       (tau - 1) * null_residuals))
            
            pseudo_r2 = 1 - (check_loss / null_loss) if null_loss > 0 else 0
            
            self.quantile_performance[tau] = {
                'pseudo_r2': pseudo_r2,
                'check_loss': check_loss,
                'coefficients': model.coef_.tolist(),
                'intercept': model.intercept_
            }
            
            logger.info(f"  Pseudo-R^2 = {pseudo_r2:.4f}")
            logger.info(f"  Check function loss = {check_loss:.2f}")
        
        # Set primary model (median)
        self.model = self.models[self.primary_quantile]
        
        logger.info("\n" + "="*80)
        logger.info(f"PRIMARY MODEL: Median Regression (? = {self.primary_quantile})")
        logger.info("="*80)
        
        # Calculate quantile spread
        self.quantile_spread = self.quantile_performance[0.90]['check_loss'] / max(self.quantile_performance[0.10]['check_loss'], 1e-6)
        logger.info(f"Quantile spread ratio (Q90/Q10): {self.quantile_spread:.2f}")
    
    def predict(self, X: np.ndarray, quantile: Optional[float] = None) -> np.ndarray:
        """
        Generate predictions at specified quantile or primary quantile
        
        Args:
            X: Feature matrix
            quantile: Which quantile to predict (default: median)
            
        Returns:
            Predictions in original dollar scale
        """
        if quantile is None:
            quantile = self.primary_quantile
        
        if quantile not in self.models:
            raise ValueError(f"Model not fitted for quantile {quantile}")
        
        # Predict in transformed space
        y_pred_sqrt = self.models[quantile].predict(X)
        
        # Back-transform to dollar scale
        y_pred = y_pred_sqrt ** 2
        
        return y_pred
    
    def predict_interval(self, X: np.ndarray, lower_q: float = 0.10, upper_q: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for quantile regression
        
        Extends base class metrics with quantile-specific measures.
        """
        # Get base metrics (uses median predictions)
        metrics = super().calculate_metrics()
        
        # Add quantile-specific metrics
        if self.y_test_pred is not None and self.X_test is not None:
            # Calculate prediction interval width
            median, lower, upper = self.predict_interval(self.X_test)
            interval_widths = upper - lower
            avg_interval_width = np.mean(interval_widths)
            
            metrics['prediction_interval_width'] = float(avg_interval_width)
            metrics['quantile_spread'] = float(self.quantile_spread) if self.quantile_spread else 0
            metrics['quantile_monotonicity'] = 100.0 * (1 - self.monotonicity_violations / len(self.y_test_pred)) if len(self.y_test_pred) > 0 else 100.0
            
            # Performance at each quantile
            for tau in self.quantiles:
                if tau in self.quantile_performance:
                    metrics[f'quantile_{int(tau*100)}_r2'] = float(self.quantile_performance[tau]['pseudo_r2'])
            
            # Regulatory compliance (CRITICAL)
            metrics['regulatory_compliant'] = self.regulatory_compliant
            metrics['regulatory_warning'] = self.regulatory_warning
        
        return metrics
    
    def plot_quantile_diagnostics(self):
        """Generate quantile regression specific diagnostic plots"""
        if self.y_test is None or self.y_test_pred is None:
            logger.warning("No test predictions available for plotting")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model 7: Quantile Regression Diagnostics - RESEARCH ONLY', 
                     fontsize=14, fontweight='bold', color='red')
        
        # 1. Predicted vs Actual (Median)
        ax = axes[0, 0]
        ax.scatter(self.y_test, self.y_test_pred, alpha=0.5, s=20)
        min_val = min(self.y_test.min(), self.y_test_pred.min())
        max_val = max(self.y_test.max(), self.y_test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Cost ($)')
        ax.set_ylabel('Predicted Cost - Median ($)')
        ax.set_title('Predicted vs Actual (Median)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Fan Chart - Prediction Intervals
        ax = axes[0, 1]
        median, lower, upper = self.predict_interval(self.X_test)
        
        # Sort by median for cleaner visualization
        sort_idx = np.argsort(median)
        x_plot = np.arange(len(median))
        
        ax.fill_between(x_plot, lower[sort_idx], upper[sort_idx], alpha=0.3, label='80% Interval (Q10-Q90)')
        ax.plot(x_plot, median[sort_idx], 'b-', linewidth=2, label='Median Prediction')
        ax.scatter(x_plot, self.y_test[sort_idx], alpha=0.3, s=10, c='red', label='Actual')
        ax.set_xlabel('Consumer (sorted by predicted median)')
        ax.set_ylabel('Cost ($)')
        ax.set_title('Fan Chart - Prediction Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Residual Distribution
        ax = axes[0, 2]
        residuals = self.y_test - self.y_test_pred
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual ($)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Residual Distribution (Median)\nMean: ${np.mean(residuals):.0f}')
        ax.grid(True, alpha=0.3)
        
        # 4. Quantile Performance Comparison
        ax = axes[1, 0]
        quantiles = list(self.quantile_performance.keys())
        pseudo_r2s = [self.quantile_performance[q]['pseudo_r2'] for q in quantiles]
        
        bars = ax.bar([f'{int(q*100)}%' for q in quantiles], pseudo_r2s, 
                     color=['lightblue' if q != 0.50 else 'darkblue' for q in quantiles])
        ax.set_xlabel('Quantile (?)')
        ax.set_ylabel('Pseudo-R^2')
        ax.set_title('Performance Across Quantiles')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight median
        for i, q in enumerate(quantiles):
            if q == 0.50:
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(3)
        
        # 5. Interval Width by Cost Level
        ax = axes[1, 1]
        interval_widths = upper - lower
        
        ax.scatter(median, interval_widths, alpha=0.5, s=20)
        ax.set_xlabel('Predicted Median Cost ($)')
        ax.set_ylabel('Interval Width ($)')
        ax.set_title('Prediction Uncertainty by Cost Level')
        ax.grid(True, alpha=0.3)
        
        # 6. Q-Q Plot for Median
        ax = axes[1, 2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Median Residuals)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "diagnostic_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plots saved to {plot_file}")
    
    def plot_coefficient_comparison(self):
        """Plot coefficient values across quantiles"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Extract coefficients for each quantile
        n_features = len(self.feature_names)
        quantiles = sorted(self.models.keys())
        
        coef_matrix = np.zeros((n_features, len(quantiles)))
        for i, tau in enumerate(quantiles):
            coef_matrix[:, i] = self.models[tau].coef_
        
        # Create heatmap
        im = ax.imshow(coef_matrix, aspect='auto', cmap='RdBu_r', 
                      vmin=-np.abs(coef_matrix).max(), vmax=np.abs(coef_matrix).max())
        
        # Set ticks
        ax.set_xticks(np.arange(len(quantiles)))
        ax.set_xticklabels([f'?={q:.2f}' for q in quantiles])
        ax.set_yticks(np.arange(n_features))
        ax.set_yticklabels(self.feature_names)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Coefficient Value')
        
        ax.set_title('Model 7: Coefficient Comparison Across Quantiles', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Quantile (?)')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        # Save
        plot_file = self.output_dir / "quantile_coefficients.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Coefficient comparison plot saved to {plot_file}")
    
    def run_complete_pipeline(self, 
                            fiscal_year_start: int = 2023,
                            fiscal_year_end: int = 2024,
                            perform_cv: bool = True,
                            test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run complete Model 7 pipeline with regulatory warnings
        
        Overrides base class to add quantile-specific diagnostics
        
        Args:
            fiscal_year_start: Start year for data (default: 2023)
            fiscal_year_end: End year for data (default: 2024)
            perform_cv: Whether to perform cross-validation
            test_size: Proportion for test set (default: 0.2)
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL 7: QUANTILE REGRESSION PIPELINE")
        logger.info("??  RESEARCH ONLY - NOT REGULATORY COMPLIANT ??")
        logger.info("="*80)
        
        # Call base class pipeline with explicit parameters
        # This ensures proper data loading and all base functionality
        base_results = super().run_complete_pipeline(
            fiscal_year_start=fiscal_year_start,
            fiscal_year_end=fiscal_year_end,
            perform_cv=perform_cv
        )
        
        # Collect results
        results = {
            'metrics': self.metrics,
            'subgroup_metrics': self.subgroup_metrics,
            'population_scenarios': self.population_scenarios
        }
        
        # Add quantile-specific diagnostics
        logger.info("\nGenerating quantile-specific diagnostics...")
        self.plot_quantile_diagnostics()
        self.plot_coefficient_comparison()
        
        # Add quantile info to results
        results['quantile_info'] = {
            'quantiles': self.quantiles,
            'primary_quantile': self.primary_quantile,
            'quantile_performance': self.quantile_performance,
            'quantile_spread': float(self.quantile_spread) if self.quantile_spread else 0,
            'monotonicity_pct': float(100.0 * (1 - self.monotonicity_violations / len(self.y_test_pred))) if self.y_test_pred is not None else 100.0,
            'regulatory_compliant': self.regulatory_compliant,
            'regulatory_warning': self.regulatory_warning
        }
        
        # Emphasize non-compliance
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


def main():
    """Main execution function"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Initialize model
    model = Model7QuantileRegression()
    
    # Run pipeline with FY2023-2024 data
    logger.info("\n" + "="*80)
    logger.info("STARTING MODEL 7 EXECUTION")
    logger.info("Using FY2023-2024 data as specified")
    logger.info("="*80)
    
    results = model.run_complete_pipeline(
        fiscal_year_start=2023,
        fiscal_year_end=2024,
        perform_cv=True
    )
    
    print("\n" + "="*80)
    print("MODEL 7 QUANTILE REGRESSION EXECUTION COMPLETE")
    print("="*80)
    print("\n??  CRITICAL REGULATORY WARNING ??")
    print("="*80)
    print("Status: NOT COMPLIANT with F.S. 393.0662")
    print("Reason: Produces distributions, not single allocations")
    print("Usage: RESEARCH AND VALIDATION ONLY")
    print("="*80)
    print("\nKey Technical Advantages:")
    print("  ? Complete outlier robustness (50% breakdown point)")
    print("  ? Natural prediction intervals from quantile spread")
    print("  ? No distributional assumptions required")
    print("  ? Distribution-free inference")
    print("  ? 100% data inclusion")
    print("\nFatal Regulatory Flaw:")
    print("  ? Cannot produce required single allocation amount")
    print("  ? Incompatible with appeals process")
    print("  ? Would require complete legal framework overhaul")
    print("\nRecommendation:")
    print("  FOR RESEARCH: Use to understand uncertainty and validate other models")
    print("  FOR PRODUCTION: Use Model 3 (Huber Regression) instead")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()
