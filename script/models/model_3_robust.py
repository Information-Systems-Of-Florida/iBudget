"""
model_3_robust.py
=================
Model 3: Robust Linear Regression with Huber M-estimation
Following the EXACT pattern from Models 1 and 2

Key features:
- Uses Model 1's exact feature specification (Model 5b features)
- Huber robust regression instead of OLS
- 100% data retention with automatic outlier downweighting
- Follows Models 1 & 2 architecture exactly
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
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

class Model3Robust(BaseiBudgetModel):
    """
    Model 3: Robust Linear Regression with Huber M-estimation
    
    Follows the EXACT pattern from Models 1 and 2:
    - Same feature preparation structure
    - Same initialization parameters
    - Same main() function pattern
    - Only difference: Huber regression in _fit_core()
    """
    
    def __init__(self,
                 use_sqrt_transform: bool = False,
                 use_outlier_removal: bool = False,  # Model 3 NEVER removes outliers
                 outlier_threshold: float = 1.645,
                 epsilon: float = 1.35,  # Huber-specific parameter
                 max_iter: int = 1000,    # Huber-specific parameter
                 random_seed: int = RANDOM_SEED,
                 log_suffix: Optional[str] = None,
                 **kwargs):
        """
        Initialize Model 3: Robust Linear Regression
        Following Model 1 & 2 pattern with additional Huber parameters
        
        Args:
            use_sqrt_transform: Apply sqrt transformation (default True like Model 1)
            use_outlier_removal: ALWAYS False for Model 3
            outlier_threshold: Kept for interface compatibility
            epsilon: Huber's tuning constant (1.35 = 95% efficiency)
            max_iter: Maximum iterations for convergence
            random_seed: Random seed for reproducibility
            log_suffix: Optional suffix for log files
        """
        # Determine transformation
        transformation = 'sqrt' if use_sqrt_transform else 'none'
        
        # Call parent class __init__
        super().__init__(
            model_id=3,
            model_name="Robust Linear Regression (Huber M-estimation)",
            transformation=transformation,
            use_outlier_removal=False,  # Model 3 NEVER removes outliers
            outlier_threshold=outlier_threshold,
            random_seed=random_seed,
            log_suffix=log_suffix
        )
        
        # Store Huber-specific parameters
        self.epsilon = epsilon
        self.max_iter = max_iter
        
        # Initialize model placeholder
        self.model = None
        # Feature scaler (fit inside each training fit/CV fold)
        self.scaler_ = None

        
        # Weight tracking (Model 3 specific)
        self.observation_weights = None
        self.weight_statistics = {}
        
        # Convergence tracking
        self.convergence_info = {
            'converged': False,
            'n_iterations': 0,
            'final_score': None
        }
        
        self.logger.info("Model 3: Robust Linear Regression initialized")
        self.logger.info(f"  Epsilon (Huber constant): {self.epsilon}")
        self.logger.info(f"  Max iterations: {self.max_iter}")
        self.logger.info(f"  Transformation: {transformation}")
        self.logger.info("  Outlier handling: Adaptive weights (no removal)")
        
        # Run complete pipeline (following Model 1 & 2 pattern)
        results = self.run_complete_pipeline(
            fiscal_year_start=2024,
            fiscal_year_end=2024,
            test_size=0.2,
            perform_cv=True,
            n_cv_folds=10
        )
        
        # Generate diagnostic plots
        self.generate_diagnostic_plots()
        
        # Log final summary (following Model 2 pattern)
        self.log_section("MODEL 3 FINAL SUMMARY", "=")
        
        self.logger.info("")
        self.logger.info("Performance Metrics:")
        self.logger.info(f"  Training R^2: {self.metrics.get('r2_train', 0):.4f}")
        self.logger.info(f"  Test R^2: {self.metrics.get('r2_test', 0):.4f}")
        self.logger.info(f"  RMSE: ${self.metrics.get('rmse_test', 0):,.2f}")
        self.logger.info(f"  MAE: ${self.metrics.get('mae_test', 0):,.2f}")
        if 'cv_mean' in self.metrics:
            self.logger.info(f"  CV R^2 (mean +- std): {self.metrics['cv_mean']:.4f} +- {self.metrics['cv_std']:.4f}")
        
        self.logger.info("")
        self.logger.info("Robust Regression Specific:")
        if self.weight_statistics:
            self.logger.info(f"  Mean weight: {self.weight_statistics['mean']:.4f}")
            self.logger.info(f"  Median weight: {self.weight_statistics['median']:.4f}")
            self.logger.info(f"  Full weight (>=0.99): {self.weight_statistics['full_weight_pct']:.1f}%")
            self.logger.info(f"  Downweighted: {self.weight_statistics['downweighted_count']} observations")
            self.logger.info(f"  Downweighted %: {self.weight_statistics['downweighted_pct']:.1f}%")
        
        self.logger.info("")
        self.logger.info("Data Utilization:")
        self.logger.info(f"  Training samples: {self.metrics.get('training_samples', 0)}")
        self.logger.info(f"  Test samples: {self.metrics.get('test_samples', 0)}")
        self.logger.info("  Outliers removed: 0 (100% data retention)")
        
        self.logger.info("")
        self.logger.info("Output:")
        self.logger.info(f"  Results: {self.output_dir_relative}")
        self.logger.info(f"  Plots: {self.output_dir_relative / 'diagnostic_plots.png'}")
        self.logger.info(f"  LaTeX: {self.output_dir_relative / f'model_{self.model_id}_renewcommands.tex'}")
        
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"MODEL {self.model_id} PIPELINE COMPLETE")
        self.logger.info("="*80)        

    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features following EXACT Model 1 pattern (Model 5b specification)
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
        
        # Use the base class method exactly like Model 1
        return self.prepare_features_from_spec(records, feature_config)
    
    def _fit_core(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Huber robust regression model
        This is the ONLY major difference from Model 1
        
        Args:
            X: Feature matrix (NO outliers removed - Model 3 uses all data)
            y: Target values (possibly sqrt-transformed)
        """
        self.log_section("FITTING HUBER ROBUST REGRESSION")
        
        # Standardize X for numerical stability (per fit / per CV fold)
        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)

        # Initialize Huber with small L2 (default), looser tol, more iters
        # NOTE: do NOT pass alpha=0.0 - keep default small L2 for stability
        self.model = HuberRegressor(
            epsilon=self.epsilon,
            max_iter=max(1000, int(self.max_iter)),  # ensure enough iterations
            tol=1e-4,
            fit_intercept=True, 
            warm_start=False,
        )

        # Fit the model on scaled features
        self.model.fit(X_scaled, y)

        
        # Store coefficients for reporting
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        self.scale_ = self.model.scale_  # Robust scale estimate
        
        # Calculate observation weights for transparency
        self._calculate_weights(X, y)
        
        # Log fitting summary
        self.logger.info(f"Huber regression fitted with {len(self.coefficients)} coefficients")
        self.logger.info(f"  Intercept: {self.intercept:.4f}")
        self.logger.info(f"  Scale estimate (MAD): {self.scale_:.4f}")
        self.logger.info(f"  Iterations: {self.model.n_iter_}")
        self.logger.info(f"  Converged: {self.model.n_iter_ < self.max_iter}")
        
        # Update convergence info
        self.convergence_info['converged'] = self.model.n_iter_ < self.max_iter
        self.convergence_info['n_iterations'] = self.model.n_iter_
        self.convergence_info['final_score'] = self.model.score(X_scaled, y)

        
        # Log weight statistics
        if self.observation_weights is not None:
            self._calculate_weight_statistics()
            self.logger.info("\nWeight Statistics:")
            self.logger.info(f"  Mean weight: {self.weight_statistics['mean']:.4f}")
            self.logger.info(f"  Median weight: {self.weight_statistics['median']:.4f}")
            self.logger.info(f"  Min weight: {self.weight_statistics['min']:.4f}")
            self.logger.info(f"  Max weight: {self.weight_statistics['max']:.4f}")
            self.logger.info(f"  Full weight (>=0.99): {self.weight_statistics['full_weight_pct']:.1f}%")
            self.logger.info(f"  Downweighted (<0.99): {self.weight_statistics['downweighted_count']} observations")
        
        # Log coefficient summary (following Model 1 pattern)
        self.logger.info("\nCoefficient summary:")
        for name, coef in zip(self.feature_names, self.coefficients):
            self.logger.info(f"  {name}: {coef:.4f}")
    
    def _predict_core(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using robust regression model
        Following Model 1 pattern
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions in training scale (sqrt if transformed)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if self.scaler_ is None:
            raise ValueError("Scaler not fitted; fit the model before predicting.")
        X_scaled = self.scaler_.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_original(self, X: np.ndarray) -> np.ndarray:
        """
        Override base-class hook for CV and evaluation.
        Following Model 1 pattern exactly
        """
        y_pred_fitted = self._predict_core(X)  # predictions in fitted scale
        y_pred_original = self.inverse_transformation(y_pred_fitted)  # convert to dollars
        return np.maximum(0.0, y_pred_original)
    
    def _calculate_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate observation weights based on Huber loss function
        Model 3 specific functionality
        """
        if self.model is None:
            return
        
        # Get predictions
        X_scaled = self.scaler_.transform(X) if self.scaler_ is not None else X
        y_pred = self.model.predict(X_scaled)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Standardize residuals using robust scale
        standardized_residuals = residuals / self.scale_
        
        # Calculate Huber weights
        weights = np.ones(len(residuals))
        outlier_mask = np.abs(standardized_residuals) > self.epsilon
        weights[outlier_mask] = self.epsilon / np.abs(standardized_residuals[outlier_mask])
        
        self.observation_weights = weights
    
    def _calculate_weight_statistics(self) -> None:
        """Calculate comprehensive weight statistics"""
        if self.observation_weights is None:
            return
        
        weights = self.observation_weights
        
        self.weight_statistics = {
            'mean': np.mean(weights),
            'median': np.median(weights),
            'std': np.std(weights),
            'min': np.min(weights),
            'max': np.max(weights),
            'full_weight_count': np.sum(weights >= 0.99),
            'full_weight_pct': 100 * np.mean(weights >= 0.99),
            'downweighted_count': np.sum(weights < 0.99),
            'downweighted_pct': 100 * np.mean(weights < 0.99),
            'severely_downweighted': np.sum(weights < 0.5),
            'total_observations': len(weights)
        }
    
    def generate_diagnostic_plots(self) -> None:
        """
        Generate diagnostic plots following Model 1 & 2 pattern
        Includes Model 3 specific weight distribution
        """
        if self.model is None or self.X_test is None:
            self.logger.warning("Model not fitted or no test data for plots")
            return
        
        self.log_section("GENERATING DIAGNOSTIC PLOTS")
        
        # Get test predictions
        test_predictions = self.predict(self.X_test)
        test_actual = self.y_test
        
        # Calculate training weights if needed
        if self.observation_weights is None and self.X_train is not None:
            self._calculate_weights(self.X_train, self.apply_transformation(self.y_train))
        
        # Create figure with 6 subplots (following Models 1 & 2 pattern)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model 3: Robust Linear Regression Diagnostic Plots', 
                     fontsize=14, fontweight='bold')
        
        # 1. Predicted vs Actual
        ax1 = axes[0, 0]
        ax1.scatter(test_actual, test_predictions, alpha=0.5, s=10)
        ax1.plot([test_actual.min(), test_actual.max()], 
                 [test_actual.min(), test_actual.max()], 
                 'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('Actual Cost ($)')
        ax1.set_ylabel('Predicted Cost ($)')
        ax1.set_title('(A) Predicted vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        ax2 = axes[0, 1]
        residuals = test_actual - test_predictions
        ax2.scatter(test_predictions, residuals, alpha=0.5, s=10)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Cost ($)')
        ax2.set_ylabel('Residuals ($)')
        ax2.set_title('(B) Residuals vs Predicted')
        ax2.grid(True, alpha=0.3)
        
        # 3. Weight Distribution (Model 3 SPECIFIC)
        ax3 = axes[0, 2]
        if self.observation_weights is not None:
            weights = self.observation_weights
            ax3.hist(weights, bins=50, edgecolor='black', alpha=0.7)
            ax3.axvline(x=0.99, color='r', linestyle='--', linewidth=2, 
                       label=f'Full weight ({self.weight_statistics["full_weight_pct"]:.1f}%)')
            ax3.set_xlabel('Observation Weight')
            ax3.set_ylabel('Frequency')
            ax3.set_title('(C) Weight Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No weights available', ha='center', va='center')
            ax3.set_title('(C) Weight Distribution')
        
        # 4. Q-Q Plot
        ax4 = axes[1, 0]
        standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
        stats.probplot(standardized_residuals, dist="norm", plot=ax4)
        ax4.set_title('(D) Q-Q Plot')
        ax4.grid(True, alpha=0.3)
        
        # 5. Residual Distribution
        ax5 = axes[1, 1]
        ax5.hist(residuals, bins=50, edgecolor='black', alpha=0.7, density=True)
        
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax5.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')
        ax5.set_xlabel('Residuals ($)')
        ax5.set_ylabel('Density')
        ax5.set_title('(E) Residual Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance by Cost Quartile
        ax6 = axes[1, 2]
        quartiles = np.percentile(test_actual, [25, 50, 75])
        q_labels = ['Q1\n(<$25K)', 'Q2\n($25-50K)', 'Q3\n($50-75K)', 'Q4\n(>$75K)']
        q_r2 = []
        
        for i in range(4):
            if i == 0:
                mask = test_actual <= quartiles[0]
            elif i < 3:
                mask = (test_actual > quartiles[i-1]) & (test_actual <= quartiles[i])
            else:
                mask = test_actual > quartiles[2]
            
            if mask.sum() > 10:
                q_r2.append(r2_score(test_actual[mask], test_predictions[mask]))
            else:
                q_r2.append(0)
        
        ax6.bar(q_labels, q_r2, edgecolor='black', alpha=0.7)
        ax6.set_ylabel('R^2 Score')
        ax6.set_title('(F) Performance by Cost Quartile')
        ax6.set_ylim([0, 1])
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add overall R^2 line
        overall_r2 = r2_score(test_actual, test_predictions)
        ax6.axhline(y=overall_r2, color='r', linestyle='--', linewidth=2, 
                   label=f'Overall R^2 = {overall_r2:.3f}')
        ax6.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'diagnostic_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Diagnostic plots saved to {plot_path}")
    
    def generate_latex_commands(self) -> None:
        """
        Generate LaTeX commands following Models 1 & 2 pattern
        CRITICAL: Must call super() FIRST, then append model-specific commands
        """
        # STEP 1: Call parent FIRST to generate standard commands
        super().generate_latex_commands()
        
        # STEP 2: Append Model 3 specific commands
        self.logger.info(f"Adding Model {self.model_id} specific LaTeX commands...")
        
        model_word = self._number_to_word(self.model_id)
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append to newcommands file
        with open(newcommands_file, 'a') as f:
            f.write("\n% Model 3 Robust Regression Specific Commands\n")
            f.write(f"\\newcommand{{\\Model{model_word}Epsilon}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}ScaleEstimate}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}NumIterations}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Converged}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MeanWeight}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MedianWeight}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MinWeight}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}FullWeightPct}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}OutliersDetected}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}OutlierPercentage}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Parameters}}{{\\WarningRunPipeline}}\n")
        
        # Append actual values to renewcommands file
        with open(renewcommands_file, 'a') as f:
            f.write("\n% Model 3 Robust Regression Specific Values\n")
            
            # Robust regression specific metrics
            if self.model is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}Epsilon}}{{{self.epsilon:.2f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}ScaleEstimate}}{{{self.scale_:.4f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}NumIterations}}{{{self.model.n_iter_}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Converged}}{{{('Yes' if self.convergence_info['converged'] else 'No')}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Parameters}}{{{len(self.coefficients) + 1}}}\n")
            
            # Weight statistics
            if self.weight_statistics:
                f.write(f"\\renewcommand{{\\Model{model_word}MeanWeight}}{{{self.weight_statistics['mean']:.4f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}MedianWeight}}{{{self.weight_statistics['median']:.4f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}MinWeight}}{{{self.weight_statistics['min']:.4f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}FullWeightPct}}{{{self.weight_statistics['full_weight_pct']:.1f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OutliersDetected}}{{{self.weight_statistics['downweighted_count']}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OutlierPercentage}}{{{self.weight_statistics['downweighted_pct']:.1f}}}\n")
            
            # Prediction accuracy bands
            if self.X_test is not None and self.y_test is not None:
                test_pred = self.predict(self.X_test)
                errors = np.abs(self.y_test - test_pred)
                
                f.write(f"\\renewcommand{{\\Model{model_word}WithinOneK}}{{{100 * np.mean(errors <= 1000):.1f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}WithinTwoK}}{{{100 * np.mean(errors <= 2000):.1f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}WithinFiveK}}{{{100 * np.mean(errors <= 5000):.1f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}WithinTenK}}{{{100 * np.mean(errors <= 10000):.1f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}WithinTwentyK}}{{{100 * np.mean(errors <= 20000):.1f}}}\n")
        
        self.logger.info("Model 3 specific commands appended to both files")


def main():
    """
    Run Model 3 Robust Linear Regression implementation
    """
    logger.info("="*80)
    logger.info("MODEL 3: ROBUST LINEAR REGRESSION (HUBER M-ESTIMATION)")
    logger.info("="*80)
    
    # Initialize model with explicit parameters (following Model 1 & 2 pattern)
    use_sqrt = False                 
    use_outlier = False
    suffix = 'Sqrt_' + str(use_sqrt) + '_Outliers_' + str(use_outlier)
    model = Model3Robust(
        use_sqrt_transform=use_sqrt,        # Use sqrt like Model 1/5b
        use_outlier_removal=use_outlier,    # Model 3 NEVER removes outliers
        outlier_threshold=1.645,            # Kept for compatibility
        epsilon=1.35,                       # Huber constant (95% efficiency)
        max_iter=1000,                      # Maximum iterations
        random_seed=42,                     # For reproducibility
        log_suffix=suffix                   # Clear log suffix
    )

if __name__ == "__main__":
    main()