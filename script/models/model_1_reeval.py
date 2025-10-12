"""
model_1_current.py
==================
Model 1: Re-evaluation of Model 5b (Tao & Niu 2015) with 2024 Data

This model is a FAITHFUL replication of Model 5b using 2024 data.
It uses the EXACT same 21 features, same transformations, and same
outlier detection method as the original Model 5b.

Model 5b Specification (Tao & Niu 2015):
- 21 features exactly (no additions, no changes)
- Square-root transformation of total cost
- Studentized residuals outlier detection (|t_i| >= 1.645)
- Ordinary Least Squares regression

Features (21 total):
    1-5:   Living Settings (LiveILSL, LiveRH1, LiveRH2, LiveRH3, LiveRH4)
    6-7:   Age Groups (Age21_30, Age31Plus)
    8:     BSum (behavioral sum)
    9-11:  Interaction Terms (FHFSum, SLFSum, SLBSum) <- CRITICAL
    12-21: QSI Questions (Q16, Q18, Q20, Q21, Q23, Q28, Q33, Q34, Q36, Q43)

Purpose: Direct comparison of Model 5b performance with 9 years of new data
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import sys

# Import base class
sys.path.append(str(Path(__file__).parent))
from base_model import BaseiBudgetModel, ConsumerRecord


class Model1Linear(BaseiBudgetModel):
    """
    Model 1: Re-evaluation of Model 5b (Tao & Niu 2015) with 2024 Data
    
    This model EXACTLY replicates Model 5b's specification:
    - 21 features (same as Model 5b)
    - Square-root transformation
    - Studentized residuals outlier detection
    - Ordinary Least Squares regression
    
    Purpose: Answer the question - "Does Model 5b remain effective with 2024 data?"
    """
    
    # Model 5b historical benchmarks (from Tao & Niu 2015)
    MODEL_5B_R2_2015 = 0.7998
    MODEL_5B_SBC_2015 = 159394.3
    MODEL_5B_RMSE_2015 = 30.82  # In sqrt-transformed scale
    MODEL_5B_OUTLIER_THRESHOLD = 1.645
    MODEL_5B_OUTLIER_PCT_2015 = 9.40
    
    def __init__(self,  
             use_sqrt_transform: bool = True, # This is the default value. Any passed value overrides it
             use_outlier_removal: bool = True,# This is the default value. Any passed value overrides it
             outlier_threshold: float = 1.645,  # This is the default value. Any passed value overrides it
             feature_config: Optional[Dict[str, Any]] = None,  
             random_seed: int = 42,
            log_suffix: Optional[str] = None
             ):
        """
        Initialize Model 1 Linear
        
        Args:
            use_sqrt_transform: Apply sqrt transformation (Model 5b default)
            use_outlier_removal: Remove outliers using studentized residuals
            outlier_threshold: Threshold for outlier removal (1.645 = ~10%)
            feature_config: Optional feature configuration from pipeline  # ADD THIS
            random_seed: Random seed for reproducibility
            log_suffix: Optional suffix for log files 
        """
        super().__init__(
            model_id=1,
            model_name="Model 5b Re-Evaluation (2024 Data)",
            use_outlier_removal=use_outlier_removal,
            outlier_threshold=outlier_threshold,
            transformation='sqrt' if use_sqrt_transform else 'none',
            random_seed=42,
            log_suffix=log_suffix
        )
        
        # Store feature configuration  
        if feature_config: 
            self.feature_config = feature_config
            
        # Store transform flag for reporting
        self.use_sqrt_transform = use_sqrt_transform
        
        # Will be set after fitting
        self.coefficients = None
        self.intercept = None
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Model 1 feature preparation using generic configuration approach
        Matches the original Model 5b specification exactly
        """
                
        # If feature_config provided from pipeline, use it
        if hasattr(self, 'feature_config') and self.feature_config is not None:
            return self.prepare_features_from_spec(records, self.feature_config)
            
        # Model 1 uses exactly 10 QSI items from Model 5b
        model_5b_qsi = [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]
        
        feature_config = {
            'categorical': {
                'living_setting': {
                    'categories': ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4'],
                    'reference': 'FH'  # FH is reference, not included
                }
            },
            'binary': {
                'Age21_30': lambda r: 21 <= r.age <= 30,
                'Age31Plus': lambda r: r.age > 30
                # Age3_20 is reference, not included
            },
            'numeric': ['bsum'],  # Only BSum, not FSum or PSum (per Model 5b)
            'interactions': [
                # FH x FSum interaction
                ('FHFSum', lambda r: (1 if r.living_setting == 'FH' else 0) * float(r.fsum)),
                # Supported Living x FSum interaction  
                ('SLFSum', lambda r: (1 if r.living_setting in ['RH1','RH2','RH3','RH4'] else 0) * float(r.fsum)),
                # Supported Living x BSum interaction
                ('SLBSum', lambda r: (1 if r.living_setting in ['RH1','RH2','RH3','RH4'] else 0) * float(r.bsum))
            ],
            'qsi': model_5b_qsi  # Exactly 10 QSI items
        }
        
        # Use the generic preparation method from base class
        #X, feature_names = self.prepare_features_from_spec(records, feature_config)
        #return X, feature_names
        
        return self.prepare_features_from_spec(records, feature_config)
    
    def _fit_core(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Core OLS fitting (data already cleaned and transformed by base class)
        
        Args:
            X: Feature matrix (outliers removed)
            y: Target values (transformed to sqrt scale)
        """
        from sklearn.linear_model import LinearRegression
        
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(X, y)
        
        # Store coefficients for reporting
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
        self.logger.info(f"OLS model fitted with {len(self.coefficients)} coefficients")
        self.logger.info(f"Intercept: {self.intercept:.4f}")
        
        # Log coefficient summary
        self.logger.info("Coefficient summary:")
        for name, coef in zip(self.feature_names, self.coefficients):
            self.logger.info(f"  {name}: {coef:.4f}")
    
    def _predict_core(self, X: np.ndarray) -> np.ndarray:
        """
        Core prediction (returns in fitted scale)
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions in transformed scale (base class will inverse transform)
        """
        return self.model.predict(X)

    def predict_original(self, X: np.ndarray) -> np.ndarray:
        """
        Override base-class hook for CV and evaluation.
        Model 1's _predict_core() outputs predictions on the fitted sqrt-cost scale,
        so we must inverse-transform them here to return dollar-scale predictions.
        """
        y_pred_fitted = self._predict_core(X)           # predictions on sqrt-scale
        y_pred_original = self.inverse_transformation(y_pred_fitted)  # square -> dollars
        return np.maximum(0.0, y_pred_original)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate metrics with Model 5b comparison"""
        metrics = super().calculate_metrics()
        
        # Add Model 5b benchmarks
        metrics['model_5b_r2_2015'] = self.MODEL_5B_R2_2015
        metrics['model_5b_sbc_2015'] = self.MODEL_5B_SBC_2015
        metrics['model_5b_rmse_2015'] = self.MODEL_5B_RMSE_2015
        metrics['model_5b_outlier_pct_2015'] = self.MODEL_5B_OUTLIER_PCT_2015
        
        # Calculate deltas
        metrics['r2_delta_from_2015'] = metrics.get('r2_test', 0) - self.MODEL_5B_R2_2015
        
        # Calculate SBC (Schwarz Bayesian Criterion) for model selection
        if hasattr(self, 'model') and self.model is not None and self.X_train is not None:
            # Use only the data that wasn't removed as outliers
            if self.use_outlier_removal and self.outlier_diagnostics:
                n = self.X_train.shape[0] - self.outlier_diagnostics.get('n_removed', 0)
            else:
                n = self.X_train.shape[0]
            
            p = len(self.feature_names) + 1  # features + intercept
            
            # Get residuals in transformed scale
            y_train_transformed = self.apply_transformation(self.y_train)
            y_pred_train_transformed = self._predict_core(self.X_train)
            residuals = y_train_transformed - y_pred_train_transformed
            
            sse = np.sum(residuals ** 2)
            sbc = n * np.log(sse / n) + p * np.log(n)
            metrics['sbc'] = sbc
            metrics['sbc_delta_from_2015'] = sbc - self.MODEL_5B_SBC_2015
        
        return metrics
    
    #renewcommands_file =  f"../../report/models/model_{self.model_id}/model_{self.model_id}_renewcommands.tex"
    def generate_latex_commands(self) -> None:
        """Generate LaTeX commands with Model 5b comparison"""
        # Call base class first (generates all common commands including outlier diagnostics)
        super().generate_latex_commands()
        
        model_word = self._number_to_word(self.model_id)
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Add Model 5b-specific newcommands
        with open(newcommands_file, 'a') as f:
            f.write("\n% Model 5b Comparison Commands\n")
            f.write(f"\\newcommand{{\\Model{model_word}FiveBRSquaredTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}FiveBSBCTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}FiveBRMSETwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}FiveBOutlierPctTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}RSquaredDeltaFromTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}SBC}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}SBCDeltaFromTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}OutlierPctDeltaFromTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}RMSEDeltaFromTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
                
        # Add Model 5b-specific renewcommands
        with open(renewcommands_file, 'a') as f:
            f.write("\n% Model 5b (2015) Benchmark Values\n")
            f.write(f"\\renewcommand{{\\Model{model_word}FiveBRSquaredTwoThousandFifteen}}{{{self.MODEL_5B_R2_2015:.4f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}FiveBSBCTwoThousandFifteen}}{{{self.MODEL_5B_SBC_2015:.1f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}FiveBRMSETwoThousandFifteen}}{{{self.MODEL_5B_RMSE_2015:.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}FiveBOutlierPctTwoThousandFifteen}}{{{self.MODEL_5B_OUTLIER_PCT_2015:.2f}}}\n")
            
            # Add RMSE delta in sqrt scale for fair comparison
            if 'rmse_test_sqrt' in self.metrics:
                delta_rmse = self.metrics['rmse_test_sqrt'] - self.MODEL_5B_RMSE_2015
                sign = '+' if delta_rmse >= 0 else ''
                f.write(f"\\renewcommand{{\\Model{model_word}RMSEDeltaFromTwoThousandFifteen}}{{{sign}{delta_rmse:.2f}}}\n")

            # Delta calculations
            if 'r2_delta_from_2015' in self.metrics:
                delta_r2 = self.metrics['r2_delta_from_2015']
                sign = '+' if delta_r2 >= 0 else ''
                f.write(f"\\renewcommand{{\\Model{model_word}RSquaredDeltaFromTwoThousandFifteen}}{{{sign}{delta_r2:.4f}}}\n")
            
            if 'sbc' in self.metrics:
                f.write(f"\\renewcommand{{\\Model{model_word}SBC}}{{{self.metrics['sbc']:.1f}}}\n")
                if 'sbc_delta_from_2015' in self.metrics:
                    delta_sbc = self.metrics['sbc_delta_from_2015']
                    sign = '+' if delta_sbc >= 0 else ''
                    f.write(f"\\renewcommand{{\\Model{model_word}SBCDeltaFromTwoThousandFifteen}}{{{sign}{delta_sbc:.1f}}}\n")
            
            # Outlier percent delta from 2015
            if self.outlier_diagnostics:
                pct_removed = self.outlier_diagnostics.get('pct_removed', 0)
                delta_pct = pct_removed - self.MODEL_5B_OUTLIER_PCT_2015
                sign = '+' if delta_pct >= 0 else ''
                f.write(f"\\renewcommand{{\\Model{model_word}OutlierPctDeltaFromTwoThousandFifteen}}{{{sign}{delta_pct:.2f}}}\n")
            else:
                # If no outlier diagnostics, the delta is just the negative of the 2015 value
                delta_pct = -self.MODEL_5B_OUTLIER_PCT_2015
                f.write(f"\\renewcommand{{\\Model{model_word}OutlierPctDeltaFromTwoThousandFifteen}}{{{delta_pct:.2f}}}\n")        
                self.logger.info("Model 5b comparison commands added")
    
    def plot_diagnostics(self) -> None:
        """Generate enhanced diagnostic plots with Model 5b comparison"""
        if self.test_predictions is None or self.y_test is None:
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        
        # 1. Actual vs Predicted with Model 5b R^2 comparison
        ax = axes[0, 0]
        ax.scatter(self.y_test / 1000, self.test_predictions / 1000, alpha=0.5, s=20, label='2024 Data')
        max_val = max(self.y_test.max(), self.test_predictions.max()) / 1000
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Cost ($1000s)', fontsize=11)
        ax.set_ylabel('Predicted Cost ($1000s)', fontsize=11)
        
        r2_2024 = self.metrics.get('r2_test', 0)
        r2_delta = r2_2024 - self.MODEL_5B_R2_2015
        delta_sign = '+' if r2_delta >= 0 else ''
        
        ax.set_title(f'Actual vs Predicted\n' + 
                    f'Model 5b (2015): R^2 = {self.MODEL_5B_R2_2015:.4f}\n' +
                    f'Model 1 (2024): R^2 = {r2_2024:.4f} ({delta_sign}{r2_delta:.4f})',
                    fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Coefficient Comparison
        ax = axes[0, 1]
        if self.coefficients is not None:
            coef_indices = np.arange(len(self.coefficients))
            colors = ['green' if c > 0 else 'red' for c in self.coefficients]
            ax.barh(coef_indices, self.coefficients, color=colors, alpha=0.7)
            ax.set_yticks(coef_indices)
            ax.set_yticklabels(self.feature_names, fontsize=8)
            ax.set_xlabel('Coefficient Value', fontsize=11)
            ax.set_title('Model 1 Coefficients (2024 Data)', fontsize=10)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
        
        # 3. Distribution comparison
        ax = axes[0, 2]
        ax.hist(self.y_test / 1000, bins=50, alpha=0.5, label='Actual', edgecolor='black')
        ax.hist(self.test_predictions / 1000, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
        ax.set_xlabel('Cost ($1000s)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Cost Distribution', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        ax = axes[1, 0]
        from scipy import stats
        residuals = self.test_predictions - self.y_test
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Residuals)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 5. Studentized Residuals
        ax = axes[1, 1]
        if self.outlier_diagnostics and 'studentized_residuals' in self.outlier_diagnostics:
            t_i = self.outlier_diagnostics['studentized_residuals']
            ax.scatter(range(len(t_i)), t_i, alpha=0.5, s=10)
            ax.axhline(y=self.MODEL_5B_OUTLIER_THRESHOLD, color='red', linestyle='--', 
                      linewidth=2, label=f'Threshold: +-{self.MODEL_5B_OUTLIER_THRESHOLD}')
            ax.axhline(y=-self.MODEL_5B_OUTLIER_THRESHOLD, color='red', linestyle='--', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
            ax.set_xlabel('Observation Index', fontsize=11)
            ax.set_ylabel('Studentized Residual ($t_i$)', fontsize=11)
            
            mean_ti = self.outlier_diagnostics['mean_ti']
            std_ti = self.outlier_diagnostics['std_ti']
            pct_within = self.outlier_diagnostics['pct_within_threshold']
            n_removed = self.outlier_diagnostics['n_removed']
            pct_removed = self.outlier_diagnostics['pct_removed']
            
            ax.set_title(f'Studentized Residuals\n' +
                        f'Mean: {mean_ti:.4f}, Std: {std_ti:.4f}\n' +
                        f'Within: {pct_within:.1f}% | Removed: {n_removed:,} ({pct_removed:.2f}%)',
                        fontsize=9)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Performance by cost quartile
        ax = axes[1, 2]
        quartiles = np.percentile(self.y_test, [25, 50, 75])
        q_labels = ['Q1\nLow', 'Q2', 'Q3', 'Q4\nHigh']
        q_r2 = []
        from sklearn.metrics import r2_score
        for i, (bounds) in enumerate([
            (0, quartiles[0]),
            (quartiles[0], quartiles[1]),
            (quartiles[1], quartiles[2]),
            (quartiles[2], np.inf)
        ]):
            mask = (self.y_test >= bounds[0]) & (self.y_test < bounds[1])
            if np.sum(mask) > 0:
                q_r2.append(r2_score(self.y_test[mask], self.test_predictions[mask]))
            else:
                q_r2.append(0)
        
        ax.bar(q_labels, q_r2, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
        ax.set_ylabel('R^2 Score', fontsize=11)
        ax.set_title('Performance by Cost Quartile', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([min(q_r2) - 0.1, 1.0])
        
        plt.suptitle(f'Model 1: Re-evaluation of Model 5b with 2024 Data\n' +
                    f'21 Features (Exact Model 5b Specification)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'diagnostic_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Diagnostic plots saved to {self.output_dir_relative / 'diagnostic_plots.png'}")


def main():
    """Run Model 1 pipeline"""
    
    # Initialize model with sqrt transform (Model 5b default)
    use_sqrt = True
    use_outlier = True
    suffix = 'Sqrt_' + str(use_sqrt) + '_Outliers_' + str(use_outlier)
    model = Model1Linear(
                use_sqrt_transform = use_sqrt,      # sqrt transformation
                use_outlier_removal = use_outlier,      # enable outlier removal
                outlier_threshold = 1.645,        # ~10% outliers (Model 5b default)
                log_suffix = suffix  # Optional log suffix
            )    
    
    model.logger.info("=" * 80)
    model.logger.info("MODEL 1: RE-EVALUATION OF MODEL 5B WITH 2024 DATA")
    model.logger.info("=" * 80)
    model.logger.info("")
    model.logger.info("This model replicates Model 5b (Tao & Niu 2015) specification exactly:")
    model.logger.info("  - 21 features (includes 5 living, 2 age dummies, BSum, 3 interactions, 10 QSI)")
    if use_sqrt: 
        model.logger.info("  - Square-root transformation")
    if use_outlier:
        model.logger.info("  - Studentized residuals outlier detection (|t_i| >= 1.645)")
    model.logger.info("  - Ordinary Least Squares regression")
    model.logger.info("")
    model.logger.info("Purpose: Direct comparison of Model 5b performance across 9 years")
    model.logger.info("=" * 80)
    model.logger.info("")
    
    # Run complete pipeline
    results = model.run_complete_pipeline(
        fiscal_year_start=2024,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )

    model.logger.info("="*80)
    model.logger.info("SUBGROUP DIAGNOSTICS")
    model.logger.info("="*80)
    model.logger.info(f"Subgroups generated: {list(model.subgroup_metrics.keys())}")
    model.logger.info("="*80)
    
    model.logger.info("")
    model.logger.info("=" * 80)
    model.logger.info("MODEL 5b COMPARISON SUMMARY")
    model.logger.info("=" * 80)
    model.logger.info(f"Model 5b (2015):")
    model.logger.info(f"  R^2 = {model.MODEL_5B_R2_2015:.4f}")
    model.logger.info(f"  SBC = {model.MODEL_5B_SBC_2015:,.1f}")
    model.logger.info(f"  RMSE = ${model.MODEL_5B_RMSE_2015:.2f} (sqrt scale)")
    if model.outlier_diagnostics:
        model.logger.info(f"  Outliers = {model.MODEL_5B_OUTLIER_PCT_2015:.2f}%")
    model.logger.info("")
    model.logger.info(f"Model 1 (2024):")
    model.logger.info(f"  R^2 = {model.metrics.get('r2_test', 0):.4f}")
    if 'sbc' in model.metrics:
        model.logger.info(f"  SBC = {model.metrics['sbc']:,.1f}")
    model.logger.info(f"  RMSE = ${model.metrics.get('rmse_test_sqrt', 0):,.2f} (sqrt scale)")
    model.logger.info(f"  RMSE = ${model.metrics.get('rmse_test', 0):,.2f} (original scale)")
    if model.outlier_diagnostics:
        model.logger.info(f"  Outliers = {model.outlier_diagnostics.get('pct_removed', 0):.2f}%")
    model.logger.info("")
    model.logger.info(f"Delta:")
    if 'r2_delta_from_2015' in model.metrics:
        delta = model.metrics['r2_delta_from_2015']
        model.logger.info(f"  Delta R^2 = {delta:+.4f} ({abs(delta)*100:.2f}% {'improvement' if delta > 0 else 'decline'})")
    if 'sbc_delta_from_2015' in model.metrics:
        delta_sbc = model.metrics['sbc_delta_from_2015']
        model.logger.info(f"  Delta SBC = {delta_sbc:+,.1f} ({'better' if delta_sbc < 0 else 'worse'})")
    if model.outlier_diagnostics:
        delta_outlier = model.outlier_diagnostics.get('pct_removed', 0) - model.MODEL_5B_OUTLIER_PCT_2015
        model.logger.info(f"  Delta Outlier% = {delta_outlier:+.2f}%")
    model.logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    main()