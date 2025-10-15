"""
model_9_random_forest.py
========================
Model 9: Random Forest Regression
Ensemble learning with automatic feature interaction detection

Following the EXACT pattern from Models 1, 2, and 3

KEY FEATURES:
- Ensemble of decision trees with bootstrap aggregation
- Automatic feature interaction detection (no manual specification needed)
- Non-parametric approach (no linearity assumptions)
- Out-of-bag (OOB) error estimation for validation
- Feature importance ranking for interpretability
- Naturally robust to outliers - NO data removal (100% utilization)
- Optional sqrt transformation (test empirically!)

IMPLEMENTATION NOTES:
- Random Forest is naturally robust, so use_outlier_removal = False
- Can work well with or without sqrt transform - test both!
- Provides feature importance for interpretability
- OOB error provides built-in validation metric
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import logging
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from base_model import BaseiBudgetModel, ConsumerRecord

logger = logging.getLogger(__name__)

# ============================================================================
# SINGLE POINT OF CONTROL FOR RANDOM SEED
# ============================================================================
RANDOM_SEED = 42


class Model9RandomForest(BaseiBudgetModel):
    """
    Model 9: Random Forest Regression
    
    Follows the EXACT pattern from Models 1, 2, and 3:
    - Same feature preparation structure (Model 5b features)
    - Same initialization parameters
    - Same main() function pattern
    - Only difference: Random Forest in _fit_core()
    """
    
    def __init__(self,
                 use_sqrt_transform: bool = True,
                 use_outlier_removal: bool = False,  # RF NEVER removes outliers
                 outlier_threshold: float = 1.645,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 n_jobs: int = -1,
                 random_seed: int = RANDOM_SEED,
                 log_suffix: Optional[str] = None,
                 **kwargs):
        """
        Initialize Model 9 Random Forest
        
        Args:
            use_sqrt_transform: Whether to use sqrt transformation
            use_outlier_removal: Should always be False for RF
            outlier_threshold: Kept for compatibility
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_split: Min samples to split internal node
            min_samples_leaf: Min samples in leaf node
            max_features: Features to consider for best split
            bootstrap: Whether to use bootstrap samples
            oob_score: Whether to compute out-of-bag score
            n_jobs: Number of parallel jobs (-1 = all cores)
            random_seed: Random seed for reproducibility
            log_suffix: Optional suffix for log file
        """
        # Determine transformation
        transformation = 'sqrt' if use_sqrt_transform else 'none'
        
        # Initialize base class
        super().__init__(
            model_id=9,
            model_name="Random-Forest",
            transformation=transformation,
            use_outlier_removal=use_outlier_removal,
            outlier_threshold=outlier_threshold,
            random_seed=random_seed,
            log_suffix=log_suffix
        )
        
        # Random Forest specific parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        
        # Model storage
        self.model = None
        
        # Additional metrics
        self.oob_r2 = None
        self.oob_error = None
        self.training_time = None
        self.mean_tree_depth = None
        self.feature_importances_dict = {}
        self.top_features = []
        
        # Log configuration
        self.logger.info(f"  - n_estimators: {self.n_estimators}")
        self.logger.info(f"  - max_depth: {self.max_depth}")
        self.logger.info(f"  - Random seed: {self.random_seed}")
        
        # Run complete pipeline (base class handles metrics)
        results = self.run_complete_pipeline(
            fiscal_year_start=2024,
            fiscal_year_end=2024,
            test_size=0.2,
            perform_cv=True,
            n_cv_folds=10
        )
        
        # Generate custom diagnostic plots for Random Forest
        self.plot_diagnostics()
        
        # Log final summary
        self.log_section(f"MODEL {self.model_id} FINAL SUMMARY", "=")
        
        self.logger.info("")
        self.logger.info("Performance Metrics (Final):")
        self.logger.info(f"  Training R^2: {self.metrics.get('r2_train', 0):.4f}")
        self.logger.info(f"  Test R^2: {self.metrics.get('r2_test', 0):.4f}")
        self.logger.info(f"  RMSE (original scale): ${self.metrics.get('rmse_test', 0):,.2f}")
        self.logger.info(f"  MAE (original scale): ${self.metrics.get('mae_test', 0):,.2f}")
        
        self.logger.info("")
        self.logger.info("Percentage Error Metrics:")
        if 'mape_test' in self.metrics:
            self.logger.info(f"  MAPE: {self.metrics['mape_test']:.2f}%")
        if 'smape' in self.metrics:
            self.logger.info(f"  SMAPE: {self.metrics['smape']:.2f}%")
        if 'mape_threshold' in self.metrics and not np.isnan(self.metrics.get('mape_threshold', np.nan)):
            threshold = self.metrics.get('mape_threshold_value', 1000)
            n = self.metrics.get('mape_n', 0)
            self.logger.info(f"  MAPE (costs >= ${threshold:,.0f}, n={n:,}): {self.metrics['mape_threshold']:.2f}%")
        
        if 'cv_mean' in self.metrics:
            self.logger.info("")
            self.logger.info(f"  CV R^2: {self.metrics['cv_mean']:.4f} +- {self.metrics['cv_std']:.4f}")
        
        self.logger.info("")
        self.logger.info("Random Forest Specific:")
        if self.oob_r2 is not None:
            self.logger.info(f"  OOB R^2: {self.oob_r2:.4f}")
            self.logger.info(f"  OOB RMSE: ${self.oob_error:,.2f}")
        self.logger.info(f"  Mean tree depth: {self.mean_tree_depth:.1f}")
        self.logger.info(f"  Training time: {self.training_time:.2f} seconds")
        
        self.logger.info("")
        self.logger.info("Top 5 Features:")
        for i, (feature, importance) in enumerate(self.top_features, 1):
            self.logger.info(f"  {i}. {feature}: {importance:.4f}")
        
        self.logger.info("")
        self.logger.info("Data Utilization:")
        self.logger.info(f"  Training samples: {self.metrics.get('training_samples', 0):,}")
        self.logger.info(f"  Test samples: {self.metrics.get('test_samples', 0):,}")
        self.logger.info(f"  Total features: {len(self.feature_names)}")
        self.logger.info(f"  Data retention: 100% (no outlier removal)")
        
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
        Prepare features following EXACT Model 5b specification
        (Same as Models 1, 2, 3)
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
        Core Random Forest fitting (data already cleaned and transformed by base class)
        
        Args:
            X: Feature matrix (outliers removed if enabled)
            y: Target values (transformed to sqrt scale if enabled)
        """
        self.log_section("FITTING RANDOM FOREST")
        
        self.logger.info(f"Random Forest Configuration:")
        self.logger.info(f"  n_estimators: {self.n_estimators}")
        self.logger.info(f"  max_depth: {self.max_depth}")
        self.logger.info(f"  min_samples_split: {self.min_samples_split}")
        self.logger.info(f"  min_samples_leaf: {self.min_samples_leaf}")
        self.logger.info(f"  max_features: {self.max_features}")
        self.logger.info(f"  bootstrap: {self.bootstrap}")
        self.logger.info(f"  oob_score: {self.oob_score}")
        
        # Initialize Random Forest
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # Fit with timing
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        
        self.logger.info(f"Random Forest fitted in {self.training_time:.2f} seconds")
        
        # Calculate OOB score if enabled
        if self.oob_score and hasattr(self.model, 'oob_prediction_'):
            # OOB predictions are in fitted scale
            oob_pred_fitted = self.model.oob_prediction_
            
            # Inverse transform OOB predictions
            oob_pred_original = self.inverse_transformation(oob_pred_fitted)
            oob_pred_original = np.maximum(0, oob_pred_original)
            
            # Calculate OOB metrics on original scale
            # Get original scale y for OOB calculation
            y_original = self.inverse_transformation(y)
            
            self.oob_r2 = r2_score(y_original, oob_pred_original)
            self.oob_error = np.sqrt(mean_squared_error(y_original, oob_pred_original))
            
            self.logger.info(f"Out-of-Bag R^2: {self.oob_r2:.4f}")
            self.logger.info(f"Out-of-Bag RMSE (original scale): ${self.oob_error:,.2f}")
        
        # Calculate mean tree depth
        tree_depths = [tree.get_depth() for tree in self.model.estimators_]
        self.mean_tree_depth = np.mean(tree_depths)
        self.logger.info(f"Mean tree depth: {self.mean_tree_depth:.1f}")
        
        # Calculate feature importance
        self.calculate_feature_importance()
    
    def _predict_core(self, X: np.ndarray) -> np.ndarray:
        """
        Core prediction (returns in fitted scale)
        Base class will handle inverse transformation
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions in transformed scale (base class will inverse transform)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def calculate_feature_importance(self) -> None:
        """Calculate and log feature importance"""
        if self.model is None:
            self.logger.warning("Model not fitted, cannot calculate feature importance")
            return
        
        # Get feature importances from Random Forest
        importances = self.model.feature_importances_
        
        # Create dictionary
        self.feature_importances_dict = {
            name: importance 
            for name, importance in zip(self.feature_names, importances)
        }
        
        # Get top 5 features
        sorted_features = sorted(
            self.feature_importances_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        self.top_features = sorted_features[:5]
        
        self.logger.info("")
        self.logger.info("Top 5 Features by Importance:")
        for i, (feature, importance) in enumerate(self.top_features, 1):
            self.logger.info(f"  {i}. {feature}: {importance:.4f}")
    
    def plot_diagnostics(self) -> None:
        """
        Model 9: 2x3 diagnostics on ORIGINAL dollar scale.
        Overrides the base plotter to provide comprehensive Random Forest diagnostics.
        """
        import matplotlib.pyplot as plt
        from scipy import stats
        
        if self.X_test is None or self.y_test is None:
            self.logger.warning("No test data available for diagnostics.")
            return
        
        # Use the model's predictions on the test set (already on original scale)
        y_true = self.y_test.astype(float)
        y_pred = self.test_predictions.astype(float)  # Already inverse-transformed by base class
        resid = y_true - y_pred
        std_resid = resid / (np.std(resid) if np.std(resid) > 0 else 1.0)
        
        self.log_section("GENERATING DIAGNOSTIC PLOTS")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle('Model 9: Random Forest Regression', fontsize=14, fontweight='bold')
        
        # (1) Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(y_true / 1000, y_pred / 1000, alpha=0.5, s=12)
        vmax = max(y_true.max(), y_pred.max()) / 1000
        ax.plot([0, vmax], [0, vmax], 'r--', lw=2, alpha=0.8)
        ax.set_xlabel('Actual Cost ($1000s)')
        ax.set_ylabel('Predicted Cost ($1000s)')
        ax.set_title(f'Actual vs Predicted\nR$^2$ = {self.metrics.get("r2_test", 0):.4f}')
        ax.grid(True, alpha=0.3)
        
        # (2) Residuals vs Predicted
        ax = axes[0, 1]
        ax.scatter(y_pred / 1000, resid / 1000, alpha=0.5, s=12)
        ax.axhline(0, color='r', linestyle='--', lw=1.5)
        ax.set_xlabel('Predicted ($1000s)')
        ax.set_ylabel('Residual ($1000s)')
        ax.set_title('Residuals vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # (3) Feature Importance (Top 10)
        ax = axes[0, 2]
        if len(self.feature_importances_dict) > 0:
            # Get top 10 features
            sorted_features = sorted(
                self.feature_importances_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            features = [f[0] for f in sorted_features]
            importances = [f[1] for f in sorted_features]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importance')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
        
        # (4) Q-Q Plot of residuals
        ax = axes[1, 0]
        stats.probplot(std_resid, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Residuals)')
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
        ax.grid(True, alpha=0.3)
        
        # (6) Calibration by prediction decile
        ax = axes[1, 2]
        q = np.quantile(y_pred, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(y_pred, q[1:-1], right=True), 0, 9)
        cal_act = np.array([np.mean(y_true[idx == b]) for b in range(10)])
        cal_pred = np.array([np.mean(y_pred[idx == b]) for b in range(10)])
        ax.plot(cal_act / 1000, cal_pred / 1000, marker='o', markersize=8, linewidth=2)
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
    
    def generate_latex_commands(self) -> None:
        """
        Generate LaTeX commands for Model 9
        
        CRITICAL: Must call super() FIRST, then append model-specific commands
        """
        # STEP 1: Call parent class method FIRST (creates files with 'w' mode)
        super().generate_latex_commands()
        
        # STEP 2: Append model-specific commands using 'a' mode
        self.logger.info(f"Appending Model {self.model_id} specific LaTeX commands...")
        
        model_word = self._number_to_word(self.model_id)
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # STEP 3: Append new command definitions
        with open(newcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Random Forest Specific Commands\n")
            f.write("% ============================================================================\n")
            f.write(f"\\newcommand{{\\Model{model_word}Transformation}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}NTrees}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MaxDepth}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MinSamplesSplit}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MinSamplesLeaf}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MaxFeatures}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}OOBRSquared}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}OOBError}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MeanTreeDepth}}{{\\placeholder}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}TrainingTime}}{{\\placeholder}}\n")
            
            # Top 5 features
            for i, word in enumerate(['One', 'Two', 'Three', 'Four', 'Five'], 1):
                f.write(f"\\newcommand{{\\Model{model_word}TopFeature{word}}}{{\\placeholder}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}TopFeature{word}Importance}}{{\\placeholder}}\n")
        
        # STEP 4: Provide values for all commands
        with open(renewcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Random Forest Specific Values\n")
            f.write("% ============================================================================\n")
            
            # Transformation
            trans_name = 'sqrt' if self.transformation == 'sqrt' else 'none (original dollars)'
            f.write(f"\\renewcommand{{\\Model{model_word}Transformation}}{{{trans_name}}}\n")
            
            # Number of features
            n_features = len(self.feature_names) if self.feature_names else 0
            f.write(f"\\renewcommand{{\\Model{model_word}NumFeatures}}{{{n_features}}}\n")
            
            # RF parameters
            f.write(f"\\renewcommand{{\\Model{model_word}NTrees}}{{{self.n_estimators}}}\n")
            max_depth_str = str(self.max_depth) if self.max_depth is not None else 'unlimited'
            f.write(f"\\renewcommand{{\\Model{model_word}MaxDepth}}{{{max_depth_str}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}MinSamplesSplit}}{{{self.min_samples_split}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}MinSamplesLeaf}}{{{self.min_samples_leaf}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}MaxFeatures}}{{{self.max_features}}}\n")
            
            # OOB metrics
            if self.oob_r2 is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}OOBRSquared}}{{{self.oob_r2:.4f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OOBError}}{{{self.oob_error:,.0f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\Model{model_word}OOBRSquared}}{{N/A}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OOBError}}{{N/A}}\n")
            
            # Tree depth
            if self.mean_tree_depth is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}MeanTreeDepth}}{{{self.mean_tree_depth:.1f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\Model{model_word}MeanTreeDepth}}{{N/A}}\n")
            
            # Training time
            if self.training_time is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}TrainingTime}}{{{self.training_time:.2f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\Model{model_word}TrainingTime}}{{0.00}}\n")
            
            # Top 5 features
            if len(self.top_features) >= 5:
                for i, word in enumerate(['One', 'Two', 'Three', 'Four', 'Five'], 0):
                    feature_name, importance = self.top_features[i]
                    # Clean feature name for LaTeX (replace underscores)
                    clean_name = feature_name.replace('_', ' ')
                    f.write(f"\\renewcommand{{\\Model{model_word}TopFeature{word}}}{{{clean_name}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}TopFeature{word}Importance}}{{{importance:.4f}}}\n")
            else:
                # Provide defaults if not enough features
                for word in ['One', 'Two', 'Three', 'Four', 'Five']:
                    f.write(f"\\renewcommand{{\\Model{model_word}TopFeature{word}}}{{N/A}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}TopFeature{word}Importance}}{{0.0000}}\n")
        
        self.logger.info(f"Model {self.model_id} specific commands appended successfully")


def main():
    """
    Run Model 9 Random Forest implementation
    Following EXACT pattern from Models 1, 2, and 3
    """
    logger.info("="*80)
    logger.info("MODEL 9: RANDOM FOREST REGRESSION")
    logger.info("="*80)
    
    # Initialize with explicit parameters (following Model 1, 2, 3 pattern)
    use_sqrt = False  # TEST EMPIRICALLY - RF may work well without transform too!
    use_outlier = False  # RF is naturally robust - NO outlier removal
    suffix = f'Sqrt_{use_sqrt}_Outliers_{use_outlier}'
    
    model = Model9RandomForest(
        use_sqrt_transform=use_sqrt,
        use_outlier_removal=use_outlier,  # Always False for RF
        outlier_threshold=1.645,  # Kept for compatibility
        n_estimators=150,  # Number of trees
        max_depth=15,  # Limit depth to prevent overfitting 
        min_samples_split=8,  # Require more samples to split (was: 2)
        min_samples_leaf=3,  # Require more samples per leaf (was: 1)
        max_features='sqrt',  # sqrt(n_features) per split
        bootstrap=True,  # Bootstrap sampling
        oob_score=True,  # Compute OOB error
        n_jobs=-1,  # Use all cores
        random_seed=42,  # For reproducibility
        log_suffix=suffix
    )

if __name__ == "__main__":
    main()