"""
model_11_consensus.py
=====================
Model 11: Consensus Model via Constrained Linear Optimization
Optimal weighted combination of all available models from Model 70 orchestrator

CLEAN IMPLEMENTATION - NO PICKLE FILES NEEDED
- Loads predictions.csv directly (has CaseNo, Dataset, Actual_Cost, Model_* columns)
- Splits by Dataset column (Train/Test)
- No ConsumerRecord objects needed
- Minimal overhead, maximum efficiency

Following the base model pattern where applicable, but optimized for consensus use case.

KEY FEATURES:
- Dynamically detects which models are available (any column starting with "Model_")
- Finds optimal weights via constrained least squares OR Lasso
- Constraints: weights sum to 1, non-negative, configurable max weight
- Cross-validation for weight stability analysis
- Marginal contribution analysis

METHODS SUPPORTED:
- 'CLS': Constrained Least Squares (default) - exact optimization
- 'lasso': Lasso regression - encourages sparsity
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import logging
import json
import warnings
from scipy.optimize import minimize
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

from base_model import BaseiBudgetModel, ConsumerRecord

logger = logging.getLogger(__name__)

# ============================================================================
# SINGLE POINT OF CONTROL FOR RANDOM SEED
# ============================================================================
RANDOM_SEED = 42


class Model11Consensus(BaseiBudgetModel):
    """
    Model 11: Consensus Model
    
    SIMPLIFIED DESIGN:
    - Loads predictions.csv directly (no pickle files)
    - Overrides run_complete_pipeline() to skip ConsumerRecord creation
    - Still inherits metrics, plotting, LaTeX generation from base class
    
    DYNAMIC MODEL DETECTION:
    - Automatically detects which models are available in predictions.csv
    - Looks for columns named "Model_1", "Model_2", etc.
    - Works with any combination (e.g., 2,3,4,5,6,9 or 1,2,3,...,10)
    """
    
    # Models will be dynamically detected from predictions.csv
    CONSTITUENT_MODELS = None
    
    def __init__(self,
                 method: str = 'CLS',
                 max_weight: float = 0.6,
                 lasso_alpha: Optional[float] = None,
                 random_seed: int = RANDOM_SEED,
                 log_suffix: Optional[str] = None,
                 **kwargs):
        """
        Initialize Model 11 Consensus
        
        Args:
            method: 'CLS' (constrained least squares) or 'lasso'
            max_weight: Maximum weight any single model can have (0.6 default)
            lasso_alpha: Lasso regularization parameter (None = cross-validated)
            random_seed: Random seed for reproducibility
        """
        super().__init__(
            model_id=11,
            model_name="Consensus",
            transformation='none',
            use_outlier_removal=False,
            outlier_threshold=0,
            random_seed=random_seed,
            log_suffix=log_suffix
        )
        
        # Consensus-specific attributes
        self.method = method.upper()
        if self.method not in ['CLS', 'LASSO']:
            raise ValueError(f"method must be 'CLS' or 'lasso', got {method}")
        
        self.max_weight = max_weight
        if not 0 < max_weight <= 1:
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        
        self.lasso_alpha = lasso_alpha
        
        # Will store optimal weights and analysis
        self.weights = None
        self.bias_term = None  # Bias in scaled space
        self.bias_original = None  # Bias in original dollar scale
        self.X_mean = None  # For scaling
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.model_contributions = {}
        self.weight_stability = {}
        self.diversity_score = None
        
        self.logger.info(f"Model 11 initialized: method={self.method}, max_weight={self.max_weight}")
    
    def run_complete_pipeline(self,
                              fiscal_year_start: int = 2024,
                              fiscal_year_end: int = 2024,
                              test_size: float = 0.2,
                              perform_cv: bool = True,
                              n_cv_folds: int = 10) -> Dict[str, Any]:
        """
        Override base class pipeline to load predictions.csv directly
        
        NO PICKLE FILES NEEDED - predictions.csv has everything:
        - CaseNo
        - Dataset (Train/Test)
        - Actual_Cost
        - Model_2, Model_3, ... (predictions)
        """
        self.log_section(f"MODEL {self.model_id} COMPLETE PIPELINE")
        
        # Load predictions directly from Model 70
        predictions_path = Path(__file__).parent.parent.parent / 'report' / 'models' / 'model_70' / 'predictions.csv'
        
        if not predictions_path.exists():
            raise FileNotFoundError(f"Model 70 predictions not found at: {predictions_path}")
        
        self.logger.info(f"Loading predictions from: {predictions_path}")
        df = pd.read_csv(predictions_path)
        
        self.logger.info(f"Loaded {len(df)} predictions")
        self.logger.info(f"Columns: {list(df.columns)}")
        
        # Detect model columns
        model_cols = [col for col in df.columns if col.startswith('Model_')]
        if not model_cols:
            raise ValueError("No 'Model_*' columns found in predictions.csv")
        
        # Extract model numbers
        model_numbers = []
        for col in model_cols:
            try:
                num = int(col.split('_')[1])
                model_numbers.append(num)
            except (IndexError, ValueError):
                self.logger.warning(f"Could not parse model number from column: {col}")
        
        self.CONSTITUENT_MODELS = sorted(model_numbers)
        model_cols = [f'Model_{m}' for m in self.CONSTITUENT_MODELS]
        
        self.logger.info(f"Detected {len(self.CONSTITUENT_MODELS)} models: {self.CONSTITUENT_MODELS}")
        
        # Split by Dataset column
        train_df = df[df['Dataset'] == 'Train'].copy()
        test_df = df[df['Dataset'] == 'Test'].copy()
        
        self.logger.info(f"Training samples: {len(train_df)}")
        self.logger.info(f"Test samples: {len(test_df)}")
        
        
        # ============================================================
        #                       DIAGNOSTIC CODE 
        # ============================================================
        from sklearn.metrics import r2_score
        
        # Check individual model performance
        self.logger.info("")
        self.logger.info("Individual Model Performance on Training Set:")
        for col in model_cols:
            model_preds = train_df[col].values
            actual = train_df['Actual_Cost'].values
            r2 = r2_score(actual, model_preds)
            corr = np.corrcoef(actual, model_preds)[0,1]
            self.logger.info(f"  {col}: R²={r2:.4f}, Correlation={corr:.4f}")
        
        # Check test set too
        self.logger.info("")
        self.logger.info("Individual Model Performance on Test Set:")
        for col in model_cols:
            model_preds = test_df[col].values
            actual = test_df['Actual_Cost'].values
            r2 = r2_score(actual, model_preds)
            corr = np.corrcoef(actual, model_preds)[0,1]
            self.logger.info(f"  {col}: R²={r2:.4f}, Correlation={corr:.4f}")
        
        # Check data alignment with specific examples
        self.logger.info("")
        self.logger.info("Sample Data Alignment Check:")
        sample_indices = [0, 100, 1000]
        for idx in sample_indices:
            if idx < len(train_df):
                self.logger.info(f"  Row {idx}: Actual=${train_df.iloc[idx]['Actual_Cost']:,.2f}")
                for col in model_cols:
                    self.logger.info(f"    {col}: ${train_df.iloc[idx][col]:,.2f}")
        
        # Check for any obvious data issues
        self.logger.info("")
        self.logger.info("Data Sanity Checks:")
        self.logger.info(f"  Any NaN in actuals (train): {train_df['Actual_Cost'].isna().any()}")
        self.logger.info(f"  Any NaN in predictions: {train_df[model_cols].isna().any().any()}")
        self.logger.info(f"  Actual range (train): ${train_df['Actual_Cost'].min():,.2f} to ${train_df['Actual_Cost'].max():,.2f}")
        for col in model_cols:
            self.logger.info(f"  {col} range: ${train_df[col].min():,.2f} to ${train_df[col].max():,.2f}")
        # ============================================================
        # END OF DIAGNOSTIC CODE
        # ============================================================        
        
        # --- PATCH: Ensure training and test targets are on raw, aligned scale ---
        #self.y_train = train_df['Actual_Cost'].values.copy()
        #self.y_test = test_df['Actual_Cost'].values.copy()

        
        # Extract features (predictions) and targets (actual costs)
        self.X_train = train_df[model_cols].values
        self.y_train = train_df['Actual_Cost'].values
        self.X_test = test_df[model_cols].values
        self.y_test = test_df['Actual_Cost'].values
        
        self.feature_names = model_cols
        
        # Log prediction statistics
        self.logger.info("Prediction summary:")
        for i, model_name in enumerate(self.feature_names):
            self.logger.info(f"  {model_name}: mean=${self.X_train[:, i].mean():,.2f}, "
                           f"std=${self.X_train[:, i].std():,.2f}")
        
        # Log target statistics
        self.logger.info("Target statistics:")
        self.logger.info(f"  Training mean: ${self.y_train.mean():,.2f}")
        self.logger.info(f"  Training std: ${self.y_train.std():,.2f}")
        self.logger.info(f"  Training range: ${self.y_train.min():,.2f} - ${self.y_train.max():,.2f}")
        
        # Fit model
        self.log_section("FITTING MODEL")
        self._fit_core(self.X_train, self.y_train)
        
        # Ensure targets are in raw (untransformed) scale for metrics
        self.y_train = train_df['Actual_Cost'].values
        self.y_test = test_df['Actual_Cost'].values

        # Make predictions
        self.log_section("GENERATING PREDICTIONS")
        self.train_predictions = self._predict_core(self.X_train)
        self.test_predictions = self._predict_core(self.X_test)
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Cross-validation
        #if perform_cv:
            #self.cv_results = self.perform_cross_validation(n_splits=n_cv_folds)
        
        # Subgroup and variance metrics (use base class methods)
        self.calculate_subgroup_metrics()
        self.calculate_variance_metrics()
        
        # Save results
        self.save_results()
        
        # Generate LaTeX commands
        self.generate_latex_commands()
        
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"MODEL {self.model_id} PIPELINE COMPLETE")
        self.logger.info("="*80)
        
        return {
            'metrics': self.metrics,
            'cv_results': self.cv_results if perform_cv else None,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'diversity_score': self.diversity_score
        }
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        NOT USED IN MODEL 11 - we override run_complete_pipeline()
        
        Kept for compatibility with base class abstract method requirement.
        """
        raise NotImplementedError("Model 11 loads predictions directly; prepare_features() not used")
    
    def _fit_core(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Optimize weights using constrained least squares or Lasso
        
        Args:
            X: [N x K] matrix of model predictions (K = number of models)
            y: [N] actual costs
        """
        self.log_section(f"FITTING CONSENSUS MODEL ({self.method})")
        
        if self.CONSTITUENT_MODELS is None:
            raise RuntimeError("CONSTITUENT_MODELS not set")
        
        n_samples, n_models = X.shape
        self.logger.info(f"Training samples: {n_samples}")
        self.logger.info(f"Constituent models: {n_models} ({self.CONSTITUENT_MODELS})")
        
        if self.method == 'CLS':
            self._fit_cls(X, y)
        elif self.method == 'LASSO':
            self._fit_lasso(X, y)
        
        # Calculate model contributions
        self._calculate_contributions(X, y)
        
        # Calculate diversity score
        self._calculate_diversity()
        
        self.logger.info("")
        self.logger.info("Optimal weights:")
        for i, model_num in enumerate(self.CONSTITUENT_MODELS):
            self.logger.info(f"  Model {model_num}: {self.weights[i]:.4f}")
        
        self.logger.info("")
        self.logger.info(f"Diversity score: {self.diversity_score:.4f}")
        self.logger.info(f"  (0=single model, 1=perfectly uniform)")
    
    def _fit_cls(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using Constrained Least Squares - SIMPLIFIED"""
        self.logger.info("Method: Constrained Least Squares (CLS)")
        self.logger.info(f"  Constraints: sum(w)=1, w_i>=0, w_i<={self.max_weight}")

        n_models = X.shape[1]
        
        # Simple objective: minimize squared errors directly
        def objective(weights):
            predictions = X @ weights
            return np.sum((y - predictions) ** 2)
        
        def gradient(weights):
            predictions = X @ weights
            residuals = predictions - y
            return 2 * (X.T @ residuals)
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds: each weight between 0 and max_weight
        bounds = [(0, self.max_weight)] * n_models
        
        # Initial guess: equal weights
        w0 = np.ones(n_models) / n_models
        
        self.logger.info("Optimizing weights...")
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not result.success:
            self.logger.warning(f"Optimization warning: {result.message}")
        
        self.weights = result.x
        
        # No bias needed for ensemble of predictions
        self.bias_term = 0.0
        self.bias_original = 0.0
        
        # Store dummy scaling values to avoid errors
        self.X_mean = np.zeros(n_models)
        self.X_std = np.ones(n_models)
        self.y_mean = 0
        self.y_std = 1
        
        self.logger.info(f"Optimization converged: {result.success}")
        self.logger.info(f"Final objective value: {result.fun:.2f}")
        self.logger.info(f"Iterations: {result.nit}")    


    def _fit_lasso(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using Lasso regression - SIMPLIFIED"""
        self.logger.info("Method: Lasso (L1 regularization)")
        
        # For Lasso, we need some standardization for regularization to work properly
        # But we only standardize X, not y
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-10
        X_scaled = (X - self.X_mean) / self.X_std
        
        # Don't scale y - keep it in original units
        # This is KEY: we're predicting actual dollar amounts, not z-scores
        
        if self.lasso_alpha is None:
            self.logger.info("  Using LassoCV to find optimal alpha...")
            alphas = np.logspace(-6, 1, 50)  # Adjusted range for unscaled y
            lasso_cv = LassoCV(
                alphas=alphas, 
                cv=5, 
                random_state=self.random_seed,
                max_iter=5000, 
                positive=True,  # Force non-negative weights
                fit_intercept=True
            )
            lasso_cv.fit(X_scaled, y)  # Note: y is NOT scaled
            self.lasso_alpha = lasso_cv.alpha_
            self.logger.info(f"  Optimal alpha: {self.lasso_alpha:.6f}")
            model = lasso_cv
        else:
            self.logger.info(f"  Using alpha: {self.lasso_alpha:.6f}")
            model = Lasso(
                alpha=self.lasso_alpha, 
                random_state=self.random_seed,
                max_iter=5000, 
                positive=True,
                fit_intercept=True
            )
            model.fit(X_scaled, y)  # Note: y is NOT scaled
        
        # Extract raw coefficients (these are for scaled X)
        raw_weights = model.coef_.copy()
        intercept = float(model.intercept_)
        
        # Convert coefficients back to original X scale
        # w_original = w_scaled / X_std
        weights_unscaled = raw_weights / self.X_std
        
        # Normalize to sum to 1 (key for ensemble)
        if np.sum(weights_unscaled) > 0:
            self.weights = weights_unscaled / np.sum(weights_unscaled)
        else:
            self.logger.warning("All Lasso weights are zero! Using equal weights.")
            self.weights = np.ones(X.shape[1]) / X.shape[1]
        
        # Check max weight constraint
        if np.max(self.weights) > self.max_weight:
            self.logger.warning(f"Max weight {np.max(self.weights):.4f} exceeds limit {self.max_weight}")
            self.logger.info("  Clipping and re-normalizing...")
            self.weights = np.clip(self.weights, 0, self.max_weight)
            self.weights = self.weights / np.sum(self.weights)
        
        # Calculate bias in original scale
        # bias = intercept + sum(w_i * X_mean_i)
        self.bias_original = intercept + np.sum((raw_weights / self.X_std) * self.X_mean)
        
        # But for ensemble, we typically don't want bias
        # Individual models already have their biases
        self.bias_original = 0.0
        self.bias_term = 0.0
        
        # Store dummy y scaling values to avoid errors elsewhere
        self.y_mean = 0
        self.y_std = 1
        
        n_nonzero = np.sum(self.weights > 1e-6)
        self.logger.info(f"  Non-zero weights: {n_nonzero}/{len(self.weights)}")

    def _predict_core(self, X: np.ndarray) -> np.ndarray:
        """Apply learned weights to predictions - SIMPLIFIED"""
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Simple weighted average - ensemble predictions are already on correct scale
        return X @ self.weights + self.bias_original

    def calculate_metrics(self) -> None:
        """Override base metrics to ensure raw-scale evaluation"""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        self.log_section("CALCULATING METRICS")

        # Compute on training data
        y_train_pred = self._predict_core(self.X_train)
        y_test_pred = self._predict_core(self.X_test)
        
        self.logger.info(f"DEBUG check alignment: corr(y_train, pred_train) = "
                 f"{np.corrcoef(self.y_train, y_train_pred)[0,1]:.4f}")

        self.metrics = {
            'r2_train': r2_score(self.y_train, y_train_pred),
            'r2_test': r2_score(self.y_test, y_test_pred),
            'rmse_test': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'mae_test': mean_absolute_error(self.y_test, y_test_pred),
            'mape_test': np.mean(
                np.abs((self.y_test - y_test_pred) / np.clip(self.y_test, 1e-6, None))
            ) * 100,
        }

        self.logger.info("")
        self.logger.info("PERFORMANCE METRICS SUMMARY")
        self.logger.info(f"  Training R²: {self.metrics['r2_train']:.4f}")
        self.logger.info(f"  Test R²: {self.metrics['r2_test']:.4f}")
        self.logger.info(f"  RMSE: ${self.metrics['rmse_test']:,.2f}")
        self.logger.info(f"  MAE: ${self.metrics['mae_test']:,.2f}")
        self.logger.info(f"  MAPE: {self.metrics['mape_test']:.2f}%")

    def _calculate_contributions(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate marginal contribution of each model - SIMPLIFIED"""
        self.logger.info("")
        self.logger.info("Calculating model contributions...")

        # Equal-weight predictions - simple average
        w_equal = np.ones(len(self.weights)) / len(self.weights)
        pred_equal = X @ w_equal
        r2_equal = r2_score(y, pred_equal)

        # Optimal-weight predictions
        pred_optimal = self._predict_core(X)
        r2_optimal = r2_score(y, pred_optimal)

        # Leave-one-out marginal contributions
        for i, model_num in enumerate(self.CONSTITUENT_MODELS):
            # Individual model R²
            r2_individual = r2_score(y, X[:, i])
            
            # Leave-one-out ensemble
            w_loo = self.weights.copy()
            w_loo[i] = 0
            if w_loo.sum() > 0:
                w_loo = w_loo / w_loo.sum()
                pred_loo = X @ w_loo
                r2_loo = r2_score(y, pred_loo)
                marginal_contrib = r2_optimal - r2_loo
            else:
                r2_loo = None
                marginal_contrib = None

            self.model_contributions[f'Model_{model_num}'] = {
                'weight': float(self.weights[i]),
                'r2_individual': float(r2_individual),
                'r2_without': float(r2_loo) if r2_loo is not None else None,
                'marginal_contribution': float(marginal_contrib) if marginal_contrib is not None else None
            }

        self.model_contributions['_summary'] = {
            'r2_equal_weights': float(r2_equal),
            'r2_optimal_weights': float(r2_optimal),
            'improvement_over_equal': float(r2_optimal - r2_equal)
        }

        self.logger.info(f"Equal weights R²: {r2_equal:.4f}")
        self.logger.info(f"Optimal weights R²: {r2_optimal:.4f}")
        self.logger.info(f"Improvement: {r2_optimal - r2_equal:+.4f}")    

    def _calculate_diversity(self) -> None:
        """Calculate diversity score using normalized entropy"""
        w_nonzero = self.weights[self.weights > 1e-10]
        
        if len(w_nonzero) <= 1:
            self.diversity_score = 0.0
        else:
            entropy = -np.sum(w_nonzero * np.log(w_nonzero))
            max_entropy = np.log(len(self.weights))
            self.diversity_score = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, Any]:
        """Cross-validation with weight stability analysis"""
        self.log_section(f"CROSS-VALIDATION ({n_splits}-FOLD)")
        
        # Use base class CV for R² scores
        cv_results = super().perform_cross_validation(n_splits=n_splits)
        
        # Weight stability analysis
        self.logger.info("")
        self.logger.info("Analyzing weight stability across folds...")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        fold_weights = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            
            if self.method == 'CLS':
                # Scale data
                X_mean_fold = X_fold_train.mean(axis=0)
                X_std_fold = X_fold_train.std(axis=0) + 1e-10
                y_mean_fold = y_fold_train.mean()
                y_std_fold = y_fold_train.std() + 1e-10
                
                X_scaled = (X_fold_train - X_mean_fold) / X_std_fold
                y_scaled = (y_fold_train - y_mean_fold) / y_std_fold
                
                n_models = X_scaled.shape[1]
                
                def objective(params):
                    weights = params[:-1]
                    bias = params[-1]
                    return np.sum((y_scaled - (X_scaled @ weights + bias))**2)
                
                constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p[:-1]) - 1}]
                bounds = [(0, self.max_weight)] * n_models + [(None, None)]
                params0 = np.concatenate([np.ones(n_models) / n_models, [0.0]])
                
                result = minimize(objective, params0, method='SLSQP', 
                                bounds=bounds, constraints=constraints,
                                options={'ftol': 1e-9})
                fold_weights.append(result.x[:-1])  # Only store weights, not bias

                # ---- FIX: re-normalize weights after scaling and ensure proper interpretation ----
                w = result.x[:-1]
                if np.sum(w) > 0:
                    w = w / np.sum(w)
                fold_weights[-1] = w
            
            elif self.method == 'LASSO':
                # Scale data
                X_mean_fold = X_fold_train.mean(axis=0)
                X_std_fold = X_fold_train.std(axis=0) + 1e-10
                y_mean_fold = y_fold_train.mean()
                y_std_fold = y_fold_train.std() + 1e-10
                
                X_scaled = (X_fold_train - X_mean_fold) / X_std_fold
                y_scaled = (y_fold_train - y_mean_fold) / y_std_fold
                
                if self.lasso_alpha is None:
                    model = LassoCV(cv=5, random_state=self.random_seed, 
                                  positive=True, max_iter=5000, fit_intercept=True)
                else:
                    model = Lasso(alpha=self.lasso_alpha, random_state=self.random_seed,
                                positive=True, max_iter=5000, fit_intercept=True)
                model.fit(X_scaled, y_scaled)
                raw_weights = model.coef_
                if raw_weights.sum() > 0:
                    fold_weights.append(raw_weights / raw_weights.sum())
                else:
                    fold_weights.append(np.ones(X_scaled.shape[1]) / X_scaled.shape[1])
        
        fold_weights = np.array(fold_weights)
        weight_means = fold_weights.mean(axis=0)
        weight_stds = fold_weights.std(axis=0)
        
        self.weight_stability = {
            'mean_weights': weight_means.tolist(),
            'std_weights': weight_stds.tolist(),
            'cv_weights': (weight_stds / (weight_means + 1e-10)).tolist()
        }
        
        self.logger.info("")
        self.logger.info("Weight stability across folds:")
        for i, model_num in enumerate(self.CONSTITUENT_MODELS):
            self.logger.info(f"  Model {model_num}: {weight_means[i]:.4f} ± {weight_stds[i]:.4f}")
        
        cv_results['weight_stability'] = self.weight_stability
        return cv_results
    
    def generate_latex_commands(self) -> None:
        """Generate LaTeX commands"""
        super().generate_latex_commands()
        self._append_consensus_commands()
    
    def _append_consensus_commands(self) -> None:
        """Append consensus-specific LaTeX commands"""
        if self.CONSTITUENT_MODELS is None:
            return
        
        model_word = 'Eleven'
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        with open(newcommands_file, 'a') as f:
            f.write("")
            f.write("% Model 11 Consensus Specific Commands\n")
            f.write(f"\\newcommand{{\\Model{model_word}Method}}{{TBD}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MaxWeight}}{{0.00}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}DiversityScore}}{{0.00}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}ImprovementOverEqual}}{{0.0000}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}NumModels}}{{0}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}BiasOriginal}}{{0.00}}\n")
            
            for model_num in self.CONSTITUENT_MODELS:
                model_name = self._get_model_word(model_num)
                f.write(f"\\newcommand{{\\Model{model_word}Weight{model_name}}}{{0.0000}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Contrib{model_name}}}{{0.0000}}\n")
            
            f.write(f"\\newcommand{{\\Model{model_word}TopContributor}}{{TBD}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}TopWeight}}{{0.0000}}\n")
            
            for model_num in self.CONSTITUENT_MODELS:
                model_name = self._get_model_word(model_num)
                f.write(f"\\newcommand{{\\Model{model_word}WeightStd{model_name}}}{{0.0000}}\n")
        
        with open(renewcommands_file, 'a') as f:
            f.write("")
            f.write("% Model 11 Consensus Specific Values\n")
            f.write(f"\\renewcommand{{\\Model{model_word}Method}}{{{self.method}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}MaxWeight}}{{{self.max_weight:.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}NumModels}}{{{len(self.CONSTITUENT_MODELS)}}}\n")
            
            if self.bias_original is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}BiasOriginal}}{{{self.bias_original:,.2f}}}\n")
            
            if self.diversity_score is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}DiversityScore}}{{{self.diversity_score:.4f}}}\n")
            
            if '_summary' in self.model_contributions:
                improvement = self.model_contributions['_summary']['improvement_over_equal']
                f.write(f"\\renewcommand{{\\Model{model_word}ImprovementOverEqual}}{{{improvement:+.4f}}}\n")
            
            if self.weights is not None:
                for i, model_num in enumerate(self.CONSTITUENT_MODELS):
                    model_name = self._get_model_word(model_num)
                    f.write(f"\\renewcommand{{\\Model{model_word}Weight{model_name}}}{{{self.weights[i]:.4f}}}\n")
                    
                    if f'Model_{model_num}' in self.model_contributions:
                        contrib = self.model_contributions[f'Model_{model_num}'].get('marginal_contribution')
                        if contrib is not None:
                            f.write(f"\\renewcommand{{\\Model{model_word}Contrib{model_name}}}{{{contrib:+.4f}}}\n")
                
                top_idx = np.argmax(self.weights)
                top_model_num = self.CONSTITUENT_MODELS[top_idx]
                f.write(f"\\renewcommand{{\\Model{model_word}TopContributor}}{{Model {top_model_num}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}TopWeight}}{{{self.weights[top_idx]:.4f}}}\n")
            
            if self.weight_stability:
                stds = self.weight_stability['std_weights']
                for i, model_num in enumerate(self.CONSTITUENT_MODELS):
                    model_name = self._get_model_word(model_num)
                    f.write(f"\\renewcommand{{\\Model{model_word}WeightStd{model_name}}}{{{stds[i]:.4f}}}\n")
        
        self.logger.info("Model 11 consensus-specific LaTeX commands generated")
    
    def _get_model_word(self, model_num: int) -> str:
        """Convert model number to word for LaTeX commands"""
        words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 
                 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten'}
        return words.get(model_num, str(model_num))
    
    def save_results(self) -> None:
        """Save results - override base class to avoid ConsumerRecord dependency"""
        
        # Save metrics (standard)
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        self.logger.info("Results saved:")
        self.logger.info(f"  - Metrics JSON: {self.output_dir_relative / 'metrics.json'}")
        
        # Save predictions (simplified - no consumer_id needed)
        if self.test_predictions is not None:
            pred_df = pd.DataFrame({
                'actual': self.y_test,
                'predicted': self.test_predictions,
                'error': self.test_predictions - self.y_test
            })
            pred_file = self.output_dir / "predictions.csv"
            pred_df.to_csv(pred_file, index=False)
            self.logger.info(f"  - Predictions CSV: {self.output_dir_relative / 'predictions.csv'}")
            self.logger.info(f"    ({len(self.test_predictions):,} predictions)")
        
        # Save weights
        weights_file = self.output_dir / "weights.json"
        with open(weights_file, 'w') as f:
            json.dump({
                'method': self.method,
                'max_weight': self.max_weight,
                'weights': {f'Model_{m}': float(self.weights[i]) 
                           for i, m in enumerate(self.CONSTITUENT_MODELS)},
                'bias': float(self.bias_original) if self.bias_original is not None else 0.0,
                'diversity_score': float(self.diversity_score) if self.diversity_score else None
            }, f, indent=2)
        self.logger.info(f"  - Weights: {self.output_dir_relative / 'weights.json'}")
        
        # Save contributions
        contrib_file = self.output_dir / "contribution_analysis.json"
        with open(contrib_file, 'w') as f:
            json.dump(self.model_contributions, f, indent=2)
        self.logger.info(f"  - Contributions: {self.output_dir_relative / 'contribution_analysis.json'}")
        
        # Save stability
        if self.weight_stability:
            stability_file = self.output_dir / "weight_stability.json"
            with open(stability_file, 'w') as f:
                json.dump(self.weight_stability, f, indent=2)
            self.logger.info(f"  - Stability: {self.output_dir_relative / 'weight_stability.json'}")
    
    def plot_diagnostics(self) -> None:
        """Generate diagnostic plots"""
        if self.test_predictions is None or self.y_test is None:
            return
        if self.CONSTITUENT_MODELS is None or self.weights is None:
            return
        
        fig = plt.figure(figsize=(18, 11))
        
        # 1. Actual vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.y_test / 1000, self.test_predictions / 1000, alpha=0.5, s=20)
        ax1.plot([self.y_test.min() / 1000, self.y_test.max() / 1000],
                [self.y_test.min() / 1000, self.y_test.max() / 1000],
                'r--', lw=2, label='Perfect')
        ax1.set_xlabel('Actual Cost ($1000s)')
        ax1.set_ylabel('Predicted Cost ($1000s)')
        ax1.set_title(f'Actual vs Predicted\nR² = {self.metrics.get("r2_test", 0):.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Weight Distribution
        ax2 = plt.subplot(2, 3, 2)
        model_labels = [f'M{m}' for m in self.CONSTITUENT_MODELS]
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.weights)))
        bars = ax2.bar(range(len(self.weights)), self.weights, color=colors)
        ax2.axhline(1/len(self.weights), color='r', linestyle='--', label='Equal', linewidth=2)
        ax2.axhline(self.max_weight, color='orange', linestyle='--', label=f'Max={self.max_weight}', linewidth=2)
        ax2.set_xticks(range(len(self.weights)))
        ax2.set_xticklabels(model_labels)
        ax2.set_ylabel('Weight')
        ax2.set_title(f'Optimal Weights ({self.method})\nDiversity = {self.diversity_score:.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        for i, (bar, weight) in enumerate(zip(bars, self.weights)):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Residual Distribution
        ax3 = plt.subplot(2, 3, 3)
        residuals = self.test_predictions - self.y_test
        ax3.hist(residuals / 1000, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Residual ($1000s)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Residual Distribution\nMean = ${residuals.mean()/1000:,.1f}K')
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q Plot
        ax4 = plt.subplot(2, 3, 4)
        scipy_stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normality)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Residuals vs Fitted
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(self.test_predictions / 1000, residuals / 1000, alpha=0.5, s=20)
        ax5.axhline(0, color='r', linestyle='--', linewidth=2)
        ax5.set_xlabel('Fitted Values ($1000s)')
        ax5.set_ylabel('Residuals ($1000s)')
        ax5.set_title('Residuals vs Fitted')
        ax5.grid(True, alpha=0.3)
        
        # 6. Model Contributions
        ax6 = plt.subplot(2, 3, 6)
        contrib_data = []
        for i, model_num in enumerate(self.CONSTITUENT_MODELS):
            if f'Model_{model_num}' in self.model_contributions:
                contrib = self.model_contributions[f'Model_{model_num}']
                contrib_data.append({
                    'model': f'M{model_num}',
                    'weight': self.weights[i],
                    'marginal': contrib.get('marginal_contribution', 0) or 0
                })
        
        if contrib_data:
            contrib_df = pd.DataFrame(contrib_data)
            x_pos = range(len(contrib_df))
            ax6.bar(x_pos, contrib_df['weight'], alpha=0.7, color=colors)
            for i, row in contrib_df.iterrows():
                ax6.text(i, row['weight']/2, f"Δ={row['marginal']:.4f}",
                        ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(contrib_df['model'])
            ax6.set_ylabel('Weight')
            ax6.set_title('Model Contributions\n(Δ = Marginal R²)')
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = self.output_dir / 'diagnostic_plots.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Diagnostic plots saved: {self.output_dir_relative / 'diagnostic_plots.png'}")


def main():
    """Run Model 11 Consensus"""
    logger.info("="*80)
    logger.info("MODEL 11: CONSENSUS MODEL")
    logger.info("="*80)
    
    method = 'Lasso'
    max_weight = 0.6
    suffix = f"{method}_{max_weight:.1f}".replace('.', 'p')
    
    model = Model11Consensus(
        method=method,
        max_weight=max_weight,
        random_seed=RANDOM_SEED,
        log_suffix=suffix
    )
    
    results = model.run_complete_pipeline(
        fiscal_year_start=2024,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    model.plot_diagnostics()
    
    # Final summary
    model.log_section("MODEL 11 FINAL SUMMARY", "=")
    
    model.logger.info("")
    model.logger.info("Consensus Configuration:")
    model.logger.info(f"  Method: {model.method}")
    model.logger.info(f"  Max weight: {model.max_weight}")
    if model.CONSTITUENT_MODELS:
        model.logger.info(f"  Models: {model.CONSTITUENT_MODELS}")
    
    model.logger.info("")
    model.logger.info("Performance Metrics:")
    model.logger.info(f"  Training R²: {model.metrics.get('r2_train', 0):.4f}")
    model.logger.info(f"  Test R²: {model.metrics.get('r2_test', 0):.4f}")
    model.logger.info(f"  RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    model.logger.info(f"  MAE: ${model.metrics.get('mae_test', 0):,.2f}")
    if 'cv_mean' in model.metrics:
        model.logger.info(f"  CV R²: {model.metrics['cv_mean']:.4f} ± {model.metrics['cv_std']:.4f}")
    
    if model.weights is not None:
        model.logger.info("")
        model.logger.info("Optimal Weights:")
        for i, m in enumerate(model.CONSTITUENT_MODELS):
            model.logger.info(f"  Model {m}: {model.weights[i]:.4f}")
    
    model.logger.info("")
    model.logger.info("Consensus Analysis:")
    model.logger.info(f"  Diversity: {model.diversity_score:.4f}")
    if '_summary' in model.model_contributions:
        s = model.model_contributions['_summary']
        model.logger.info(f"  Equal weights R²: {s['r2_equal_weights']:.4f}")
        model.logger.info(f"  Optimal R²: {s['r2_optimal_weights']:.4f}")
        model.logger.info(f"  Improvement: {s['improvement_over_equal']:+.4f}")
    
    if model.weights is not None:
        top_idx = np.argmax(model.weights)
        top_model = model.CONSTITUENT_MODELS[top_idx]
        model.logger.info(f"  Top: Model {top_model} (w={model.weights[top_idx]:.4f})")
    
    model.logger.info("")
    model.logger.info("Output:")
    model.logger.info(f"  Results: {model.output_dir_relative}")
    model.logger.info(f"  Plots: {model.output_dir_relative / 'diagnostic_plots.png'}")
    
    model.logger.info("")
    model.logger.info("="*80)
    model.logger.info(f"MODEL {model.model_id} COMPLETE")
    model.logger.info("="*80)
    
    return results


if __name__ == "__main__":
    main()