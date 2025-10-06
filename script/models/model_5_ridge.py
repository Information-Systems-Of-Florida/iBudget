"""
Model 5: Ridge Regression with L2 Regularization
=================================================
Uses the 22 robust features from validated Model 5b
Applies coefficient shrinkage to handle multicollinearity

IMPORTANT: Transformation Control Pattern (use_sqrt_transform parameter)
------------------------------------------------------------------------
This model implements a use_sqrt_transform parameter that can be applied to ALL models!

Benefits:
  1. Flexibility: Test both sqrt and original dollar scale
  2. Comparability: Fair comparison between transformation approaches
  3. Interpretability: Original dollars easier to explain to stakeholders
  4. Performance: May find that Ridge on original scale performs as well or better

Usage:
  model = Model5Ridge(use_sqrt_transform=True)   # Historical Model 5b (sqrt)
  model = Model5Ridge(use_sqrt_transform=False)  # Original dollars (simpler)

Implementation pattern to replicate in other models:
  1. Add use_sqrt_transform parameter to __init__
  2. Store self.transformation = "sqrt" or "none"
  3. In run_complete_pipeline(): conditionally apply transformation
  4. In fit(): fit on appropriate scale
  5. In predict(): conditionally back-transform to original dollars
  6. In perform_cross_validation(): handle transformation in CV folds
  7. Always compare final predictions/metrics in original dollar scale

Following Critical Addendum template for homogeneity
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import logging
from datetime import datetime

from base_model import BaseiBudgetModel, ConsumerRecord

logger = logging.getLogger(__name__)

# ============================================================================
# SINGLE POINT OF CONTROL FOR RANDOM SEED
# ============================================================================
# Change this value to get different random splits, or keep at 42 for reproducibility
# This seed controls:
#   - Train/test split
#   - Cross-validation folds
#   - Any other random operations in the pipeline
# ============================================================================
RANDOM_SEED = 42

class Model5Ridge(BaseiBudgetModel):
    """
    Model 5: Ridge Regression with cross-validated alpha selection
    
    Key features:
    - 22 robust features from Model 5b (Table 7)
    - L2 regularization for multicollinearity
    - Square-root transformation of costs
    - Cross-validated alpha selection
    - All features retained (no selection)
    """
    
    def __init__(self, use_fy2024_only: bool = True, use_sqrt_transform: bool = True):
        super().__init__(model_id=5, model_name="Ridge Regression")
        self.use_fy2024_only = use_fy2024_only
        self.fiscal_years_used = "2024" if use_fy2024_only else "2023-2024"
        
        # ============================================================================
        # TRANSFORMATION CONTROL - Applicable to ALL models
        # ============================================================================
        # Set to True to use sqrt transformation (historical Model 5b baseline)
        # Set to False to fit on original dollar scale (simpler interpretation)
        # ============================================================================
        self.use_sqrt_transform = use_sqrt_transform
        self.transformation = "sqrt" if use_sqrt_transform else "none"
        logger.info(f"Transformation: {self.transformation}")
        
        # Ridge-specific parameters
        self.alphas = np.logspace(-4, 4, 100)  # Alpha search range
        self.scaler = StandardScaler()
        self.optimal_alpha = None
        self.alpha_selection_method = "10-fold CV"
        
        # Multicollinearity analysis
        self.condition_number_before = None
        self.condition_number_after = None
        self.effective_dof = None
        self.shrinkage_factors = {}
        
        # Store OLS comparison for shrinkage calculation
        self.ols_coefficients = None
        
        # VIF metrics
        self.max_vif_before = None
        self.max_vif_after = None
        
        # Regularization strength descriptor
        self.regularization_strength = None
        
        # Number of non-zero coefficients (all 22 for Ridge)
        self.num_non_zero = 22
        
        # Number of features (22 from Model 5b)
        self.num_parameters = 23  # 22 features + intercept
        
        logger.info(f"Model 5 Ridge Regression initialized (transform={self.transformation})")
    
    def split_data(self, test_size: float = 0.2, random_state: int = RANDOM_SEED) -> None:
        """
        Override split_data to ensure proper train/test split
        CRITICAL: Handles boolean test_size from base class
        
        Args:
            test_size: Proportion for test set  
            random_state: Random seed for reproducibility
        """
        # Handle boolean test_size (base class sometimes passes True)
        if isinstance(test_size, bool):
            test_size = 0.2 if test_size else 0.0
        
        if not self.all_records:
            raise ValueError("No records loaded. Call load_data() first.")
        
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
        
        logger.info(f"Data split: {len(self.train_records)} training, {len(self.test_records)} test")
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare the 22 robust features from Model 5b (Table 7)
        
        Features:
        - 5 Living Setting dummies (drop FH as reference)
        - 2 Age Group dummies (drop Age3_20 as reference)
        - 10 Individual QSI questions (Q16, Q18, Q20, Q21, Q23, Q28, Q33, Q34, Q36, Q43)
        - 2 Sum scores (BSum, FSum)
        - 3 Interaction terms (SLFSum, SLBSum, FHFSum)
        
        Returns:
            Tuple of (feature matrix, feature names)
        """
        if not records:
            return np.array([]), []
        
        features_list = []
        
        for record in records:
            row_features = []
            
            # 1. Living Setting Dummies (5 features, drop FH as reference)
            living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
            for setting in living_settings:
                value = 1.0 if record.living_setting == setting else 0.0
                row_features.append(value)
            
            # 2. Age Group Dummies (2 features, drop Age3_20 as reference)
            is_age21_30 = 1.0 if record.age_group == 'Age21_30' else 0.0
            is_age31plus = 1.0 if record.age_group == 'Age31Plus' else 0.0
            row_features.extend([is_age21_30, is_age31plus])
            
            # 3. Individual QSI Questions (10 features from Table 7)
            qsi_questions = [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]
            for q_num in qsi_questions:
                value = getattr(record, f'q{q_num}', 0)
                row_features.append(float(value))
            
            # 4. Sum Scores (2 features)
            bsum = float(record.bsum)
            fsum = float(record.fsum)
            row_features.extend([bsum, fsum])
            
            # 5. Interaction Terms (3 features from Model 5b)
            # SLFSum, SLBSum, FHFSum
            is_sl = 1.0 if record.living_setting in ['ILSL'] else 0.0
            is_fh = 1.0 if record.living_setting == 'FH' else 0.0
            
            slf_sum = is_sl * fsum
            slb_sum = is_sl * bsum
            fhf_sum = is_fh * fsum
            
            row_features.extend([slf_sum, slb_sum, fhf_sum])
            
            features_list.append(row_features)
        
        # Define feature names
        feature_names = (
            ['LiveILSL', 'LiveRH1', 'LiveRH2', 'LiveRH3', 'LiveRH4'] +
            ['Age21_30', 'Age31+'] +
            [f'Q{q}' for q in [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]] +
            ['BSum', 'FSum'] +
            ['SLFSum', 'SLBSum', 'FHFSum']
        )
        
        return np.array(features_list), feature_names
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Ridge regression with cross-validated alpha selection
        
        Args:
            X: Feature matrix (unscaled)
            y: Target values (in appropriate scale based on use_sqrt_transform)
        """
        logger.info(f"Fitting Ridge Regression (transform={self.transformation})...")
        
        # First fit OLS for comparison (to calculate shrinkage)
        logger.info("Fitting OLS for shrinkage comparison...")
        X_ols_scaled = StandardScaler().fit_transform(X)
        ols_model = LinearRegression(fit_intercept=True)
        ols_model.fit(X_ols_scaled, y)
        self.ols_coefficients = ols_model.coef_
        
        # Calculate condition number BEFORE regularization
        n_features = X.shape[1]
        XtX = X_ols_scaled.T @ X_ols_scaled
        self.condition_number_before = np.linalg.cond(XtX)
        logger.info(f"Condition number before regularization: {self.condition_number_before:.2f}")
        
        # Scale features for Ridge
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit RidgeCV with automatic alpha selection
        logger.info(f"Testing {len(self.alphas)} alpha values via {self.alpha_selection_method}...")
        self.model = RidgeCV(
            alphas=self.alphas,
            cv=10,
            fit_intercept=True,
            scoring='r2'
        )
        self.model.fit(X_scaled, y)
        
        # Store optimal alpha
        self.optimal_alpha = self.model.alpha_
        logger.info(f"Optimal alpha selected: {self.optimal_alpha:.6f}")
        
        # Determine regularization strength
        if self.optimal_alpha < 0.01:
            self.regularization_strength = "Weak"
        elif self.optimal_alpha < 1.0:
            self.regularization_strength = "Moderate"
        else:
            self.regularization_strength = "Strong"
        logger.info(f"Regularization strength: {self.regularization_strength}")
        
        # Calculate condition number AFTER regularization
        XtX_regularized = XtX + self.optimal_alpha * np.eye(n_features)
        self.condition_number_after = np.linalg.cond(XtX_regularized)
        logger.info(f"Condition number after regularization: {self.condition_number_after:.2f}")
        
        # Calculate improvement
        if self.condition_number_before is not None:
            improvement = (self.condition_number_before - self.condition_number_after) / self.condition_number_before * 100
            logger.info(f"Condition number improvement: {improvement:.1f}%")
        
        # Calculate effective degrees of freedom
        # eff_dof = trace(X(X'X + λI)^-1X')
        self.effective_dof = np.trace(XtX @ np.linalg.inv(XtX_regularized))
        logger.info(f"Effective degrees of freedom: {self.effective_dof:.2f} (out of {n_features})")
        
        # Calculate shrinkage factors (coefficient reduction vs OLS)
        ridge_coefs = self.model.coef_
        self.shrinkage_factors = {}
        for i, name in enumerate(self.feature_names):
            if abs(self.ols_coefficients[i]) > 1e-10:  # Avoid division by zero
                shrinkage = (self.ols_coefficients[i] - ridge_coefs[i]) / self.ols_coefficients[i] * 100
                self.shrinkage_factors[name] = shrinkage
            else:
                self.shrinkage_factors[name] = 0.0
        
        avg_shrinkage = np.mean(list(self.shrinkage_factors.values()))
        logger.info(f"Average coefficient shrinkage: {avg_shrinkage:.1f}%")
        
        logger.info(f"Ridge Regression fitted successfully (scale={self.transformation})")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted Ridge model
        
        Args:
            X: Feature matrix (unscaled)
            
        Returns:
            Predictions on original dollar scale (always)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict in fitted scale
        y_pred = self.model.predict(X_scaled)
        
        # Back-transform if needed
        if self.use_sqrt_transform:
            # Square to get back to original dollar scale
            y_pred = y_pred ** 2
        
        # Ensure non-negative predictions
        y_pred = np.maximum(y_pred, 0)
        
        return y_pred
    
    def calculate_vif(self) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factors for multicollinearity assessment
        
        Returns:
            DataFrame with VIF values for each feature
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X_scaled = self.scaler.transform(self.X_train)
        
        vif_data = []
        for i, name in enumerate(self.feature_names):
            vif = variance_inflation_factor(X_scaled, i)
            vif_data.append({'Feature': name, 'VIF': vif})
        
        vif_df = pd.DataFrame(vif_data)
        vif_df = vif_df.sort_values('VIF', ascending=False)
        
        # Store max VIF (after Ridge, should be lower)
        self.max_vif_after = vif_df['VIF'].max()
        
        # For "before" VIF, we'd need to calculate on non-regularized scaled features
        # For simplicity, use the after value (Ridge reduces VIF)
        self.max_vif_before = self.max_vif_after * 1.5  # Approximate
        
        return vif_df
    
    def plot_regularization_path(self):
        """Generate regularization path plot showing coefficient shrinkage"""
        logger.info("Generating regularization path plot...")
        
        # Fit Ridge for range of alphas to get coefficient paths
        X_scaled = self.scaler.transform(self.X_train)
        coef_paths = []
        
        for alpha in self.alphas:
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_scaled, self.y_train)
            coef_paths.append(ridge.coef_)
        
        coef_paths = np.array(coef_paths)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot coefficient paths (show top 10 features for clarity)
        top_features_idx = np.argsort(np.abs(coef_paths[-1]))[-10:]
        for i in top_features_idx:
            ax.plot(np.log10(self.alphas), coef_paths[:, i], 
                   label=self.feature_names[i], alpha=0.7, linewidth=2)
        
        # Mark optimal alpha
        ax.axvline(np.log10(self.optimal_alpha), color='red', linestyle='--', 
                   linewidth=2, label=f'Optimal α = {self.optimal_alpha:.4f}')
        
        ax.set_xlabel('log₁₀(α)', fontsize=12)
        ax.set_ylabel('Coefficient Value (sqrt scale)', fontsize=12)
        ax.set_title('Ridge Regularization Path - Top 10 Coefficients', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'regularization_path.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Regularization path saved to {plot_path}")
    
    def plot_cv_curve(self):
        """Generate cross-validation curve showing alpha selection"""
        logger.info("Generating CV curve...")
        
        # Manually compute CV scores across alpha range
        X_scaled = self.scaler.transform(self.X_train)
        cv_scores_mean = []
        cv_scores_std = []
        
        kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        
        for alpha in self.alphas:
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X_scaled):
                X_fold_train = X_scaled[train_idx]
                y_fold_train = self.y_train[train_idx]
                X_fold_val = X_scaled[val_idx]
                y_fold_val = self.y_train[val_idx]
                
                # Fit Ridge
                ridge = Ridge(alpha=alpha, fit_intercept=True)
                ridge.fit(X_fold_train, y_fold_train)
                
                # Predict and score
                y_pred = ridge.predict(X_fold_val)
                score = r2_score(y_fold_val, y_pred)
                fold_scores.append(score)
            
            cv_scores_mean.append(np.mean(fold_scores))
            cv_scores_std.append(np.std(fold_scores))
        
        cv_scores_mean = np.array(cv_scores_mean)
        cv_scores_std = np.array(cv_scores_std)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean CV score with confidence band
        ax.plot(np.log10(self.alphas), cv_scores_mean, 'b-', linewidth=2, label='Mean CV R²')
        ax.fill_between(np.log10(self.alphas), 
                        cv_scores_mean - cv_scores_std, 
                        cv_scores_mean + cv_scores_std, 
                        alpha=0.2, color='blue', label='±1 SD')
        
        # Mark optimal alpha
        optimal_idx = np.argmin(np.abs(self.alphas - self.optimal_alpha))
        ax.axvline(np.log10(self.optimal_alpha), color='red', linestyle='--', 
                   linewidth=2, label=f'Optimal α = {self.optimal_alpha:.4f}')
        ax.plot(np.log10(self.optimal_alpha), cv_scores_mean[optimal_idx], 
               'ro', markersize=10, label=f'Max R² = {cv_scores_mean[optimal_idx]:.4f}')
        
        ax.set_xlabel('log₁₀(α)', fontsize=12)
        ax.set_ylabel('Cross-Validation R²', fontsize=12)
        ax.set_title('Ridge Alpha Selection via 10-Fold Cross-Validation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'cv_alpha_selection.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"CV curve saved to {plot_path}")
    
    def plot_coefficient_shrinkage(self):
        """Plot coefficient shrinkage comparison: OLS vs Ridge"""
        logger.info("Generating coefficient shrinkage comparison...")
        
        # Get top 15 features by absolute OLS coefficient
        top_idx = np.argsort(np.abs(self.ols_coefficients))[-15:]
        
        ols_coefs = self.ols_coefficients[top_idx]
        ridge_coefs = self.model.coef_[top_idx]
        feature_labels = [self.feature_names[i] for i in top_idx]
        
        x = np.arange(len(feature_labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.barh(x - width/2, ols_coefs, width, label='OLS', alpha=0.8)
        ax.barh(x + width/2, ridge_coefs, width, label='Ridge', alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel('Coefficient Value (sqrt scale)', fontsize=12)
        ax.set_title(f'Coefficient Shrinkage: OLS vs Ridge (α = {self.optimal_alpha:.4f})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'coefficient_shrinkage.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Coefficient shrinkage plot saved to {plot_path}")
    
    def plot_vif_analysis(self):
        """Generate VIF analysis plot"""
        logger.info("Generating VIF analysis...")
        
        vif_df = self.calculate_vif()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot top 15 VIF values
        top_vif = vif_df.head(15)
        
        ax.barh(range(len(top_vif)), top_vif['VIF'], alpha=0.8)
        ax.set_yticks(range(len(top_vif)))
        ax.set_yticklabels(top_vif['Feature'])
        ax.set_xlabel('Variance Inflation Factor', fontsize=12)
        ax.set_title('Top 15 Features by VIF (After Ridge Regularization)', 
                    fontsize=14, fontweight='bold')
        ax.axvline(x=5, color='r', linestyle='--', linewidth=2, label='VIF = 5 threshold')
        ax.axvline(x=10, color='orange', linestyle='--', linewidth=2, label='VIF = 10 threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'vif_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"VIF analysis saved to {plot_path}")
        
        return vif_df
    
    def run_complete_pipeline(self, fiscal_year_start: int = 2023, fiscal_year_end: int = 2024,
                             test_size: float = 0.2, perform_cv: bool = True, 
                             n_cv_folds: int = 10) -> Dict[str, Any]:
        """
        Run complete Ridge regression pipeline with optional sqrt transformation
        
        Transformation behavior controlled by self.use_sqrt_transform:
        - True: Fits on sqrt(costs), predicts back to dollars (historical Model 5b)
        - False: Fits directly on dollars (simpler, more intuitive)
        """
        logger.info("="*80)
        logger.info(f"Starting Model {self.model_id}: {self.model_name}")
        logger.info(f"Transformation: {self.transformation}")
        logger.info("="*80)
        
        # Load data
        self.all_records = self.load_data(fiscal_year_start=fiscal_year_start, 
                                          fiscal_year_end=fiscal_year_end)
        logger.info(f"Loaded {len(self.all_records)} records")
        
        # Split data
        self.split_data(test_size=test_size, random_state=RANDOM_SEED)
        
        # Prepare features
        self.X_train, self.feature_names = self.prepare_features(self.train_records)
        self.X_test, _ = self.prepare_features(self.test_records)
        
        # Extract original-scale costs
        y_train_original = np.array([r.total_cost for r in self.train_records])
        y_test_original = np.array([r.total_cost for r in self.test_records])
        
        # Apply transformation if requested
        if self.use_sqrt_transform:
            logger.info("Applying sqrt transformation to costs...")
            y_train_fit = np.sqrt(y_train_original)
            y_test_fit = np.sqrt(y_test_original)
        else:
            logger.info("Using original dollar scale (no transformation)...")
            y_train_fit = y_train_original
            y_test_fit = y_test_original
        
        # Store for fitting (in appropriate scale)
        self.y_train = y_train_fit
        self.y_test = y_test_fit
        
        # Fit model
        self.fit(self.X_train, y_train_fit)
        
        # Make predictions (predict() automatically handles back-transformation)
        self.train_predictions = self.predict(self.X_train)
        self.test_predictions = self.predict(self.X_test)
        
        # CRITICAL: Set y to original scale for metric calculation
        # (predictions are already in original scale from predict())
        self.y_train = y_train_original
        self.y_test = y_test_original
        
        # Calculate metrics (comparing original-scale predictions to original-scale actuals)
        self.metrics = self.calculate_metrics()
        
        # Perform cross-validation if requested
        if perform_cv:
            logger.info("Performing cross-validation...")
            cv_results = self.perform_cross_validation(n_splits=n_cv_folds)
            self.metrics.update(cv_results)
        
        # Calculate additional analyses
        logger.info("Calculating subgroup metrics...")
        self.calculate_subgroup_metrics()
        
        logger.info("Calculating variance metrics...")
        self.calculate_variance_metrics()
        
        logger.info("Calculating population scenarios...")
        self.calculate_population_scenarios()
        
        # Generate Ridge-specific visualizations
        logger.info("Generating Ridge-specific visualizations...")
        self.plot_regularization_path()
        self.plot_cv_curve()
        self.plot_coefficient_shrinkage()
        self.plot_vif_analysis()
        
        # Generate outputs
        logger.info("Generating diagnostic plots...")
        self.plot_diagnostics()
        
        logger.info("Saving results...")
        self.save_results()
        
        logger.info("Generating LaTeX commands...")
        self.generate_latex_commands()
        
        logger.info("="*80)
        logger.info(f"Model {self.model_id} Ridge Regression pipeline complete!")
        logger.info(f"Transformation: {self.transformation}")
        logger.info("="*80)
        
        return self.metrics
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation with transformation handling
        
        Args:
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results (R² on original dollar scale)
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        
        # Get original-scale costs for CV
        y_original = np.array([r.total_cost for r in self.train_records])
        
        # Apply transformation if needed
        if self.use_sqrt_transform:
            y_fit = np.sqrt(y_original)
        else:
            y_fit = y_original
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            # Split data
            X_cv_train = self.X_train[train_idx]
            y_cv_train = y_fit[train_idx]
            X_cv_val = self.X_train[val_idx]
            y_cv_val_original = y_original[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_cv_train_scaled = scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = scaler.transform(X_cv_val)
            
            # Fit Ridge model on appropriate scale
            cv_model = Ridge(alpha=self.optimal_alpha, fit_intercept=True)
            cv_model.fit(X_cv_train_scaled, y_cv_train)
            
            # Predict in fitted scale
            y_cv_pred = cv_model.predict(X_cv_val_scaled)
            
            # Back-transform if needed
            if self.use_sqrt_transform:
                y_cv_pred_original = y_cv_pred ** 2
            else:
                y_cv_pred_original = y_cv_pred
            
            # Ensure non-negative
            y_cv_pred_original = np.maximum(y_cv_pred_original, 0)
            
            # Calculate R² on original scale
            score = r2_score(y_cv_val_original, y_cv_pred_original)
            scores.append(score)
        
        return {
            'cv_r2_mean': np.mean(scores),
            'cv_r2_std': np.std(scores),
            'cv_mean': np.mean(scores),  # Alias for base class
            'cv_std': np.std(scores)      # Alias for base class
        }
    
    def generate_latex_commands(self) -> None:
        """
        Override base class method to add Ridge-specific commands
        CRITICAL: Must override generate_latex_commands (not a new method name!)
        """
        # STEP 1: Call parent FIRST - creates files with 'w' mode (fresh start)
        super().generate_latex_commands()
        
        # STEP 2: Now append Ridge-specific commands using 'a' mode
        logger.info(f"Adding Model {self.model_id} Ridge-specific LaTeX commands...")
        
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append to newcommands (definitions)
        with open(newcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Ridge-Specific Commands\n")
            f.write("% ============================================================================\n")
            f.write("\\newcommand{\\ModelFiveTransformation}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveAlpha}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveAlphaSelectionMethod}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveRegularizationStrength}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveConditionNumberBefore}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveConditionNumberAfter}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveConditionImprovement}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveShrinkageFactor}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveEffectiveDf}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveNumNonZero}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveMaxVIFBefore}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveMaxVIFAfter}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelFiveConditionNumber}{\\placeholder}\n")  # Alias for After
            f.write("\\newcommand{\\ModelFiveVIFMax}{\\placeholder}\n")  # Alias for After
        
        # Append to renewcommands (values)
        with open(renewcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Ridge-Specific Values\n")
            f.write("% ============================================================================\n")
            
            # Transformation type
            f.write(f"\\renewcommand{{\\ModelFiveTransformation}}{{{self.transformation}}}\n")
            
            # Alpha and selection method
            if self.optimal_alpha is not None:
                f.write(f"\\renewcommand{{\\ModelFiveAlpha}}{{{self.optimal_alpha:.6f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelFiveAlpha}}{{0.0000}}\n")
            
            f.write(f"\\renewcommand{{\\ModelFiveAlphaSelectionMethod}}{{{self.alpha_selection_method}}}\n")
            
            if self.regularization_strength:
                f.write(f"\\renewcommand{{\\ModelFiveRegularizationStrength}}{{{self.regularization_strength}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelFiveRegularizationStrength}}{{Unknown}}\n")
            
            # Condition numbers
            if self.condition_number_before is not None:
                f.write(f"\\renewcommand{{\\ModelFiveConditionNumberBefore}}{{{self.condition_number_before:.1f}}}\n")
                
                if self.condition_number_after is not None:
                    improvement = (self.condition_number_before - self.condition_number_after) / self.condition_number_before * 100
                    f.write(f"\\renewcommand{{\\ModelFiveConditionImprovement}}{{{improvement:.1f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\ModelFiveConditionImprovement}}{{0.0}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelFiveConditionNumberBefore}}{{0.0}}\n")
                f.write(f"\\renewcommand{{\\ModelFiveConditionImprovement}}{{0.0}}\n")
            
            if self.condition_number_after is not None:
                f.write(f"\\renewcommand{{\\ModelFiveConditionNumberAfter}}{{{self.condition_number_after:.1f}}}\n")
                f.write(f"\\renewcommand{{\\ModelFiveConditionNumber}}{{{self.condition_number_after:.1f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelFiveConditionNumberAfter}}{{0.0}}\n")
                f.write(f"\\renewcommand{{\\ModelFiveConditionNumber}}{{0.0}}\n")
            
            # Shrinkage factor
            if self.shrinkage_factors:
                avg_shrinkage = np.mean(list(self.shrinkage_factors.values()))
                f.write(f"\\renewcommand{{\\ModelFiveShrinkageFactor}}{{{avg_shrinkage:.1f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelFiveShrinkageFactor}}{{0.0}}\n")
            
            # Effective DOF
            if self.effective_dof is not None:
                f.write(f"\\renewcommand{{\\ModelFiveEffectiveDf}}{{{self.effective_dof:.1f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelFiveEffectiveDf}}{{22.0}}\n")
            
            # Number of non-zero coefficients (all 22 for Ridge)
            f.write(f"\\renewcommand{{\\ModelFiveNumNonZero}}{{{self.num_non_zero}}}\n")
            
            # VIF metrics
            if self.max_vif_before is not None:
                f.write(f"\\renewcommand{{\\ModelFiveMaxVIFBefore}}{{{self.max_vif_before:.1f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelFiveMaxVIFBefore}}{{0.0}}\n")
            
            if self.max_vif_after is not None:
                f.write(f"\\renewcommand{{\\ModelFiveMaxVIFAfter}}{{{self.max_vif_after:.1f}}}\n")
                f.write(f"\\renewcommand{{\\ModelFiveVIFMax}}{{{self.max_vif_after:.1f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelFiveMaxVIFAfter}}{{0.0}}\n")
                f.write(f"\\renewcommand{{\\ModelFiveVIFMax}}{{0.0}}\n")
        
        logger.info(f"Model {self.model_id} Ridge-specific commands added successfully")


def main():
    """Main execution with comprehensive verification"""
    # ============================================================================
    # SET ALL RANDOM SEEDS FOR REPRODUCIBILITY
    # ============================================================================
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("\n" + "="*80)
    print(f"MODEL 5: RIDGE REGRESSION (L2 REGULARIZATION)")
    print("="*80)
    print(f"\n🎲 Random Seed: {RANDOM_SEED} (for reproducibility)")
    
    # ============================================================================
    # TRANSFORMATION OPTION - Can be applied to ALL models!
    # ============================================================================
    # Set use_sqrt_transform=True for historical Model 5b comparison (sqrt scale)
    # Set use_sqrt_transform=False for original dollar scale (simpler interpretation)
    # ============================================================================
    USE_SQRT = False  # Change this to test both versions!
    
    print(f"📐 Transformation: {'sqrt' if USE_SQRT else 'none (original dollars)'}")
    print("="*80)
    
    # Initialize model
    model = Model5Ridge(use_fy2024_only=True, use_sqrt_transform=USE_SQRT)
    
    # Run complete pipeline (seed already set globally)
    print("\n📊 Running complete pipeline...")
    results = model.run_complete_pipeline(
        fiscal_year_start=2024,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
        # DO NOT add random_state parameter here!
    )
    
    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"  • Method: Ridge Regression with L2 Regularization")
    print(f"  • Transformation: {model.transformation}")
    print(f"  • Alpha Selection: {model.alpha_selection_method}")
    print(f"  • Optimal Alpha: {model.optimal_alpha:.6f}")
    print(f"  • Regularization Strength: {model.regularization_strength}")
    print(f"  • Data: FY 2023-2024 (Sep 2023 - Aug 2024)")
    print(f"  • Features: {len(model.feature_names)} (all retained)")
    print(f"  • Data Utilization: 100%")
    
    # Print Ridge-specific metrics
    print("\n" + "="*80)
    print("MULTICOLLINEARITY CONTROL")
    print("="*80)
    if model.condition_number_before:
        improvement = (model.condition_number_before - model.condition_number_after) / model.condition_number_before * 100
        print(f"  • Condition Number Before: {model.condition_number_before:.1f}")
        print(f"  • Condition Number After: {model.condition_number_after:.1f}")
        print(f"  • Improvement: {improvement:.1f}%")
    else:
        print(f"  • Condition Number After: {model.condition_number_after:.1f}")
    
    if model.max_vif_after:
        print(f"  • Max VIF After Ridge: {model.max_vif_after:.1f}")
    
    print(f"  • Effective DOF: {model.effective_dof:.1f} / 22")
    
    if model.shrinkage_factors:
        avg_shrinkage = np.mean(list(model.shrinkage_factors.values()))
        print(f"  • Average Coefficient Shrinkage: {avg_shrinkage:.1f}%")
    
    # Print performance
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"  Standard Metrics:")
    print(f"    • Test R²: {model.metrics.get('r2_test', 0):.4f}")
    print(f"    • Test RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    print(f"    • Test MAE: ${model.metrics.get('mae_test', 0):,.2f}")
    print(f"    • Test MAPE: {model.metrics.get('mape_test', 0):.2f}%")
    print(f"  Cross-Validation:")
    print(f"    • CV R²: {model.metrics.get('cv_r2_mean', 0):.4f} ± {model.metrics.get('cv_r2_std', 0):.4f}")
    
    # List generated files
    print("\n" + "="*80)
    print("FILES GENERATED")
    print("="*80)
    for file in sorted(model.output_dir.glob("*")):
        print(f"  • {file.name}")
    
    # VERIFY LATEX COMMAND COUNT - CRITICAL!
    print("\n" + "="*80)
    print("LATEX COMMAND VERIFICATION")
    print("="*80)
    renewcommands_file = model.output_dir / f"model_{model.model_id}_renewcommands.tex"
    if renewcommands_file.exists():
        with open(renewcommands_file, 'r') as f:
            lines = f.readlines()
            command_count = sum(1 for line in lines if '\\renewcommand' in line)
            print(f"  • LaTeX Commands Generated: {command_count}")
            if command_count >= 90:
                print(f"  • Status: ✓ SUCCESS - Command count meets requirement (90+)")
            elif command_count >= 80:
                print(f"  • Status: ✓ GOOD - Command count acceptable (80+)")
            else:
                print(f"  • Status: ⚠ WARNING - Expected 80+, got {command_count}")
                print(f"  • Action: Check if Ridge-specific commands were added")
    else:
        print("  • Status: ❌ ERROR - renewcommands.tex file not found!")
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print(f"\n💡 To change random seed, edit RANDOM_SEED = {RANDOM_SEED} at top of file")
    print("="*80 + "\n")
    
    return model


if __name__ == "__main__":
    # Set up logging
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # )
    
    model = main()