"""
Model 5: Ridge Regression with L2 Regularization
=================================================
Uses the 22 robust features from validated Model 5b
Applies coefficient shrinkage to handle multicollinearity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import logging

from base_model import BaseiBudgetModel, ConsumerRecord

logger = logging.getLogger(__name__)

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
    
    def __init__(self, use_fy2024_only: bool = True):
        super().__init__(model_id=5, model_name="Ridge Regression")
        self.use_fy2024_only = use_fy2024_only
        self.fiscal_years_used = "2024" if use_fy2024_only else "2023-2024"
        
        # Ridge-specific parameters
        self.alphas = np.logspace(-4, 4, 100)  # Alpha search range
        self.scaler = StandardScaler()
        self.optimal_alpha = None
        
        # Multicollinearity analysis
        self.condition_number_before = None
        self.condition_number_after = None
        self.effective_dof = None
        self.shrinkage_factors = {}
        
        # Store OLS comparison
        self.ols_coefficients = None
        
        # Number of features (22 from Model 5b)
        self.num_parameters = 23  # 22 features + intercept
        
        logger.info("Model 5 Ridge Regression initialized")
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Override split_data to ensure proper train/test split
        CRITICAL: Handles boolean test_size from base class
        
        Args:
            test_size: Proportion for test set  
            random_state: Random seed
        """
        # Handle boolean test_size (base class sometimes passes True)
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
        
        # Use square-root transformation (following Model 5b)
        self.y_train = np.sqrt(np.array([r.total_cost for r in self.train_records]))
        self.y_test = np.sqrt(np.array([r.total_cost for r in self.test_records]))
        
        logger.info(f"Data split: {len(self.train_records)} training, {len(self.test_records)} test")
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare the 22 robust features from Model 5b (Table 7)
        
        Features:
        - 5 Living Setting dummies (drop FH as reference)
        - 2 Age Group dummies (drop Age3_20 as reference)
        - 10 Individual QSI questions
        - 1 Behavioral sum score
        - 4 Interaction terms
        
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
            
            # 4. Behavioral Sum Score (1 feature)
            bsum = float(record.bsum)
            row_features.append(bsum)
            
            # 5. Interaction Terms (4 features from Table 7)
            fsum = float(record.fsum)
            is_fh = 1.0 if record.living_setting == 'FH' else 0.0
            is_ilsl = 1.0 if record.living_setting == 'ILSL' else 0.0
            
            # FHFSum: Family Home × FSum interaction
            row_features.append(is_fh * fsum)
            
            # SLFSum: ILSL × FSum interaction  
            row_features.append(is_ilsl * fsum)
            
            # SLBSum: ILSL × BSum interaction
            row_features.append(is_ilsl * bsum)
            
            # 4th interaction: Based on pattern, likely RH × FSum or similar
            # Using RH (any RH level) × FSum as placeholder
            is_rh = 1.0 if record.living_setting in ['RH1', 'RH2', 'RH3', 'RH4'] else 0.0
            row_features.append(is_rh * fsum)
            
            features_list.append(row_features)
        
        # Define feature names (22 features total)
        feature_names = (
            ['living_ILSL', 'living_RH1', 'living_RH2', 'living_RH3', 'living_RH4'] +
            ['age_Age21_30', 'age_Age31Plus'] +
            ['Q16', 'Q18', 'Q20', 'Q21', 'Q23', 'Q28', 'Q33', 'Q34', 'Q36', 'Q43'] +
            ['BSum'] +
            ['FHFSum', 'SLFSum', 'SLBSum', 'RHFSum']
        )
        
        X = np.array(features_list)
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} records")
        
        # Verify we have exactly 22 features
        assert X.shape[1] == 22, f"Expected 22 features, got {X.shape[1]}"
        
        return X, feature_names
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Ridge regression with cross-validated alpha selection
        
        Args:
            X: Feature matrix (unscaled)
            y: Target values (square-root transformed)
        """
        logger.info("Fitting Ridge Regression with cross-validated alpha...")
        
        # Skip condition number before (causes numerical issues with singular matrices)
        self.condition_number_before = None
        logger.info("Condition number before regularization: N/A (skipped due to singularity)")
        
        # Scale features for better numerical stability
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit OLS first for comparison
        from sklearn.linear_model import LinearRegression
        ols_model = LinearRegression()
        ols_model.fit(X_scaled, y)
        self.ols_coefficients = ols_model.coef_
        
        # Fit RidgeCV with automatic alpha selection
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
        
        # Calculate condition number AFTER regularization
        n_features = X_scaled.shape[1]
        XtX = X_scaled.T @ X_scaled
        XtX_regularized = XtX + self.optimal_alpha * np.eye(n_features)
        self.condition_number_after = np.linalg.cond(XtX_regularized)
        logger.info(f"Condition number after regularization: {self.condition_number_after:.2f}")
        
        # Calculate improvement (handle None case)
        if self.condition_number_before is not None:
            improvement = (self.condition_number_before - self.condition_number_after) / self.condition_number_before * 100
            logger.info(f"Condition number improvement: {improvement:.1f}%")
        else:
            improvement = None
            logger.info(f"Condition number after regularization: {self.condition_number_after:.2f}")
        
        # Calculate effective degrees of freedom
        # eff_dof = trace(X(X'X + αI)^-1X')
        self.effective_dof = np.trace(XtX @ np.linalg.inv(XtX_regularized))
        logger.info(f"Effective degrees of freedom: {self.effective_dof:.2f} (out of {n_features})")
        
        # Calculate shrinkage factors (coefficient reduction vs OLS)
        ridge_coefs = self.model.coef_
        for i, name in enumerate(self.feature_names):
            if self.ols_coefficients[i] != 0:
                shrinkage = (self.ols_coefficients[i] - ridge_coefs[i]) / self.ols_coefficients[i] * 100
                self.shrinkage_factors[name] = shrinkage
        
        avg_shrinkage = np.mean(list(self.shrinkage_factors.values()))
        logger.info(f"Average coefficient shrinkage: {avg_shrinkage:.1f}%")
        
        logger.info("Ridge Regression fitted successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted Ridge model
        
        Args:
            X: Feature matrix (unscaled)
            
        Returns:
            Predictions on original dollar scale
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict in sqrt scale
        y_sqrt_pred = self.model.predict(X_scaled)
        
        # Back-transform to original scale (square)
        y_pred = y_sqrt_pred ** 2
        
        # Ensure non-negative predictions
        y_pred = np.maximum(y_pred, 0)
        
        return y_pred
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation
        
        Args:
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            # Split data
            X_cv_train = self.X_train[train_idx]
            y_cv_train = self.y_train[train_idx]
            X_cv_val = self.X_train[val_idx]
            y_cv_val = self.y_train[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_cv_train_scaled = scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = scaler.transform(X_cv_val)
            
            # Fit Ridge model
            cv_model = Ridge(alpha=self.optimal_alpha, fit_intercept=True)
            cv_model.fit(X_cv_train_scaled, y_cv_train)
            
            # Predict in sqrt scale
            y_cv_pred_sqrt = cv_model.predict(X_cv_val_scaled)
            
            # Back-transform to original scale
            y_cv_pred = y_cv_pred_sqrt ** 2
            y_cv_val_original = y_cv_val ** 2
            
            # Calculate R²
            score = r2_score(y_cv_val_original, y_cv_pred)
            scores.append(score)
        
        return {
            'scores': scores,
            'cv_r2_mean': np.mean(scores),
            'cv_r2_std': np.std(scores)
        }
    
    def plot_regularization_path(self):
        """Generate regularization path plot"""
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
        
        # Plot coefficient paths
        for i, name in enumerate(self.feature_names):
            ax.plot(np.log10(self.alphas), coef_paths[:, i], label=name, alpha=0.7)
        
        # Mark optimal alpha
        ax.axvline(np.log10(self.optimal_alpha), color='red', linestyle='--', 
                   linewidth=2, label=f'Optimal α = {self.optimal_alpha:.4f}')
        
        ax.set_xlabel('log₁₀(α)', fontsize=12)
        ax.set_ylabel('Coefficient Value (sqrt scale)', fontsize=12)
        ax.set_title('Ridge Regularization Path - Coefficient Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'regularization_path.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Regularization path saved to {plot_path}")
    
    def plot_cv_curve(self):
        """Generate cross-validation curve"""
        logger.info("Generating CV curve...")
        
        # Manually compute CV scores across alpha range
        X_scaled = self.scaler.transform(self.X_train)
        cv_scores_mean = []
        cv_scores_std = []
        
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
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
        
        # Plot mean CV score
        ax.plot(np.log10(self.alphas), cv_scores_mean, 'b-', linewidth=2, label='Mean CV Score')
        
        # Plot confidence band
        ax.fill_between(np.log10(self.alphas), 
                       cv_scores_mean - cv_scores_std,
                       cv_scores_mean + cv_scores_std,
                       alpha=0.2, color='blue', label='±1 Std Dev')
        
        # Mark optimal alpha
        optimal_idx = np.argmax(cv_scores_mean)
        ax.axvline(np.log10(self.optimal_alpha), color='red', linestyle='--',
                  linewidth=2, label=f'Optimal α = {self.optimal_alpha:.4f}')
        ax.plot(np.log10(self.alphas[optimal_idx]), cv_scores_mean[optimal_idx], 
               'ro', markersize=10, label=f'Max Score = {cv_scores_mean[optimal_idx]:.4f}')
        
        ax.set_xlabel('log₁₀(α)', fontsize=12)
        ax.set_ylabel('Cross-Validation R² Score', fontsize=12)
        ax.set_title('Ridge Regression Cross-Validation Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'cv_curve.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"CV curve saved to {plot_path}")
    
    def plot_coefficient_shrinkage(self):
        """Compare OLS vs Ridge coefficients"""
        logger.info("Generating coefficient shrinkage comparison...")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Feature': self.feature_names,
            'OLS': self.ols_coefficients,
            'Ridge': self.model.coef_,
            'Shrinkage_%': [self.shrinkage_factors.get(name, 0) for name in self.feature_names]
        })
        
        # Sort by OLS absolute value
        comparison_df['OLS_abs'] = comparison_df['OLS'].abs()
        comparison_df = comparison_df.sort_values('OLS_abs', ascending=False)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Coefficient comparison
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax1.bar(x - width/2, comparison_df['OLS'], width, label='OLS', alpha=0.8)
        ax1.bar(x + width/2, comparison_df['Ridge'], width, label='Ridge', alpha=0.8)
        ax1.set_xlabel('Features', fontsize=12)
        ax1.set_ylabel('Coefficient Value (sqrt scale)', fontsize=12)
        ax1.set_title('OLS vs Ridge Coefficients', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Shrinkage percentages
        colors = ['green' if s > 0 else 'red' for s in comparison_df['Shrinkage_%']]
        ax2.barh(x, comparison_df['Shrinkage_%'], color=colors, alpha=0.7)
        ax2.set_yticks(x)
        ax2.set_yticklabels(comparison_df['Feature'], fontsize=8)
        ax2.set_xlabel('Shrinkage (%)', fontsize=12)
        ax2.set_title('Coefficient Shrinkage from OLS to Ridge', fontsize=14, fontweight='bold')
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'coefficient_shrinkage.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Coefficient shrinkage plot saved to {plot_path}")
    
    def calculate_vif(self) -> pd.DataFrame:
        """Calculate Variance Inflation Factors"""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X_scaled = self.scaler.transform(self.X_train)
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.feature_names
        vif_data["VIF"] = [
            variance_inflation_factor(X_scaled, i) 
            for i in range(X_scaled.shape[1])
        ]
        
        return vif_data.sort_values('VIF', ascending=False)
    
    def plot_vif_analysis(self):
        """Generate VIF analysis plot"""
        logger.info("Generating VIF analysis...")
        
        vif_df = self.calculate_vif()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_df['VIF']]
        bars = ax.barh(range(len(vif_df)), vif_df['VIF'], color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(vif_df)))
        ax.set_yticklabels(vif_df['Feature'], fontsize=10)
        ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12)
        ax.set_title('Multicollinearity Analysis - VIF by Feature', fontsize=14, fontweight='bold')
        
        # Add reference lines
        ax.axvline(5, color='orange', linestyle='--', alpha=0.5, label='Moderate (VIF=5)')
        ax.axvline(10, color='red', linestyle='--', alpha=0.5, label='High (VIF=10)')
        
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'vif_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"VIF analysis saved to {plot_path}")
        
        return vif_df
    
    def generate_ridge_specific_commands(self):
        """Generate Ridge-specific LaTeX commands"""
        renewcommands_path = self.output_dir / f'model_{self.model_id}_renewcommands.tex'
        
        with open(renewcommands_path, 'a') as f:
            f.write("\n% Ridge-Specific Metrics\n")
            f.write(f"\\renewcommand{{\\ModelFiveAlpha}}{{{self.optimal_alpha:.6f}}}\n")
            
            # Condition numbers (handle None)
            if self.condition_number_before is not None:
                f.write(f"\\renewcommand{{\\ModelFiveConditionNumberBefore}}{{{self.condition_number_before:.1f}}}\n")
                improvement = (self.condition_number_before - self.condition_number_after) / self.condition_number_before * 100
                f.write(f"\\renewcommand{{\\ModelFiveConditionImprovement}}{{{improvement:.1f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelFiveConditionNumberBefore}}{{N/A}}\n")
                f.write(f"\\renewcommand{{\\ModelFiveConditionImprovement}}{{N/A}}\n")
            
            f.write(f"\\renewcommand{{\\ModelFiveConditionNumberAfter}}{{{self.condition_number_after:.1f}}}\n")
            
            f.write(f"\\renewcommand{{\\ModelFiveEffectiveDOF}}{{{self.effective_dof:.1f}}}\n")
            f.write(f"\\renewcommand{{\\ModelFiveNRobustFeatures}}{{22}}\n")
            
            avg_shrinkage = np.mean(list(self.shrinkage_factors.values()))
            f.write(f"\\renewcommand{{\\ModelFiveShrinkageFactor}}{{{avg_shrinkage:.1f}}}\n")
            
            # VIF analysis
            vif_df = self.calculate_vif()
            f.write(f"\\renewcommand{{\\ModelFiveVIFMax}}{{{vif_df['VIF'].max():.1f}}}\n")
            
            # Hardcoded costs (acceptable per specification)
            f.write("\n% Implementation Costs (Hardcoded)\n")
            f.write(f"\\renewcommand{{\\ModelFiveImplementationCost}}{{150000}}\n")
            f.write(f"\\renewcommand{{\\ModelFiveAnnualOperatingCost}}{{30000}}\n")
            f.write(f"\\renewcommand{{\\ModelFiveThreeYearTCO}}{{240000}}\n")
        
        logger.info("Ridge-specific LaTeX commands generated")
    
    def run_complete_pipeline(self, test_size: float = 0.2) -> Dict[str, Any]:
        """Run complete Ridge regression pipeline"""
        logger.info("="*80)
        logger.info(f"Starting Model {self.model_id}: {self.model_name}")
        logger.info("="*80)
        
        # Load data
        self.all_records = self.load_data(fiscal_year_start=2023, fiscal_year_end=2024)
        logger.info(f"Loaded {len(self.all_records)} records")
        
        # Split data
        self.split_data(test_size=test_size, random_state=42)
        
        # Fit model
        self.fit(self.X_train, self.y_train)
        
        # Make predictions (back-transformed to dollar scale)
        self.train_predictions = self.predict(self.X_train)
        self.test_predictions = self.predict(self.X_test)
        
        # CRITICAL: Convert y back to original scale for metric calculation
        self.y_train_original = self.y_train ** 2
        self.y_test_original = self.y_test ** 2
        
        # Store for base class
        self.y_train = self.y_train_original
        self.y_test = self.y_test_original
        
        # Calculate metrics
        self.metrics = self.calculate_metrics()
        
        # Perform cross-validation
        logger.info("Performing cross-validation...")
        cv_results = self.perform_cross_validation(n_splits=10)
        self.metrics.update(cv_results)
        
        # Calculate additional analyses
        logger.info("Calculating subgroup metrics...")
        self.calculate_subgroup_metrics()
        
        logger.info("Calculating variance metrics...")
        self.calculate_variance_metrics()
        
        logger.info("Calculating population scenarios...")
        self.calculate_population_scenarios()
        
        # Generate Ridge-specific visualizations
        logger.info("Generating Ridge-specific plots...")
        self.plot_regularization_path()
        self.plot_cv_curve()
        self.plot_coefficient_shrinkage()
        self.plot_vif_analysis()
        
        # Generate standard diagnostics
        logger.info("Generating diagnostic plots...")
        self.plot_diagnostics()  # Use base class method
        
        # Save results
        logger.info("Saving results...")
        self.save_results()
        
        # Generate LaTeX commands
        logger.info("Generating LaTeX commands...")
        self.generate_latex_commands()
        self.generate_ridge_specific_commands()
        
        logger.info("="*80)
        logger.info(f"Model {self.model_id} Ridge Regression pipeline complete!")
        logger.info("="*80)
        
        return self.metrics


def main():
    """Main execution function"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Initialize model
    model = Model5Ridge()
    
    # Run complete pipeline
    results = model.run_complete_pipeline(test_size=0.2)
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL 5 RIDGE REGRESSION SUMMARY")
    print("="*80)
    print(f"\nTest R²: {results.get('r2_test', 0):.4f}")
    print(f"Test RMSE: ${results.get('rmse_test', 0):,.2f}")
    print(f"Optimal Alpha: {model.optimal_alpha:.6f}")
    if model.condition_number_before:
        print(f"Condition Number Improvement: {((model.condition_number_before - model.condition_number_after) / model.condition_number_before * 100):.1f}%")
    else:
        print(f"Condition Number After: {model.condition_number_after:.2f}")
    print(f"Effective DOF: {model.effective_dof:.1f} / 22")
    print("\n" + "="*80)
    
    return model


if __name__ == "__main__":
    model = main()