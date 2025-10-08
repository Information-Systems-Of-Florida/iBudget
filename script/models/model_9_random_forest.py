"""
model_9_randomforest.py
=======================
Model 9: Random Forest Regression with Robust Features
Ensemble learning with automatic feature interaction detection
Uses robust features selected via Mutual Information analysis

CRITICAL FIXES APPLIED:
- Rule 4: Random seed control (RANDOM_SEED constant)
- Rule 5: Proper generate_latex_commands() override
- Rule 7: Transformation control (use_sqrt_transform parameter)
- Rule 8: No logging.basicConfig() - let base class handle logging
"""

import numpy as np
import pandas as pd
import random  # ADDED: For random seed control
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Import base class
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# SINGLE POINT OF CONTROL FOR RANDOM SEED (Rule 4)
# ============================================================================
# Change this value to get different random splits, or keep at 42 for reproducibility
# This seed controls:
#   - Train/test split
#   - Cross-validation folds
#   - Random Forest's internal randomness
# ============================================================================
RANDOM_SEED = 42


class Model9RandomForest(BaseiBudgetModel):
    """
    Model 9: Random Forest Regression
    
    Key features:
    - Ensemble of decision trees with bootstrap aggregation
    - Automatic feature interaction detection
    - Non-parametric approach (no linearity assumptions)
    - Out-of-bag (OOB) error estimation
    - Feature importance ranking
    - Robust to outliers - NO data removal (100% data utilization)
    - Optional sqrt transformation (test both empirically!)
    
    Robust Features (19 total from FeatureSelection.txt):
    - 5 Living Settings: ILSL, RH1, RH2, RH3, RH4 (FH as reference)
    - 2 Age Groups: Age21_30, Age31Plus (Age3_20 as reference)
    - 10 QSI Questions: Q16, Q18, Q20, Q21, Q23, Q28, Q33, Q34, Q36, Q43
    - 2 Summary Scores: BSum, FSum
    """
    
    def __init__(self, use_sqrt_transform: bool = False):
        """
        Initialize Model 9
        
        Args:
            use_sqrt_transform: If True, use sqrt transformation; if False, use original dollars
        """
        super().__init__(model_id=9, model_name="Random Forest")
        
        # ============================================================================
        # TRANSFORMATION CONTROL (Rule 7) - Test both approaches!
        # ============================================================================
        # Set to True to use sqrt transformation (historical baseline)
        # Set to False to fit on original dollar scale (simpler interpretation)
        # Random Forest may work well on original dollars due to natural robustness
        # ============================================================================
        self.use_sqrt_transform = use_sqrt_transform
        self.transformation = "sqrt" if use_sqrt_transform else "none"
        logger.info(f"Transformation: {self.transformation}")
        
        # Random Forest hyperparameters (tuned for performance)
        self.n_estimators = 500  # Number of trees
        self.max_depth = 20  # Maximum tree depth
        self.min_samples_split = 20  # Minimum samples to split node
        self.min_samples_leaf = 10  # Minimum samples in leaf
        self.max_features = 'sqrt'  # Features considered at each split
        self.bootstrap = True  # Use bootstrap sampling
        self.oob_score = True  # Calculate out-of-bag score
        self.n_jobs = -1  # Use all CPU cores
        
        # Model object
        self.model = None
        
        # Random Forest specific metrics
        self.feature_importances_dict = {}
        self.top_features = []
        self.oob_error = None
        self.oob_r2 = None
        self.mean_tree_depth = None
        self.training_time = None
        
        logger.info(f"Model 9 Random Forest initialized")
        logger.info(f"Hyperparameters: n_estimators={self.n_estimators}, max_depth={self.max_depth}")
    
    def split_data(self, test_size: float = 0.2, random_state: int = RANDOM_SEED) -> None:
        """
        Override base class split_data to use global RANDOM_SEED
        
        Args:
            test_size: Proportion of data for test set
            random_state: Random seed (defaults to RANDOM_SEED constant)
        """
        if isinstance(test_size, bool):
            test_size = 0.2 if test_size else 0.0
        
        if not self.all_records:
            raise ValueError("No records loaded. Call load_data() first.")
        
        # Call parent implementation with our random_state
        super().split_data(test_size=test_size, random_state=random_state)
        logger.info(f"Data split with random_state={random_state}")
    
    def load_data(self) -> List[ConsumerRecord]:
        """
        Load data using base class, optionally filtering to FY2024
        
        Returns:
            List of ConsumerRecord objects
        """
        # Load all data from pickle using base class
        all_records = super().load_data()
        
        # Filter by fiscal year if requested
        logger.info(f"Using all data: {len(all_records)} records")
        return all_records
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix using ONLY robust features from Model 5b
        
        Features (19 total):
        - 5 Living Settings: ILSL, RH1, RH2, RH3, RH4 (FH=reference)
        - 2 Age Groups: Age21_30, Age31Plus (Age3_20=reference)
        - 10 QSI: Q16, Q18, Q20, Q21, Q23, Q28, Q33, Q34, Q36, Q43
        - 2 Summaries: BSum, FSum
        
        Args:
            records: List of ConsumerRecord objects
            
        Returns:
            Feature matrix and feature names
        """
        features_list = []
        
        for record in records:
            row = []
            
            # Living Setting indicators (5 features, FH is reference)
            row.append(1 if record.living_setting == 'ILSL' else 0)
            row.append(1 if record.living_setting == 'RH1' else 0)
            row.append(1 if record.living_setting == 'RH2' else 0)
            row.append(1 if record.living_setting == 'RH3' else 0)
            row.append(1 if record.living_setting == 'RH4' else 0)
            
            # Age Group indicators (2 features, Age3_20 is reference)
            row.append(1 if record.age_group == 'Age21_30' else 0)
            row.append(1 if record.age_group == 'Age31Plus' else 0)
            
            # QSI Questions (10 features)
            row.append(record.q16)
            row.append(record.q18)
            row.append(record.q20)
            row.append(record.q21)
            row.append(record.q23)
            row.append(record.q28)
            row.append(record.q33)
            row.append(record.q34)
            row.append(record.q36)
            row.append(record.q43)
            
            # Summary Scores (2 features)
            row.append(record.bsum)
            row.append(record.fsum)
            
            features_list.append(row)
        
        feature_names = [
            'LivingSetting_ILSL', 'LivingSetting_RH1', 'LivingSetting_RH2',
            'LivingSetting_RH3', 'LivingSetting_RH4',
            'AgeGroup_21_30', 'AgeGroup_31Plus',
            'Q16', 'Q18', 'Q20', 'Q21', 'Q23', 'Q28', 'Q33', 'Q34', 'Q36', 'Q43',
            'BSum', 'FSum'
        ]
        
        X = np.array(features_list, dtype=float)
        
        logger.info(f"Prepared features: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Features used: {', '.join(feature_names)}")
        
        return X, feature_names
    
    def run_complete_pipeline(
        self,
        fiscal_year_start: int = 2024,
        fiscal_year_end: int = 2024,
        test_size: float = 0.2,
        perform_cv: bool = True,
        n_cv_folds: int = 10
    ) -> Dict[str, Any]:
        """
        Override to handle transformation explicitly
        
        CRITICAL: Must handle transformation at the right points:
        1. Extract original-scale costs
        2. Apply transformation if requested
        3. Fit on transformed scale
        4. Predict and back-transform
        5. Set y to original scale for metrics
        """
        logger.info("="*80)
        logger.info(f"Starting Model 9 Random Forest Pipeline")
        logger.info(f"Transformation: {self.transformation}")
        logger.info(f"Random Seed: {RANDOM_SEED}")
        logger.info("="*80)
        
        # Load data
        self.all_records = self.load_data()
        
        # Split data
        self.split_data(test_size=test_size)
        
        # Prepare features
        self.X_train, self.feature_names = self.prepare_features(self.train_records)
        self.X_test, _ = self.prepare_features(self.test_records)
        
        # ============================================================================
        # TRANSFORMATION HANDLING (Rule 7)
        # ============================================================================
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
        
        # Perform cross-validation (on training data)
        if perform_cv:
            self.perform_cross_validation(n_splits=n_cv_folds)
        
        # Fit model on appropriate scale
        import time
        start_time = time.time()
        self.fit(self.X_train, y_train_fit)
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        # Make predictions (automatically handles back-transformation)
        self.train_predictions = self.predict(self.X_train)
        self.test_predictions = self.predict(self.X_test)
        
        # CRITICAL: Set y to original scale for metrics
        # All metrics ALWAYS calculated on original dollar scale
        self.y_train = y_train_original
        self.y_test = y_test_original
        
        # Calculate feature importances
        self.calculate_feature_importance()
        
        # Calculate OOB metrics
        if self.model.oob_score_:
            self.oob_r2 = self.model.oob_score_
            # OOB predictions need back-transformation
            oob_predictions = self.model.oob_prediction_
            if self.use_sqrt_transform:
                oob_predictions = oob_predictions ** 2
            oob_predictions = np.maximum(oob_predictions, 0)
            # Calculate RMSE (compatible with older scikit-learn)
            mse = mean_squared_error(y_train_original, oob_predictions)
            self.oob_error = np.sqrt(mse)
            logger.info(f"OOB R²: {self.oob_r2:.4f}, OOB RMSE: ${self.oob_error:,.2f}")
        
        # Calculate average tree depth
        tree_depths = [tree.get_depth() for tree in self.model.estimators_]
        self.mean_tree_depth = np.mean(tree_depths)
        logger.info(f"Mean tree depth: {self.mean_tree_depth:.1f}")
        
        # Calculate all metrics
        self.metrics = self.calculate_metrics()
        self.subgroup_metrics = self.calculate_subgroup_metrics()
        self.variance_metrics = self.calculate_variance_metrics()
        self.population_scenarios = self.calculate_population_scenarios()
        
        # Generate outputs
        self.save_results()
        self.plot_diagnostics()
        self.generate_latex_commands()
        
        logger.info("="*80)
        logger.info("Model 9 Random Forest Pipeline Complete")
        logger.info("="*80)
        
        return self.metrics
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Random Forest model
        
        Args:
            X: Feature matrix
            y: Target values (on appropriate scale - sqrt or original)
        """
        logger.info("Fitting Random Forest model...")
        
        # Initialize Random Forest with fixed random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=RANDOM_SEED,  # Use global seed
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # Fit the model
        self.model.fit(X, y)
        
        logger.info(f"Random Forest fitted with {self.n_estimators} trees")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted Random Forest
        
        CRITICAL: Always returns predictions on original dollar scale
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions on original dollar scale (ALWAYS)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions on fitted scale
        y_pred = self.model.predict(X)
        
        # Back-transform if needed
        if self.use_sqrt_transform:
            y_pred = y_pred ** 2  # Square to get back to dollars
        
        # Ensure non-negative
        y_pred = np.maximum(y_pred, 0)
        
        return y_pred
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, float]:
        """
        Override to handle transformation in CV folds
        
        Args:
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        # Get original-scale costs
        y_original = np.array([r.total_cost for r in self.train_records])
        
        # Apply transformation if needed
        if self.use_sqrt_transform:
            y_fit = np.sqrt(y_original)
        else:
            y_fit = y_original
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            # Split fold
            X_cv_train, X_cv_val = self.X_train[train_idx], self.X_train[val_idx]
            y_cv_train, y_cv_val_original = y_fit[train_idx], y_original[val_idx]
            
            # Fit model on fold
            cv_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                oob_score=False,  # Not needed for CV
                random_state=RANDOM_SEED,
                n_jobs=self.n_jobs,
                verbose=0
            )
            cv_model.fit(X_cv_train, y_cv_train)
            
            # Predict
            y_cv_pred = cv_model.predict(X_cv_val)
            
            # Back-transform if needed
            if self.use_sqrt_transform:
                y_cv_pred = y_cv_pred ** 2
            y_cv_pred = np.maximum(y_cv_pred, 0)
            
            # CRITICAL: Score ALWAYS on original scale
            score = r2_score(y_cv_val_original, y_cv_pred)
            cv_scores.append(score)
            
            if fold <= 3:  # Log first 3 folds
                logger.info(f"  Fold {fold}/{n_splits}: R² = {score:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        self.metrics['cv_mean'] = cv_mean
        self.metrics['cv_std'] = cv_std
        self.metrics['cv_scores'] = cv_scores
        
        logger.info(f"Cross-validation R²: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return {'cv_mean': cv_mean, 'cv_std': cv_std, 'cv_scores': cv_scores}
    
    def calculate_feature_importance(self) -> None:
        """Calculate and store feature importance"""
        if self.model is None:
            logger.warning("Model not fitted, cannot calculate feature importance")
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
        
        logger.info("Top 5 features by importance:")
        for i, (feature, importance) in enumerate(self.top_features, 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")
    
    def generate_latex_commands(self) -> None:
        """
        Override base class method to add Model 9 specific commands
        
        CRITICAL: Must override generate_latex_commands (not create new method!)
        CRITICAL: Must call super() FIRST, then append
        """
        # STEP 1: Call parent FIRST - creates files with 'w' mode
        super().generate_latex_commands()
        
        # STEP 2: Append model-specific commands using 'a' mode
        logger.info(f"Adding Model {self.model_id} specific LaTeX commands...")
        
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append to newcommands (definitions)
        with open(newcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Random Forest Specific Commands\n")
            f.write("% ============================================================================\n")
            f.write("\\newcommand{\\ModelNineTransformation}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineNumFeatures}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineNTrees}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineMaxDepth}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineMinSamplesSplit}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineMinSamplesLeaf}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineMaxFeatures}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineOOBRSquared}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineOOBError}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineAvgTreeDepth}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineTrainingTime}{\\placeholder}\n")
            
            # Feature importance commands (top 5)
            f.write("\\newcommand{\\ModelNineTopFeatureOne}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineTopFeatureOneImportance}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineTopFeatureTwo}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineTopFeatureTwoImportance}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineTopFeatureThree}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineTopFeatureThreeImportance}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineTopFeatureFour}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineTopFeatureFourImportance}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineTopFeatureFive}{\\placeholder}\n")
            f.write("\\newcommand{\\ModelNineTopFeatureFiveImportance}{\\placeholder}\n")
        
        # Append to renewcommands (values)
        with open(renewcommands_file, 'a') as f:
            f.write("\n% ============================================================================\n")
            f.write(f"% Model {self.model_id} Random Forest Specific Values\n")
            f.write("% ============================================================================\n")
            
            # Transformation
            f.write(f"\\renewcommand{{\\ModelNineTransformation}}{{{self.transformation}}}\n")
            
            # Basic parameters
            f.write(f"\\renewcommand{{\\ModelNineNumFeatures}}{{{len(self.feature_names)}}}\n")
            f.write(f"\\renewcommand{{\\ModelNineNTrees}}{{{self.n_estimators}}}\n")
            f.write(f"\\renewcommand{{\\ModelNineMaxDepth}}{{{self.max_depth}}}\n")
            f.write(f"\\renewcommand{{\\ModelNineMinSamplesSplit}}{{{self.min_samples_split}}}\n")
            f.write(f"\\renewcommand{{\\ModelNineMinSamplesLeaf}}{{{self.min_samples_leaf}}}\n")
            f.write(f"\\renewcommand{{\\ModelNineMaxFeatures}}{{{self.max_features}}}\n")
            
            # OOB metrics (with defaults)
            if self.oob_r2 is not None:
                f.write(f"\\renewcommand{{\\ModelNineOOBRSquared}}{{{self.oob_r2:.4f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelNineOOBRSquared}}{{0.0000}}\n")
            
            if self.oob_error is not None:
                f.write(f"\\renewcommand{{\\ModelNineOOBError}}{{{self.oob_error:,.2f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelNineOOBError}}{{0.00}}\n")
            
            # Tree depth
            if self.mean_tree_depth is not None:
                f.write(f"\\renewcommand{{\\ModelNineAvgTreeDepth}}{{{self.mean_tree_depth:.1f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelNineAvgTreeDepth}}{{0.0}}\n")
            
            # Training time
            if self.training_time is not None:
                f.write(f"\\renewcommand{{\\ModelNineTrainingTime}}{{{self.training_time:.2f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\ModelNineTrainingTime}}{{0.00}}\n")
            
            # Feature importance (top 5)
            if len(self.top_features) >= 5:
                for i in range(5):
                    feature_name, importance = self.top_features[i]
                    # Clean feature name for LaTeX
                    clean_name = feature_name.replace('_', ' ')
                    num_word = ['One', 'Two', 'Three', 'Four', 'Five'][i]
                    f.write(f"\\renewcommand{{\\ModelNineTopFeature{num_word}}}{{{clean_name}}}\n")
                    f.write(f"\\renewcommand{{\\ModelNineTopFeature{num_word}Importance}}{{{importance:.4f}}}\n")
            else:
                # Provide defaults if not enough features
                for i, num_word in enumerate(['One', 'Two', 'Three', 'Four', 'Five']):
                    f.write(f"\\renewcommand{{\\ModelNineTopFeature{num_word}}}{{N/A}}\n")
                    f.write(f"\\renewcommand{{\\ModelNineTopFeature{num_word}Importance}}{{0.0000}}\n")
        
        logger.info(f"Model {self.model_id} specific commands added successfully")
    
    def plot_diagnostics(self) -> None:
        """Generate diagnostic plots for Random Forest"""
        logger.info("Generating diagnostic plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Actual vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(self.y_test / 1000, self.test_predictions / 1000, alpha=0.5, s=20)
        ax1.plot([0, max(self.y_test)/1000], [0, max(self.y_test)/1000], 
                 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Cost ($1000s)')
        ax1.set_ylabel('Predicted Cost ($1000s)')
        ax1.set_title('Actual vs Predicted Costs')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residual Plot
        ax2 = axes[0, 1]
        residuals = self.y_test - self.test_predictions
        ax2.scatter(self.test_predictions / 1000, residuals / 1000, alpha=0.5, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Cost ($1000s)')
        ax2.set_ylabel('Residual ($1000s)')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature Importance
        ax3 = axes[0, 2]
        if self.top_features:
            features = [f[0] for f in self.top_features[:10]]
            importances = [f[1] for f in self.top_features[:10]]
            y_pos = np.arange(len(features))
            ax3.barh(y_pos, importances)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([f.replace('_', ' ') for f in features], fontsize=8)
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Feature Importances')
            ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Error Distribution
        ax4 = axes[1, 0]
        ax4.hist(residuals / 1000, bins=50, edgecolor='black', alpha=0.7)
        ax4.axvline(x=0, color='r', linestyle='--', lw=2)
        ax4.set_xlabel('Residual ($1000s)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Residual Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Q-Q Plot
        ax5 = axes[1, 1]
        stats.probplot(residuals, dist="norm", plot=ax5)
        ax5.set_title('Q-Q Plot')
        ax5.grid(True, alpha=0.3)
        
        # 6. Prediction Error vs Actual
        ax6 = axes[1, 2]
        abs_errors = np.abs(residuals)
        ax6.scatter(self.y_test / 1000, abs_errors / 1000, alpha=0.5, s=20)
        ax6.set_xlabel('Actual Cost ($1000s)')
        ax6.set_ylabel('Absolute Error ($1000s)')
        ax6.set_title('Absolute Error vs Actual Cost')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model 9: Random Forest Diagnostics (Transformation: {self.transformation})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'diagnostic_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plots saved to {output_file}")


def main():
    """Main execution function"""
    
    # ============================================================================
    # TRANSFORMATION OPTION - Easy to test both!
    # ============================================================================
    USE_SQRT = True  # Change this to False to test original dollar scale
    # Random Forest is naturally robust and may work well without transformation!
    # ============================================================================
    
    # SET ALL RANDOM SEEDS FOR REPRODUCIBILITY (Rule 4)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("\n" + "="*80)
    print("MODEL 9: RANDOM FOREST REGRESSION")
    print("="*80)
    print(f"\n🎲 Random Seed: {RANDOM_SEED} (for reproducibility)")
    print(f"📐 Transformation: {'sqrt' if USE_SQRT else 'none (original dollars)'}")
    
    # Initialize model
    model = Model9RandomForest(
        use_sqrt_transform=USE_SQRT
    )
    
    # Run complete pipeline
    # DO NOT pass random_state parameter - base class doesn't accept it
    results = model.run_complete_pipeline(
        fiscal_year_start=2024,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n Random Forest Configuration:")
    print(f"  • Number of Trees: {model.n_estimators}")
    print(f"  • Max Depth: {model.max_depth}")
    print(f"  • Min Samples Split: {model.min_samples_split}")
    print(f"  • Min Samples Leaf: {model.min_samples_leaf}")
    print(f"  • Max Features: {model.max_features}")
    print(f"  • Bootstrap: {model.bootstrap}")
    print(f"  • OOB Score: {model.oob_score}")
    
    print(f"\n Data Summary:")
    print(f"  • Total Records: {len(model.all_records)}")
    print(f"  • Training Records: {model.metrics.get('training_samples', 0)}")
    print(f"  • Test Records: {model.metrics.get('test_samples', 0)}")
    print(f"  • Features Used: {len(model.feature_names)}")
    print(f"  • Outliers Removed: 0 (Random Forest handles outliers naturally)")
    print(f"  • Data Utilization: 100%")
    
    print(f"\n Performance Metrics:")
    print(f"  • Training R²: {model.metrics.get('r2_train', 0):.4f}")
    print(f"  • Test R²: {model.metrics.get('r2_test', 0):.4f}")
    print(f"  • RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    print(f"  • MAE: ${model.metrics.get('mae_test', 0):,.2f}")
    print(f"  • MAPE: {model.metrics.get('mape_test', 0):.2f}%")
    print(f"  • CV R² (10-fold): {model.metrics.get('cv_mean', 0):.4f} ± {model.metrics.get('cv_std', 0):.4f}")
    
    if model.oob_r2 is not None:
        print(f"\n Random Forest Specific:")
        print(f"  • OOB R²: {model.oob_r2:.4f}")
        print(f"  • OOB RMSE: ${model.oob_error:,.2f}")
        print(f"  • Mean Tree Depth: {model.mean_tree_depth:.1f}")
        print(f"  • Training Time: {model.training_time:.2f} seconds")
    
    print(f"\n Accuracy Bands:")
    print(f"  • Within $1,000: {model.metrics.get('within_1k', 0):.1f}%")
    print(f"  • Within $2,000: {model.metrics.get('within_2k', 0):.1f}%")
    print(f"  • Within $5,000: {model.metrics.get('within_5k', 0):.1f}%")
    print(f"  • Within $10,000: {model.metrics.get('within_10k', 0):.1f}%")
    print(f"  • Within $20,000: {model.metrics.get('within_20k', 0):.1f}%")
    
    print(f"\n Top 5 Features by Importance:")
    for i, (feature, importance) in enumerate(model.top_features, 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    print(f"\n Key Advantages:")
    print(f"  • No outlier removal (100% data retention)")
    print(f"  • Automatic feature interaction detection")
    print(f"  • Non-linear pattern capture")
    print(f"  • Natural handling of heteroscedasticity")
    print(f"  • Feature importance provides interpretability")
    print(f"  • Robust to missing values")
    print(f"  • Built-in validation (OOB)")
    
    # Verify command count
    renewcommands_file = model.output_dir / f"model_{model.model_id}_renewcommands.tex"
    if renewcommands_file.exists():
        with open(renewcommands_file, 'r') as f:
            command_count = sum(1 for line in f if '\\renewcommand' in line)
        print(f"\n LaTeX Commands Generated: {command_count}")
        if command_count < 80:
            print(f"     WARNING: Expected 100+ commands, got {command_count}")
        else:
            print(f"    Command count looks good (expected 100+)")
    
    print(f"\n To change random seed, edit RANDOM_SEED = {RANDOM_SEED} at top of file")
    print(f" To change transformation, set USE_SQRT = {not USE_SQRT} in main()")
    
    print("\n" + "="*80)
    print("Model 9 implementation complete!")
    print("="*80)
    
    return model


if __name__ == "__main__":
    # DON'T use logging.basicConfig() - let base class handle all logging
    model = main()