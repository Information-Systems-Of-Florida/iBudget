"""
model_9_randomforest.py
=======================
Model 9: Random Forest Regression
Ensemble learning with automatic feature interaction detection
Uses robust features selected via Mutual Information analysis (FeatureSelection.txt)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

# Import base class
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

class Model9RandomForest(BaseiBudgetModel):
    """
    Model 9: Random Forest Regression
    
    Key features:
    - Ensemble of decision trees with bootstrap aggregation
    - Automatic feature interaction detection
    - Non-parametric approach (no linearity assumptions)
    - Out-of-bag (OOB) error estimation
    - Feature importance ranking
    - Robust to outliers - NO data removal
    """
    
    def __init__(self, use_fy2024_only: bool = True):
        """Initialize Model 9"""
        super().__init__(model_id=9, model_name="Random Forest")
        self.use_fy2024_only = use_fy2024_only
        self.fiscal_years_used = "2024" if use_fy2024_only else "2023-2024"
        
        # Random Forest hyperparameters
        self.n_estimators = 500  # Number of trees
        self.max_depth = 20  # Maximum tree depth
        self.min_samples_split = 20  # Minimum samples to split node
        self.min_samples_leaf = 10  # Minimum samples in leaf
        self.max_features = 'sqrt'  # Features considered at each split
        self.bootstrap = True  # Use bootstrap sampling
        self.oob_score = True  # Calculate out-of-bag score
        self.random_state = 42
        self.n_jobs = -1  # Use all CPU cores
        
        # Model object
        self.model = None
        
        # Feature importance analysis
        self.feature_importances_dict = {}
        self.oob_error = None
        self.tree_depths = []
        self.mean_tree_depth = None
        
        # Top features for LaTeX
        self.top_features = []
        
        # Store transformation info
        self.transformation = "square_root"
        
        logger.info(f"Model 9 initialized with {self.n_estimators} trees, max_depth={self.max_depth}")
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features using robust variables from FeatureSelection.txt
        
        Top features by Mutual Information:
        - RESIDENCETYPE (0.2031) - Living setting
        - County (0.1229)
        - BSum (0.1131)
        - BLEVEL, Q26, LOSRI, OLEVEL, PLEVEL
        - Age, Q36, Q30, Q21, FLEVEL, Q27, FSum, PSum
        - Q25, AgeGroup, Q17, Q20, Q44, Q18, Q19, Q16, Q23, Q29, Q28
        
        Returns:
            Tuple of (feature matrix, feature names)
        """
        if not records:
            return np.array([]), []
        
        features_list = []
        feature_names = []
        
        for record in records:
            row_features = []
            
            # 1. Living Setting dummies (RESIDENCETYPE - MI: 0.2031)
            # Drop FH as reference category
            living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
            for setting in living_settings:
                row_features.append(1.0 if record.living_setting == setting else 0.0)
            
            # 2. Age Group dummies (AgeGroup - MI: 0.0530)
            # Drop Age3_20 as reference
            row_features.append(1.0 if record.age_group == 'Age21_30' else 0.0)
            row_features.append(1.0 if record.age_group == 'Age31Plus' else 0.0)
            
            # 3. Continuous age (Age - MI: 0.0681)
            row_features.append(float(record.age))
            
            # 4. Summary Scores
            # BSum (MI: 0.1131)
            bsum = float(getattr(record, 'bsum', 0))
            row_features.append(bsum)
            
            # FSum (MI: 0.0588)
            fsum = float(getattr(record, 'fsum', 0))
            row_features.append(fsum)
            
            # PSum (MI: 0.0568)
            psum = float(getattr(record, 'psum', 0))
            row_features.append(psum)
            
            # 5. Individual QSI Questions (top ones by MI)
            # Q26 (MI: 0.0887), Q36 (0.0660), Q30 (0.0652), Q21 (0.0623)
            # Q27 (0.0613), Q25 (0.0546), Q17 (0.0522), Q20 (0.0521)
            # Q44 (0.0495), Q18 (0.0481), Q19 (0.0481), Q16 (0.0471)
            # Q23 (0.0462), Q29 (0.0458), Q28 (0.0428)
            selected_qsi = [16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29, 30, 36, 44]
            for q_num in selected_qsi:
                value = getattr(record, f'q{q_num}', 0)
                row_features.append(float(value))
            
            features_list.append(row_features)
        
        # Build feature names (do once)
        if not feature_names:
            # Living settings (5)
            for setting in ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']:
                feature_names.append(f'Living_{setting}')
            
            # Age groups (2)
            feature_names.extend(['Age21_30', 'Age31Plus'])
            
            # Continuous age (1)
            feature_names.append('Age')
            
            # Summary scores (3)
            feature_names.extend(['BSum', 'FSum', 'PSum'])
            
            # QSI questions (15)
            for q_num in [16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29, 30, 36, 44]:
                feature_names.append(f'Q{q_num}')
            
            self.feature_names = feature_names
        
        X = np.array(features_list, dtype=np.float64)
        self.num_parameters = self.n_estimators * 10  # Rough estimate for RF
        
        logger.info(f"Prepared {X.shape[1]} robust features for {X.shape[0]} records")
        logger.info(f"Feature count: {len(self.feature_names)}")
        
        return X, self.feature_names
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Random Forest model
        
        Args:
            X: Feature matrix
            y: Target values (sqrt-transformed costs)
        """
        logger.info("Fitting Random Forest model...")
        logger.info(f"  Trees: {self.n_estimators}, Max Depth: {self.max_depth}")
        logger.info(f"  Min samples split: {self.min_samples_split}, Min samples leaf: {self.min_samples_leaf}")
        
        # Initialize Random Forest
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # Fit model
        self.model.fit(X, y)
        
        # Extract feature importance
        for i, feature in enumerate(self.feature_names):
            self.feature_importances_dict[feature] = self.model.feature_importances_[i]
        
        # Sort by importance
        sorted_importance = sorted(self.feature_importances_dict.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        # Store top 5 features for LaTeX
        self.top_features = sorted_importance[:5]
        
        # Get OOB score if available
        if self.oob_score and hasattr(self.model, 'oob_score_'):
            self.oob_error = 1 - self.model.oob_score_
            logger.info(f"  OOB R^2: {self.model.oob_score_:.4f}")
            logger.info(f"  OOB Error: {self.oob_error:.4f}")
        
        # Analyze tree structure
        self.tree_depths = [tree.tree_.max_depth for tree in self.model.estimators_]
        self.mean_tree_depth = np.mean(self.tree_depths)
        
        logger.info(f"  Mean tree depth: {self.mean_tree_depth:.1f}")
        logger.info(f"  Tree depth range: [{min(self.tree_depths)}, {max(self.tree_depths)}]")
        
        # Log top features
        logger.info("Top 10 features by importance:")
        for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
            logger.info(f"    {i}. {feature}: {importance:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted Random Forest
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (sqrt scale - will be squared by base class)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    

    def generate_model_specific_commands(self) -> None:
        """Generate Model 9-specific LaTeX commands"""
        
        newcommands_path = self.output_dir / f'model_{self.model_id}_newcommands.tex'
        renewcommands_path = self.output_dir / f'model_{self.model_id}_renewcommands.tex'
        
        # Append to newcommands file (placeholders)
        try:
            with open(newcommands_path, 'a') as f:
                f.write("\n% Model 9 Random Forest Specific Commands\n")
                f.write("\\newcommand{\\ModelNineNumFeatures}{\\placeholder}\n")  # ADD THIS
                f.write("\\newcommand{\\ModelNineNTrees}{\\placeholder}\n")
                f.write("\\newcommand{\\ModelNineMaxDepth}{\\placeholder}\n")
                f.write("\\newcommand{\\ModelNineOOBError}{\\placeholder}\n")
                f.write("\\newcommand{\\ModelNineOOBRSquared}{\\placeholder}\n")
                f.write("\\newcommand{\\ModelNineAvgTreeDepth}{\\placeholder}\n")
                f.write("\\newcommand{\\ModelNineMinSamplesSplit}{\\placeholder}\n")
                f.write("\\newcommand{\\ModelNineMinSamplesLeaf}{\\placeholder}\n")
                f.write("\\newcommand{\\ModelNineMaxFeatures}{\\placeholder}\n")
                f.write("\\newcommand{\\ModelNineTrainingTime}{\\placeholder}\n")
                
                # Top 5 feature commands
                f.write("\n% Top 5 Features by Importance\n")
                feature_words = ['One', 'Two', 'Three', 'Four', 'Five']
                for word in feature_words:
                    f.write(f"\\newcommand{{\\ModelNineTopFeature{word}}}{{\\placeholder}}\n")
                    f.write(f"\\newcommand{{\\ModelNineTopFeature{word}Importance}}{{\\placeholder}}\n")
            
            logger.info(f"Appended Model 9 specific commands to {newcommands_path}")
            
        except Exception as e:
            logger.error(f"Error appending to newcommands file: {e}")
        
        # Append to renewcommands file (actual values)
        try:
            with open(renewcommands_path, 'a') as f:
                f.write("\n% Model 9 Random Forest Specific Values\n")
                f.write(f"\\renewcommand{{\\ModelNineNumFeatures}}{{{len(self.feature_names)}}}\n")  # ADD THIS
                f.write(f"\\renewcommand{{\\ModelNineNTrees}}{{{self.n_estimators}}}\n")
                f.write(f"\\renewcommand{{\\ModelNineMaxDepth}}{{{self.max_depth}}}\n")
                f.write(f"\\renewcommand{{\\ModelNineOOBError}}{{{self.oob_error if self.oob_error else 0:.4f}}}\n")
                
                # Calculate OOB R^2 from OOB error
                oob_r2 = (1 - self.oob_error) if self.oob_error else 0
                f.write(f"\\renewcommand{{\\ModelNineOOBRSquared}}{{{oob_r2:.4f}}}\n")
                
                f.write(f"\\renewcommand{{\\ModelNineAvgTreeDepth}}{{{self.mean_tree_depth if self.mean_tree_depth else 0:.1f}}}\n")
                f.write(f"\\renewcommand{{\\ModelNineMinSamplesSplit}}{{{self.min_samples_split}}}\n")
                f.write(f"\\renewcommand{{\\ModelNineMinSamplesLeaf}}{{{self.min_samples_leaf}}}\n")
                f.write("\\renewcommand{\\ModelNineMaxFeatures}{sqrt}\n")
                
                # Training time from metrics
                training_time = self.metrics.get('training_time', 0)
                f.write(f"\\renewcommand{{\\ModelNineTrainingTime}}{{{training_time:.1f}}}\n")
                
                # Top 5 features with values
                f.write("\n% Top 5 Features by Importance\n")
                feature_words = ['One', 'Two', 'Three', 'Four', 'Five']
                for i, word in enumerate(feature_words):
                    if i < len(self.top_features):
                        feature, importance = self.top_features[i]
                        clean_feature = feature.replace('_', ' ')
                        f.write(f"\\renewcommand{{\\ModelNineTopFeature{word}}}{{{clean_feature}}}\n")
                        f.write(f"\\renewcommand{{\\ModelNineTopFeature{word}Importance}}{{{importance:.4f}}}\n")
                    else:
                        f.write(f"\\renewcommand{{\\ModelNineTopFeature{word}}}{{N/A}}\n")
                        f.write(f"\\renewcommand{{\\ModelNineTopFeature{word}Importance}}{{0.0000}}\n")
            
            logger.info(f"Appended Model 9 specific values to {renewcommands_path}")
            
        except Exception as e:
            logger.error(f"Error appending to renewcommands file: {e}")
        
    def save_results(self) -> None:
        """Save Model 9-specific results"""
        # Save base results (this generates the base LaTeX commands)
        super().save_results()
        
        # Generate Model 9-specific LaTeX commands
        self.generate_model_specific_commands()
        
        # Save feature importances
        importance_df = pd.DataFrame(list(self.feature_importances_dict.items()),
                                    columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_file = self.output_dir / 'feature_importances.csv'
        importance_df.to_csv(importance_file, index=False)
        logger.info(f"Feature importances saved to {importance_file}")
        
        # Save tree depth analysis
        tree_stats = {
            'mean_depth': float(self.mean_tree_depth) if self.mean_tree_depth else 0,
            'min_depth': int(min(self.tree_depths)) if self.tree_depths else 0,
            'max_depth': int(max(self.tree_depths)) if self.tree_depths else 0,
            'std_depth': float(np.std(self.tree_depths)) if self.tree_depths else 0,
            'n_trees': self.n_estimators
        }
        tree_file = self.output_dir / 'tree_statistics.json'
        with open(tree_file, 'w') as f:
            json.dump(tree_stats, f, indent=2)
        logger.info(f"Tree statistics saved to {tree_file}")
                

    def plot_model_specific_diagnostics(self) -> None:
        """Generate Random Forest-specific diagnostic plots"""
        if self.model is None:
            logger.warning("Model not fitted - cannot create RF diagnostics")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Feature Importance (Top 20)
        ax1 = plt.subplot(2, 3, 1)
        importance_df = pd.DataFrame(list(self.feature_importances_dict.items()),
                                    columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
        
        y_pos = np.arange(len(importance_df))
        ax1.barh(y_pos, importance_df['Importance'].values, color='steelblue')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(importance_df['Feature'].values, fontsize=8)
        ax1.set_xlabel('Feature Importance')
        ax1.set_title('Top 20 Features by Importance')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Cumulative Feature Importance
        ax2 = plt.subplot(2, 3, 2)
        sorted_importance = sorted(self.feature_importances_dict.values(), reverse=True)
        cumsum_importance = np.cumsum(sorted_importance)
        ax2.plot(range(1, len(cumsum_importance) + 1), cumsum_importance, 'b-', linewidth=2)
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80% threshold', alpha=0.5)
        ax2.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold', alpha=0.5)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance')
        ax2.set_title('Cumulative Feature Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Tree Depth Distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(self.tree_depths, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax3.axvline(x=self.mean_tree_depth, color='r', linestyle='--', 
                   label=f'Mean: {self.mean_tree_depth:.1f}', linewidth=2)
        ax3.set_xlabel('Tree Depth')
        ax3.set_ylabel('Number of Trees')
        ax3.set_title('Distribution of Tree Depths')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. OOB Error Analysis (if available)
        ax4 = plt.subplot(2, 3, 4)
        if self.oob_score and hasattr(self.model, 'oob_score_'):
            # Estimate OOB error vs number of trees
            n_estimators_range = range(10, self.n_estimators + 1, 10)
            oob_errors = []
            
            for n_est in n_estimators_range:
                # Use subset of trees
                temp_predictions = np.mean([tree.predict(self.X_train) 
                                          for tree in self.model.estimators_[:n_est]], axis=0)
                oob_r2 = r2_score(self.y_train, temp_predictions)
                oob_errors.append(1 - oob_r2)
            
            ax4.plot(n_estimators_range, oob_errors, 'b-', linewidth=2)
            ax4.set_xlabel('Number of Trees')
            ax4.set_ylabel('OOB Error')
            ax4.set_title('OOB Error vs Forest Size')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'OOB Score Not Available', 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('OOB Error Analysis')
        
        # 5. Prediction Variance (from tree variance)
        ax5 = plt.subplot(2, 3, 5)
        if self.test_predictions is not None and len(self.test_predictions) > 0:
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(self.X_test) 
                                        for tree in self.model.estimators_])
            tree_std = np.std(tree_predictions, axis=0)
            
            # Plot prediction std vs mean prediction
            mean_pred = np.mean(tree_predictions, axis=0)
            ax5.scatter(mean_pred, tree_std, alpha=0.5, s=10)
            ax5.set_xlabel('Mean Prediction (sqrt scale)')
            ax5.set_ylabel('Std Dev Across Trees')
            ax5.set_title('Prediction Uncertainty')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Test Predictions Not Available', 
                    ha='center', va='center', fontsize=12)
            ax5.set_title('Prediction Uncertainty')
        
        # 6. Bias by Feature Value (for top feature)
        ax6 = plt.subplot(2, 3, 6)
        if (self.test_predictions is not None and len(self.test_predictions) > 0 
            and len(self.top_features) > 0):
            top_feature_name = self.top_features[0][0]
            top_feature_idx = self.feature_names.index(top_feature_name)
            top_feature_values = self.X_test[:, top_feature_idx]
            
            # Calculate bias (residuals)
            bias = self.test_predictions - self.y_test
            
            # Bin by feature value and plot mean bias
            n_bins = 10
            bins = np.linspace(top_feature_values.min(), top_feature_values.max(), n_bins + 1)
            bin_indices = np.digitize(top_feature_values, bins)
            
            bin_means = []
            bin_centers = []
            for i in range(1, n_bins + 1):
                mask = bin_indices == i
                if mask.sum() > 0:
                    bin_means.append(np.mean(bias[mask]))
                    bin_centers.append((bins[i-1] + bins[i]) / 2)
            
            ax6.plot(bin_centers, bin_means, 'bo-', linewidth=2, markersize=6)
            ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax6.set_xlabel(f'{top_feature_name} Value')
            ax6.set_ylabel('Mean Bias (Predicted - Actual)')
            ax6.set_title(f'Bias Pattern by {top_feature_name}')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Feature Analysis Not Available', 
                    ha='center', va='center', fontsize=12)
            ax6.set_title('Bias by Feature Value')
        
        plt.suptitle('Model 9: Random Forest Diagnostics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'model9_rf_diagnostics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Random Forest diagnostic plots saved to {output_file}")


def main():
    """Main execution function"""
    # Initialize model
    model = Model9RandomForest()
    
    # Run complete pipeline
    results = model.run_complete_pipeline(
        fiscal_year_start=2023,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL 9: RANDOM FOREST - RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nRandom Forest Configuration:")
    print(f"  ? Number of Trees: {model.n_estimators}")
    print(f"  ? Max Depth: {model.max_depth}")
    print(f"  ? Min Samples Split: {model.min_samples_split}")
    print(f"  ? Min Samples Leaf: {model.min_samples_leaf}")
    print(f"  ? Max Features: {model.max_features}")
    print(f"  ? OOB Score: {model.oob_score}")
    
    print(f"\nData Summary:")
    print(f"  ? Total Records: {len(model.all_records)}")
    print(f"  ? Training Records: {model.metrics.get('training_samples', 0)}")
    print(f"  ? Test Records: {model.metrics.get('test_samples', 0)}")
    print(f"  ? Features Used: {len(model.feature_names)}")
    print(f"  ? Outliers Removed: 0 (Random Forest handles outliers naturally)")
    
    print(f"\nPerformance Metrics:")
    print(f"  ? Training R^2: {model.metrics.get('r2_train', 0):.4f}")
    print(f"  ? Test R^2: {model.metrics.get('r2_test', 0):.4f}")
    print(f"  ? RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    print(f"  ? MAE: ${model.metrics.get('mae_test', 0):,.2f}")
    print(f"  ? MAPE: {model.metrics.get('mape_test', 0):.2f}%")
    print(f"  ? CV R^2 (10-fold): {model.metrics.get('cv_mean', 0):.4f} +- {model.metrics.get('cv_std', 0):.4f}")
    
    if model.oob_error is not None:
        print(f"\nRandom Forest Specific:")
        print(f"  ? OOB R^2: {1 - model.oob_error:.4f}")
        print(f"  ? OOB Error: {model.oob_error:.4f}")
        print(f"  ? Mean Tree Depth: {model.mean_tree_depth:.1f}")
    
    print(f"\nAccuracy Bands:")
    print(f"  ? Within $1,000: {model.metrics.get('within_1k', 0):.1f}%")
    print(f"  ? Within $2,000: {model.metrics.get('within_2k', 0):.1f}%")
    print(f"  ? Within $5,000: {model.metrics.get('within_5k', 0):.1f}%")
    print(f"  ? Within $10,000: {model.metrics.get('within_10k', 0):.1f}%")
    
    print(f"\nTop 5 Features by Importance:")
    for i, (feature, importance) in enumerate(model.top_features, 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    print("\nKey Advantages:")
    print("  ? No outlier removal (100% data retention)")
    print("  ? Automatic feature interaction detection")
    print("  ? Non-linear pattern capture")
    print("  ? Feature importance provides interpretability")
    print("  ? Robust to outliers and missing values")
    print("  ? Natural handling of heteroscedasticity")
    
    print("\n" + "="*80)
    print("Model 9 implementation complete!")
    print("="*80)
    
    return model


if __name__ == "__main__":
    model = main()
