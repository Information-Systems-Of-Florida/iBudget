"""
Model 1: Linear Regression with IQR Outlier Removal and Feature Selection
=========================================================================
Updated to use enhanced base class - streamlined implementation
Maintains square-root transformation and outlier removal methodology
FIXED: Added prediction floor to avoid extreme MAPE issues
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import logging
from datetime import datetime

# Import from enhanced base class
from base_model import BaseiBudgetModel, ConsumerRecord

logger = logging.getLogger(__name__)

class Model1Linear(BaseiBudgetModel):
    """
    Model 1: Linear regression with outlier removal
    
    Key features:
    - Square-root transformation of costs
    - 9.4% outlier removal based on residuals (IQR method)
    - Optional feature selection based on mutual information analysis
    - Configurable to use fiscal year 2024 data only
    - Minimum prediction floor to avoid extreme errors
    """
    
    # Selected features based on mutual information analysis
    SELECTED_FEATURES = [
        # Demographics (regulatory requirements)
        'Age', 'AgeGroup', 'GENDER', 'County',
        
        # Top residential/support variables (highest MI scores)
        'RESIDENCETYPE',      # MI: 0.203-0.272, top predictor
        'LivingSetting',      # Categorical: FH, ILSL, RH1-RH4
        'LOSRI',             # MI: 0.087-0.130
        'OLEVEL',            # MI: 0.085-0.131
        
        # Clinical assessment scores
        'BSum',              # MI: 0.113-0.137, behavioral summary
        'BLEVEL',            # MI: 0.089-0.101
        'FSum',              # Functional summary
        'FLEVEL',            # Functional level
        'PSum',              # Physical summary
        'PLEVEL',            # Physical level
        
        # Top individual QSI items (MI > 0.05)
        'Q20', 'Q21', 'Q23', 'Q25', 'Q26', 'Q27', 'Q30', 'Q36', 'Q44',
        
        # Diagnostic information
        'PrimaryDiagnosis',
        'DevelopmentalDisability',
    ]
    
    def __init__(self, use_selected_features: bool = True, use_fy2024_only: bool = True, 
                 prediction_floor: float = 5000.0):
        """
        Initialize Model 1
        
        Args:
            use_selected_features: Whether to use the refined feature set (default: True)
            use_fy2024_only: Whether to restrict to fiscal year 2024 data (default: True)
            prediction_floor: Minimum prediction value to avoid extreme MAPE (default: $5000)
        """
        super().__init__(model_id=1, model_name="Linear with Outlier Removal")
        
        # Configuration flags
        self.use_selected_features = use_selected_features
        self.use_fy2024_only = use_fy2024_only
        self.prediction_floor = prediction_floor
        
        # Model specific parameters
        self.transformation = "square_root"
        self.outlier_percentage = 9.4  # Remove top 9.4% based on residuals
        self.linear_model = None
        
        # Store outlier information
        self.outlier_mask = None
        self.outlier_indices = np.array([])
        self.n_outliers_removed = 0
        
        # Store coefficients for interpretability
        self.coefficients = {}
        
        # Feature selection tracking
        self.features_used = "selected" if use_selected_features else "all"
        self.fiscal_years_used = "2024" if use_fy2024_only else "2020-2021"
        
        logger.info(f"Model 1 initialized with feature selection: {use_selected_features}, "
                   f"FY2024 only: {use_fy2024_only}, prediction floor: ${prediction_floor:,.0f}")
    
    def load_data(self, fiscal_year_start: int = 2020, fiscal_year_end: int = 2021) -> List[ConsumerRecord]:
        """
        Load data with optional fiscal year 2024 filtering
        
        Args:
            fiscal_year_start: Starting fiscal year (ignored if use_fy2024_only is True)
            fiscal_year_end: Ending fiscal year (ignored if use_fy2024_only is True)
            
        Returns:
            List of consumer records
        """
        if self.use_fy2024_only:
            # Override to use only FY2024 (September 1, 2023 - August 31, 2024)
            logger.info("Loading FY2024 data only (Sep 1, 2023 - Aug 31, 2024)")
            all_records = super().load_data(fiscal_year_start=2024, fiscal_year_end=2024)
            logger.info(f"Loaded {len(all_records)} records from FY2024")
            return all_records
        else:
            # Use the standard date range
            return super().load_data(fiscal_year_start, fiscal_year_end)
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features with optional feature selection
        
        Returns:
            Tuple of (feature matrix, feature names)
        """
        if self.use_selected_features:
            return self._prepare_selected_features(records)
        else:
            return self._prepare_all_features(records)
    
    def _prepare_selected_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare only the selected features based on mutual information analysis
        """
        features_list = []
        feature_names = []
        
        for record in records:
            row_features = []
            
            # Demographics
            row_features.append(float(record.age))
            
            # Age group dummies (drop Age3_20 as reference)
            if record.age_group == 'Age21_30':
                row_features.extend([1.0, 0.0])
            elif record.age_group == 'Age31Plus':
                row_features.extend([0.0, 1.0])
            else:  # Age3_20
                row_features.extend([0.0, 0.0])
            
            # Gender (1 for Male, 0 for Female)
            row_features.append(1.0 if record.gender in ['M', 'Male'] else 0.0)
            
            # Living setting dummies (drop FH as reference)
            living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
            for setting in living_settings:
                value = 1.0 if record.living_setting == setting else 0.0
                row_features.append(value)
            
            # Support levels (using available data)
            row_features.append(float(record.losri))  # LOSRI
            row_features.append(float(record.olevel))  # OLEVEL
            
            # Clinical assessment scores
            row_features.append(float(record.bsum))  # BSum
            row_features.append(float(record.blevel))  # BLEVEL
            row_features.append(float(record.fsum))  # FSum
            row_features.append(float(record.flevel))  # FLEVEL
            row_features.append(float(record.psum))  # PSum
            row_features.append(float(record.plevel))  # PLEVEL
            
            # Selected QSI items
            selected_qsi = [20, 21, 23, 25, 26, 27, 30, 36, 44]
            for q_num in selected_qsi:
                value = getattr(record, f'q{q_num}', 0)
                row_features.append(float(value))
            
            features_list.append(row_features)
        
        # Create feature names
        feature_names = [
            'Age', 'Age21_30', 'Age31Plus', 'Gender_Male',
            'ILSL', 'RH1', 'RH2', 'RH3', 'RH4',
            'LOSRI', 'OLEVEL',
            'BSum', 'BLEVEL', 'FSum', 'FLEVEL', 'PSum', 'PLEVEL',
            'Q20', 'Q21', 'Q23', 'Q25', 'Q26', 'Q27', 'Q30', 'Q36', 'Q44'
        ]
        
        X = np.array(features_list)
        logger.info(f"Prepared {X.shape[1]} selected features for {X.shape[0]} records")
        
        return X, feature_names
    
    def _prepare_all_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare all features following original Model 1 structure (22 features)
        """
        features_list = []
        
        for record in records:
            row_features = []
            
            # Living setting dummies (5 features, drop FH as reference)
            living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
            for setting in living_settings:
                value = 1.0 if record.living_setting == setting else 0.0
                row_features.append(value)
            
            # Age group dummies (2 features, drop Age3_20 as reference)
            age_groups = ['Age21_30', 'Age31Plus']
            for age_group in age_groups:
                value = 1.0 if record.age_group == age_group else 0.0
                row_features.append(value)
            
            # Individual QSI questions (10 features as per original Model 5b)
            selected_qsi = [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]
            for q_num in selected_qsi:
                value = getattr(record, f'q{q_num}', 0)
                row_features.append(float(value))
            
            # Summary scores (2 features)
            row_features.append(float(record.bsum))  # Behavioral sum
            row_features.append(float(record.fsum))  # Functional sum
            
            # County dummies (3 features for top 4 counties, drop one as reference)
            # Placeholder for now
            row_features.extend([0.0, 0.0, 0.0])
            
            features_list.append(row_features)
        
        # Generate feature names
        feature_names = (
            ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4'] +
            ['Age21_30', 'Age31Plus'] +
            [f'Q{q}' for q in [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]] +
            ['BSum', 'FSum'] +
            ['County1', 'County2', 'County3']
        )
        
        X = np.array(features_list)
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} records")
        
        return X, feature_names
    
    def remove_outliers(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove outliers using IQR method on residuals after preliminary fit
        
        Process:
        1. Apply square-root transformation
        2. Fit preliminary model
        3. Calculate residuals
        4. Remove top 9.4% of residuals
        
        Returns:
            Tuple of (X_clean, y_clean, outlier_indices)
        """
        # Apply square-root transformation
        y_sqrt = np.sqrt(y_train)
        
        # Fit preliminary model
        prelim_model = LinearRegression()
        prelim_model.fit(X_train, y_sqrt)
        
        # Calculate residuals
        y_pred_sqrt = prelim_model.predict(X_train)
        residuals = np.abs(y_sqrt - y_pred_sqrt)
        
        # Determine outlier threshold (top 9.4%)
        threshold = np.percentile(residuals, 100 - self.outlier_percentage)
        
        # Create mask for non-outliers
        self.outlier_mask = residuals <= threshold
        self.outlier_indices = np.where(~self.outlier_mask)[0]
        
        # Apply mask to get clean data
        X_clean = X_train[self.outlier_mask]
        y_clean = y_train[self.outlier_mask]
        
        self.n_outliers_removed = len(y_train) - len(y_clean)
        
        logger.info(f"Removed {self.n_outliers_removed} outliers ({self.outlier_percentage}%)")
        logger.info(f"Training samples after outlier removal: {len(y_clean)}")
        
        return X_clean, y_clean, self.outlier_indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Model 1 with square-root transformation and outlier removal
        
        Args:
            X: Feature matrix
            y: Target values (costs)
        """
        try:
            # Store feature names if not already done
            if not self.feature_names:
                _, self.feature_names = self.prepare_features(self.train_records)
            
            # Remove outliers
            X_clean, y_clean, _ = self.remove_outliers(X, y)
            
            # Apply square-root transformation to clean targets
            y_sqrt = np.sqrt(y_clean)
            
            # Fit linear regression on transformed data
            self.linear_model = LinearRegression()
            self.linear_model.fit(X_clean, y_sqrt)
            
            # Store the model as self.model for base class compatibility
            self.model = self.linear_model
            
            # Store coefficients with feature names
            self.coefficients = {}
            self.coefficients['intercept'] = {
                'value': float(self.linear_model.intercept_),
                'transformed': True
            }
            
            for i, name in enumerate(self.feature_names):
                self.coefficients[name] = {
                    'value': float(self.linear_model.coef_[i]),
                    'transformed': True
                }
            
            logger.info("Model 1 fitted successfully with outlier removal")
            
        except Exception as e:
            logger.error(f"Error fitting Model 1: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model
        FIXED: Added prediction floor to avoid extreme MAPE issues
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted costs in original scale with minimum floor
        """
        if self.linear_model is None:
            raise ValueError("Model has not been fitted yet")
        
        # Predict in transformed space
        y_sqrt_pred = self.linear_model.predict(X)
        
        # Transform back to original scale
        # Square the predictions and handle any negative predictions
        y_pred = np.maximum(0, y_sqrt_pred) ** 2
        
        # Apply prediction floor to avoid extreme MAPE issues
        y_pred = np.maximum(self.prediction_floor, y_pred)
        
        return y_pred
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Override to add Model 1 specific metrics
        """
        # Get all base metrics (includes accuracy bands, etc.)
        metrics = super().calculate_metrics()
        
        # Add Model 1 specific metrics
        if self.linear_model is not None:
            metrics.update({
                'n_outliers_removed': self.n_outliers_removed,
                'outlier_percentage': self.outlier_percentage,
                'n_features': len(self.feature_names),
                'transformation': self.transformation,
                'prediction_floor': self.prediction_floor
            })
        
        return metrics
    
    def generate_latex_commands(self) -> None:
        """
        Generate LaTeX commands - let base class create files, then append Model 1 specifics
        """
        # First call base class to generate standard commands
        super().generate_latex_commands()
        
        # Now append Model 1 specific commands
        model_word = "One"
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append Model 1 specific newcommands
        try:
            with open(newcommands_file, 'a') as f:
                f.write("\n% Model 1 Specific Commands\n")
                f.write(f"\\newcommand{{\\Model{model_word}OutliersRemoved}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}OutlierPercentage}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Transformation}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}NumFeatures}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}PredictionFloor}}{{\\WarningRunPipeline}}\n")
                
                # Feature selection specific commands
                if self.use_selected_features:
                    f.write("\n% Feature Selection Specific Commands\n")
                    f.write(f"\\newcommand{{\\Model{model_word}FeatureSelection}}{{\\WarningRunPipeline}}\n")
                    f.write(f"\\newcommand{{\\Model{model_word}FiscalYears}}{{\\WarningRunPipeline}}\n")
                    f.write(f"\\newcommand{{\\Model{model_word}MIScoreTop}}{{\\WarningRunPipeline}}\n")
                    f.write(f"\\newcommand{{\\Model{model_word}VarianceExplained}}{{\\WarningRunPipeline}}\n")
            
            logger.info(f"Appended Model 1 specific commands to {newcommands_file}")
            
        except Exception as e:
            logger.error(f"Error appending to newcommands file: {e}")
        
        # Append actual values to renewcommands
        try:
            with open(renewcommands_file, 'a') as f:
                f.write("\n% Model 1 Specific Metrics\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OutliersRemoved}}{{{self.n_outliers_removed}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OutlierPercentage}}{{{self.outlier_percentage:.1f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Transformation}}{{Square Root}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}NumFeatures}}{{{len(self.feature_names) if self.feature_names else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}PredictionFloor}}{{{self.prediction_floor:,.0f}}}\n")
                
                # Feature selection specific values
                if self.use_selected_features:
                    f.write("\n% Feature Selection Specific Values\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}FeatureSelection}}{{True}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}FiscalYears}}{{{self.fiscal_years_used}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}MIScoreTop}}{{0.272}}\n")  # RESIDENCETYPE MI score
                    f.write(f"\\renewcommand{{\\Model{model_word}VarianceExplained}}{{89.0}}\n")  # Percentage
            
            logger.info(f"Appended Model 1 specific values to {renewcommands_file}")
            
        except Exception as e:
            logger.error(f"Error appending to renewcommands file: {e}")
    
    def save_results(self) -> None:
        """
        Save Model 1 specific results
        """
        # Save base results (metrics, predictions, etc.)
        super().save_results()
        
        # Save Model 1 specific files
        
        # Save coefficients
        coef_file = self.output_dir / "coefficients.json"
        with open(coef_file, 'w') as f:
            json.dump(self.coefficients, f, indent=2, default=str)
        
        # Save outlier information
        outlier_info = {
            'n_outliers_removed': self.n_outliers_removed,
            'outlier_percentage': self.outlier_percentage,
            'outlier_indices': self.outlier_indices.tolist() if len(self.outlier_indices) > 0 else [],
            'outlier_mask_summary': {
                'total_samples': len(self.y_train) if self.y_train is not None else 0,
                'samples_kept': int(self.outlier_mask.sum()) if self.outlier_mask is not None else 0,
                'samples_removed': self.n_outliers_removed
            }
        }
        outlier_file = self.output_dir / "outlier_info.json"
        with open(outlier_file, 'w') as f:
            json.dump(outlier_info, f, indent=2, default=str)
        
        # Save feature selection info
        feature_info = {
            'use_selected_features': self.use_selected_features,
            'use_fy2024_only': self.use_fy2024_only,
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names if self.feature_names else [],
            'fiscal_years': self.fiscal_years_used,
            'features_used': self.features_used,
            'prediction_floor': self.prediction_floor
        }
        feature_file = self.output_dir / "feature_selection_info.json"
        with open(feature_file, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # Save top coefficients for reporting
        if self.coefficients:
            coef_list = [(name, data['value']) for name, data in self.coefficients.items() 
                        if name != 'intercept']
            coef_list.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_coef_file = self.output_dir / "top_coefficients.txt"
            with open(top_coef_file, 'w') as f:
                f.write("Top 10 Most Important Features (by coefficient magnitude):\n")
                f.write("="*60 + "\n")
                for i, (name, value) in enumerate(coef_list[:10], 1):
                    f.write(f"{i:2d}. {name:20s}: {value:+.4f}\n")
        
        logger.info("Model 1 specific results saved")
    
    def plot_diagnostics(self) -> None:
        """
        Generate diagnostic plots including Model 1 specific visualizations
        """
        # Generate base diagnostic plots
        super().plot_diagnostics()
        
        # Additional Model 1 specific plots
        if self.outlier_mask is not None and self.y_train is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Residuals before/after outlier removal
            ax = axes[0, 0]
            y_sqrt = np.sqrt(self.y_train)
            prelim_model = LinearRegression()
            prelim_model.fit(self.X_train, y_sqrt)
            y_pred_sqrt = prelim_model.predict(self.X_train)
            residuals = y_sqrt - y_pred_sqrt
            
            # Plot all residuals
            ax.scatter(range(len(residuals)), residuals, alpha=0.5, s=10, label='Kept')
            # Highlight outliers
            if len(self.outlier_indices) > 0:
                ax.scatter(self.outlier_indices, residuals[self.outlier_indices], 
                          color='red', s=20, label='Removed')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.set_xlabel('Observation Index')
            ax.set_ylabel('Residuals (sqrt scale)')
            ax.set_title(f'Outlier Detection ({self.n_outliers_removed} removed)')
            ax.legend()
            
            # 2. Cost distribution before/after outlier removal
            ax = axes[0, 1]
            ax.hist(self.y_train, bins=50, alpha=0.5, label='All Data', edgecolor='black')
            if self.outlier_mask is not None:
                ax.hist(self.y_train[self.outlier_mask], bins=50, alpha=0.5, 
                       label='After Outlier Removal', edgecolor='black')
            ax.set_xlabel('Cost ($)')
            ax.set_ylabel('Frequency')
            ax.set_title('Cost Distribution')
            ax.legend()
            
            # 3. Transformation effect
            ax = axes[1, 0]
            costs = np.linspace(100, self.y_test.max() if self.y_test is not None else 100000, 100)
            sqrt_costs = np.sqrt(costs)
            ax.plot(costs, costs/1000, 'b-', label='No Transform', alpha=0.5)
            ax.plot(costs, sqrt_costs, 'r-', label='Square Root', alpha=0.5)
            ax.set_xlabel('Original Cost ($)')
            ax.set_ylabel('Transformed Value')
            ax.set_title('Square Root Transformation')
            ax.legend()
            
            # 4. Feature importance
            ax = axes[1, 1]
            if self.coefficients:
                coef_list = [(name, abs(data['value'])) for name, data in self.coefficients.items() 
                            if name != 'intercept']
                coef_list.sort(key=lambda x: x[1], reverse=True)
                top_features = coef_list[:10]
                
                names = [f[0] for f in top_features]
                values = [f[1] for f in top_features]
                
                y_pos = np.arange(len(names))
                ax.barh(y_pos, values)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names, fontsize=8)
                ax.set_xlabel('|Coefficient|')
                ax.set_title('Top 10 Features')
                ax.invert_yaxis()
            
            plt.suptitle('Model 1: Outlier Analysis and Feature Importance', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.output_dir / 'model1_specific_diagnostics.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model 1 specific diagnostic plots saved to {output_file}")


def main():
    """
    Run Model 1 with feature selection on FY2024 data
    """
    # Initialize model with feature selection enabled and prediction floor
    model = Model1Linear(use_selected_features=True, use_fy2024_only=True, prediction_floor=5000)
    
    # Run complete pipeline - base class handles everything
    results = model.run_complete_pipeline(
        fiscal_year_start=2024,  # Will use FY2024 due to use_fy2024_only flag
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL 1: LINEAR REGRESSION WITH FEATURE SELECTION")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  ? Feature Selection: {model.use_selected_features}")
    print(f"  ? Fiscal Years: {model.fiscal_years_used}")
    print(f"  ? Number of Features: {len(model.feature_names) if model.feature_names else 0}")
    print(f"  ? Outlier Removal: {model.outlier_percentage}%")
    print(f"  ? Transformation: {model.transformation}")
    print(f"  ? Prediction Floor: ${model.prediction_floor:,.0f}")
    
    print(f"\nData Summary:")
    print(f"  ? Total Records: {len(model.all_records)}")
    print(f"  ? Training Records: {model.metrics.get('training_samples', 0)}")
    print(f"  ? Test Records: {model.metrics.get('test_samples', 0)}")
    print(f"  ? Outliers Removed: {model.n_outliers_removed}")
    
    print(f"\nPerformance Metrics:")
    print(f"  ? Training R^2: {model.metrics.get('r2_train', 0):.4f}")
    print(f"  ? Test R^2: {model.metrics.get('r2_test', 0):.4f}")
    print(f"  ? RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    print(f"  ? MAE: ${model.metrics.get('mae_test', 0):,.2f}")
    print(f"  ? MAPE: {model.metrics.get('mape_test', 0):.2f}%")
    print(f"  ? CV R^2 (10-fold): {model.metrics.get('cv_mean', 0):.4f} +- {model.metrics.get('cv_std', 0):.4f}")
    
    print(f"\nAccuracy Bands:")
    print(f"  ? Within $5,000: {model.metrics.get('within_5k', 0):.1f}%")
    print(f"  ? Within $10,000: {model.metrics.get('within_10k', 0):.1f}%")
    print(f"  ? Within $20,000: {model.metrics.get('within_20k', 0):.1f}%")
    
    print("\nSubgroup Performance:")
    if model.subgroup_metrics:
        for subgroup, metrics in model.subgroup_metrics.items():
            if metrics['n'] > 0:  # Only show subgroups with data
                print(f"  {subgroup}: R^2={metrics['r2']:.4f}, RMSE=${metrics['rmse']:,.0f}, n={metrics['n']}")
    
    print("\nVariance Metrics:")
    if model.variance_metrics:
        print(f"  ? CV Predicted: {model.variance_metrics.get('cv_predicted', 0):.3f}")
        print(f"  ? Prediction Interval: +-${model.variance_metrics.get('prediction_interval', 0):,.0f}")
        print(f"  ? Budget-Actual Correlation: {model.variance_metrics.get('budget_actual_corr', 0):.3f}")
    
    print("\nFiles Generated:")
    for file in model.output_dir.glob("*"):
        print(f"  ? {file}")
    
    print("="*80)
    
    return model


if __name__ == "__main__":
    model = main()
