"""
Model 1: Model 5b Re-Evaluation with 2024 Data
===============================================
Faithful replication of Tao & Niu (2015) Model 5b specification
Uses EXACT 21 features from original Model 5b for direct comparison
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
    Model 1: Model 5b Re-Evaluation (Tao & Niu 2015) with 2024 Data
    
    Exact Model 5b Specification (21 features):
    - 5 Living Setting dummies (FH as reference)
    - 2 Age Group dummies (Age 3-20 as reference)
    - 1 BSum (behavioral sum)
    - 3 Interaction Terms (FHFSum, SLFSum, SLBSum)
    - 10 QSI Questions (Q16, Q18, Q20, Q21, Q23, Q28, Q33, Q34, Q36, Q43)
    
    Methodology:
    - Square-root transformation (configurable)
    - 9.4% outlier removal via residual analysis
    - OLS regression
    """
    
    # Model 5b exact feature specification
    MODEL_5B_QSI_QUESTIONS = [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]
    MODEL_5B_N_FEATURES = 21
    MODEL_5B_OUTLIER_PCT = 9.4
    
    def __init__(self, use_sqrt_transform: bool = True):
        """
        Initialize Model 1 - Model 5b Re-evaluation
        
        Args:
            use_sqrt_transform: Use square-root transformation (default: True, matches Model 5b)
        """
        super().__init__(
            model_id=1,
            model_name="Model 5b Re-Evaluation (2024)"
        )
        
        # Model configuration
        self.use_sqrt_transform = use_sqrt_transform
        self.transformation = "square_root" if use_sqrt_transform else "none"
        self.outlier_percentage = self.MODEL_5B_OUTLIER_PCT
        
        # Model object
        self.linear_model = None
        
        # Outlier tracking
        self.outlier_mask = None
        self.outlier_indices = np.array([])
        self.n_outliers_removed = 0
        
        # Coefficients for interpretation
        self.coefficients = {}
        
        # Model 5b comparison metrics
        self.model_5b_comparison = {
            'original_r2_2015': 0.7998,
            'original_adj_r2_2015': 0.7996,
            'original_rmse_2015': 30.82,
            'original_sbc_2015': 159394.3,
            'original_n_train_2015': 23215,
            'original_outliers_removed_2015': 2410,
            'original_outlier_pct_2015': 9.4
        }
        
        logger.info(f"Model 1 initialized: Model 5b re-evaluation")
        logger.info(f"  - Square-root transform: {use_sqrt_transform}")
        logger.info(f"  - Features: {self.MODEL_5B_N_FEATURES} (exact Model 5b specification)")
        logger.info(f"  - Outlier removal: {self.outlier_percentage}%")
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features matching Model 5b EXACTLY (21 features)
        
        Model 5b Feature Structure:
        1-5:   Living Settings (5 dummy variables, FH as reference)
        6-7:   Age Groups (2 dummy variables, Age 3-20 as reference)
        8:     BSum (behavioral sum)
        9-11:  Interaction Terms (FHFSum, SLFSum, SLBSum)
        12-21: QSI Questions (Q16, Q18, Q20, Q21, Q23, Q28, Q33, Q34, Q36, Q43)
        
        Returns:
            Tuple of (feature matrix, feature names)
        """
        features_list = []
        
        for record in records:
            row_features = []
            
            # 1-5. Living Settings (5 dummy variables, FH as reference)
            # Model 5b uses: LiveILSL, LiveRH1, LiveRH2, LiveRH3, LiveRH4
            living = record.living_setting
            row_features.extend([
                1.0 if living == 'ILSL' else 0.0,          # LiveILSL
                1.0 if living in ['RH1'] else 0.0,         # LiveRH1
                1.0 if living in ['RH2'] else 0.0,         # LiveRH2
                1.0 if living in ['RH3'] else 0.0,         # LiveRH3
                1.0 if living in ['RH4'] else 0.0,         # LiveRH4
            ])
            
            # 6-7. Age Groups (2 dummy variables, Age 3-20 as reference)
            age_group = record.age_group
            row_features.extend([
                1.0 if age_group == 'Age21_30' else 0.0,   # Age21-30
                1.0 if age_group == 'Age31Plus' else 0.0,  # Age31+
            ])
            
            # 8. BSum (behavioral sum)
            bsum = float(record.bsum or 0)
            fsum = float(record.fsum or 0)
            row_features.append(bsum)
            
            # 9-11. CRITICAL: Interaction Terms (Model 5b's key innovation)
            # These capture differential effects of functional/behavioral needs by setting
            is_fh = 1.0 if living == 'FH' else 0.0
            is_sl = 1.0 if living == 'ILSL' else 0.0
            
            fhf_sum = is_fh * fsum  # FHFSum: Family Home × Functional Sum
            slf_sum = is_sl * fsum  # SLFSum: Supported Living × Functional Sum
            slb_sum = is_sl * bsum  # SLBSum: Supported Living × Behavioral Sum
            
            row_features.extend([fhf_sum, slf_sum, slb_sum])
            
            # 12-21. QSI Questions (EXACT 10 from Model 5b Table 7)
            for q_num in self.MODEL_5B_QSI_QUESTIONS:
                value = getattr(record, f'q{q_num}', 0) or 0
                row_features.append(float(value))
            
            features_list.append(row_features)
        
        # Feature names matching Model 5b Table 7
        feature_names = [
            # Living Settings
            'LiveILSL', 'LiveRH1', 'LiveRH2', 'LiveRH3', 'LiveRH4',
            # Age Groups
            'Age21_30', 'Age31Plus',
            # Behavioral Sum
            'BSum',
            # Interaction Terms
            'FHFSum', 'SLFSum', 'SLBSum',
            # QSI Questions
            'Q16', 'Q18', 'Q20', 'Q21', 'Q23', 'Q28', 'Q33', 'Q34', 'Q36', 'Q43'
        ]
        
        X = np.array(features_list)
        
        # Verify correct number of features
        assert X.shape[1] == self.MODEL_5B_N_FEATURES, \
            f"Expected {self.MODEL_5B_N_FEATURES} features, got {X.shape[1]}"
        
        logger.info(f"Prepared {X.shape[1]} features (Model 5b specification) for {X.shape[0]} records")
        
        return X, feature_names
    
    def remove_outliers(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove outliers using Model 5b methodology
        
        Process (from Tao & Niu 2015):
        1. Apply square-root transformation (if enabled)
        2. Fit preliminary OLS model
        3. Calculate absolute residuals
        4. Remove top 9.4% of residuals
        5. Return clean data
        
        Returns:
            Tuple of (X_clean, y_clean, outlier_indices)
        """
        # Apply transformation
        if self.use_sqrt_transform:
            y_transformed = np.sqrt(y_train)
        else:
            y_transformed = y_train.copy()
        
        # Fit preliminary model
        prelim_model = LinearRegression()
        prelim_model.fit(X_train, y_transformed)
        
        # Calculate residuals
        y_pred_transformed = prelim_model.predict(X_train)
        residuals = np.abs(y_transformed - y_pred_transformed)
        
        # Determine outlier threshold (top 9.4%)
        threshold = np.percentile(residuals, 100 - self.outlier_percentage)
        
        # Create mask for non-outliers
        self.outlier_mask = residuals <= threshold
        self.outlier_indices = np.where(~self.outlier_mask)[0]
        
        # Apply mask to get clean data
        X_clean = X_train[self.outlier_mask]
        y_clean = y_train[self.outlier_mask]
        
        self.n_outliers_removed = len(y_train) - len(y_clean)
        outlier_pct_actual = (self.n_outliers_removed / len(y_train)) * 100
        
        logger.info(f"Removed {self.n_outliers_removed} outliers ({outlier_pct_actual:.2f}%)")
        logger.info(f"Training samples after outlier removal: {len(y_clean)}")
        
        return X_clean, y_clean, self.outlier_indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Model 1 using Model 5b methodology
        
        Args:
            X: Feature matrix (21 features)
            y: Target values (costs in original scale)
        """
        try:
            # Store feature names if not already done
            if not self.feature_names:
                _, self.feature_names = self.prepare_features(self.train_records)
            
            # Remove outliers (Model 5b methodology)
            X_clean, y_clean, _ = self.remove_outliers(X, y)
            
            # Apply transformation to clean targets
            if self.use_sqrt_transform:
                y_transformed = np.sqrt(y_clean)
                logger.info("Applied square-root transformation to clean targets")
            else:
                y_transformed = y_clean.copy()
                logger.info("No transformation applied (linear scale)")
            
            # Fit OLS regression
            self.linear_model = LinearRegression()
            self.linear_model.fit(X_clean, y_transformed)
            
            # Store the model as self.model for base class compatibility
            self.model = self.linear_model
            
            # Store coefficients with feature names
            self.coefficients = {}
            self.coefficients['intercept'] = {
                'value': float(self.linear_model.intercept_),
                'transformed': self.use_sqrt_transform
            }
            
            for i, name in enumerate(self.feature_names):
                self.coefficients[name] = {
                    'value': float(self.linear_model.coef_[i]),
                    'transformed': self.use_sqrt_transform
                }
            
            # Log top coefficients
            logger.info("\nTop 5 coefficients (by magnitude):")
            coef_list = [(name, abs(data['value'])) for name, data in self.coefficients.items() 
                        if name != 'intercept']
            coef_list.sort(key=lambda x: x[1], reverse=True)
            for name, value in coef_list[:5]:
                logger.info(f"  {name}: {self.coefficients[name]['value']:.4f}")
            
            logger.info("Model 1 fitted successfully (Model 5b methodology)")
            
        except Exception as e:
            logger.error(f"Error fitting Model 1: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted costs in original scale
        """
        if self.linear_model is None:
            raise ValueError("Model has not been fitted yet")
        
        # Predict in transformed space
        y_pred_transformed = self.linear_model.predict(X)
        
        # Transform back to original scale
        if self.use_sqrt_transform:
            # Square the predictions and handle any negative predictions
            y_pred = np.maximum(0, y_pred_transformed) ** 2
        else:
            y_pred = y_pred_transformed
        
        # Ensure non-negative predictions
        y_pred = np.maximum(0, y_pred)
        
        return y_pred
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Override to add Model 1 specific metrics and Model 5b comparison
        """
        # Get all base metrics
        metrics = super().calculate_metrics()
        
        # Add Model 1 specific metrics
        if self.linear_model is not None:
            metrics.update({
                'n_outliers_removed': self.n_outliers_removed,
                'outlier_percentage': self.outlier_percentage,
                'n_features': len(self.feature_names),
                'transformation': self.transformation,
                'use_sqrt_transform': int(self.use_sqrt_transform)
            })
        
        # Add Model 5b comparison metrics
        metrics.update(self.model_5b_comparison)
        
        # Calculate comparison deltas
        if 'r2_test' in metrics:
            metrics['r2_delta_from_2015'] = metrics['r2_test'] - self.model_5b_comparison['original_r2_2015']
        if 'rmse_test' in metrics:
            metrics['rmse_delta_from_2015'] = metrics['rmse_test'] - self.model_5b_comparison['original_rmse_2015']
        
        return metrics
    
    def calculate_sbc(self) -> float:
        """
        Calculate Schwarz Bayesian Criterion (SBC) for comparison with Model 5b
        
        Returns:
            SBC value
        """
        if self.model is None or self.y_train is None:
            return 0.0
        
        n = len(self.y_train[self.outlier_mask])  # Sample size after outlier removal
        k = len(self.feature_names) + 1  # Number of parameters (including intercept)
        
        # Calculate residuals in transformed space
        X_clean = self.X_train[self.outlier_mask]
        y_clean = self.y_train[self.outlier_mask]
        
        if self.use_sqrt_transform:
            y_transformed = np.sqrt(y_clean)
        else:
            y_transformed = y_clean
        
        y_pred_transformed = self.model.predict(X_clean)
        residuals = y_transformed - y_pred_transformed
        
        # Calculate RSS and sigma^2
        rss = np.sum(residuals ** 2)
        sigma_sq = rss / (n - k)
        
        # SBC = n * ln(sigma^2) + k * ln(n)
        sbc = n * np.log(sigma_sq) + k * np.log(n)
        
        return float(sbc)
    
    def generate_latex_commands(self) -> None:
        """
        Generate LaTeX commands with Model 5b comparison
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
                f.write(f"\\newcommand{{\\Model{model_word}SBC}}{{\\WarningRunPipeline}}\n")
                
                # Model 5b comparison commands
                f.write("\n% Model 5b Comparison Commands (2015 baseline)\n")
                f.write(f"\\newcommand{{\\Model{model_word}FiveBRSquaredTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}FiveBRMSETwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}FiveBSBCTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}FiveBSamplesTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}RSquaredDeltaFromTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}RMSEDeltaFromTwoThousandFifteen}}{{\\WarningRunPipeline}}\n")
            
            logger.info(f"Appended Model 1 specific commands to {newcommands_file}")
            
        except Exception as e:
            logger.error(f"Error appending to newcommands file: {e}")
        
        # Calculate SBC
        sbc_value = self.calculate_sbc()
        
        # Append actual values to renewcommands
        try:
            with open(renewcommands_file, 'a') as f:
                f.write("\n% Model 1 Specific Metrics\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OutliersRemoved}}{{{self.n_outliers_removed}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OutlierPercentage}}{{{self.outlier_percentage:.1f}}}\n")
                
                transform_text = "Square Root" if self.use_sqrt_transform else "None"
                f.write(f"\\renewcommand{{\\Model{model_word}Transformation}}{{{transform_text}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}NumFeatures}}{{{len(self.feature_names)}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}SBC}}{{{sbc_value:,.1f}}}\n")
                
                # Model 5b comparison values
                f.write("\n% Model 5b Comparison Values (2015 baseline)\n")
                f.write(f"\\renewcommand{{\\Model{model_word}FiveBRSquaredTwoThousandFifteen}}{{0.7998}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}FiveBRMSETwoThousandFifteen}}{{30.82}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}FiveBSBCTwoThousandFifteen}}{{159,394.3}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}FiveBSamplesTwoThousandFifteen}}{{23,215}}\n")
                
                # Calculate and write deltas
                if self.metrics:
                    r2_delta = self.metrics.get('r2_test', 0) - 0.7998
                    rmse_delta = self.metrics.get('rmse_test', 0) - 30.82
                    f.write(f"\\renewcommand{{\\Model{model_word}RSquaredDeltaFromTwoThousandFifteen}}{{{r2_delta:+.4f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}RMSEDeltaFromTwoThousandFifteen}}{{{rmse_delta:+.2f}}}\n")
            
            logger.info(f"Appended Model 1 specific values to {renewcommands_file}")
            
        except Exception as e:
            logger.error(f"Error appending to renewcommands file: {e}")
    
    def save_results(self) -> None:
        """
        Save Model 1 specific results including Model 5b comparison
        """
        # Save base results
        super().save_results()
        
        # Save coefficients with comparison to Model 5b
        coef_file = self.output_dir / "coefficients.json"
        coef_data = {
            'coefficients': self.coefficients,
            'model_5b_specification': {
                'n_features': self.MODEL_5B_N_FEATURES,
                'feature_names': self.feature_names,
                'transformation': self.transformation,
                'outlier_percentage': self.outlier_percentage
            }
        }
        with open(coef_file, 'w') as f:
            json.dump(coef_data, f, indent=2, default=str)
        
        # Save outlier information
        outlier_info = {
            'n_outliers_removed': self.n_outliers_removed,
            'outlier_percentage': self.outlier_percentage,
            'outlier_indices': self.outlier_indices.tolist() if len(self.outlier_indices) > 0 else [],
            'model_5b_comparison': {
                'original_outliers_2015': 2410,
                'original_percentage_2015': 9.4
            }
        }
        outlier_file = self.output_dir / "outlier_info.json"
        with open(outlier_file, 'w') as f:
            json.dump(outlier_info, f, indent=2)
        
        # Save Model 5b comparison
        comparison_file = self.output_dir / "model_5b_comparison.json"
        comparison_data = {
            'model_5b_2015': self.model_5b_comparison,
            'model_1_2024': {
                'r2_test': self.metrics.get('r2_test', 0),
                'rmse_test': self.metrics.get('rmse_test', 0),
                'n_train': self.metrics.get('training_samples', 0),
                'n_test': self.metrics.get('test_samples', 0),
                'sbc': self.calculate_sbc()
            },
            'deltas': {
                'r2_change': self.metrics.get('r2_delta_from_2015', 0),
                'rmse_change': self.metrics.get('rmse_delta_from_2015', 0)
            }
        }
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info("Model 1 specific results saved including Model 5b comparison")
    
    def plot_diagnostics(self) -> None:
        """
        Generate diagnostic plots including Model 5b comparison
        """
        # Generate base diagnostic plots
        super().plot_diagnostics()
        
        # Additional Model 1 specific plots
        if self.outlier_mask is not None and self.y_train is not None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            
            # 1. Outlier detection visualization
            ax = axes[0, 0]
            if self.use_sqrt_transform:
                y_sqrt = np.sqrt(self.y_train)
            else:
                y_sqrt = self.y_train
            
            prelim_model = LinearRegression()
            prelim_model.fit(self.X_train, y_sqrt)
            y_pred_sqrt = prelim_model.predict(self.X_train)
            residuals = y_sqrt - y_pred_sqrt
            
            ax.scatter(range(len(residuals)), residuals, alpha=0.5, s=10, 
                      color='blue', label=f'Kept ({len(residuals) - self.n_outliers_removed})')
            if len(self.outlier_indices) > 0:
                ax.scatter(self.outlier_indices, residuals[self.outlier_indices], 
                          color='red', s=30, label=f'Removed ({self.n_outliers_removed})', zorder=5)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.set_xlabel('Observation Index')
            ax.set_ylabel('Residuals (transformed scale)')
            ax.set_title(f'Model 5b Outlier Detection Method\n(Top {self.outlier_percentage}% removed)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Coefficient comparison (if Model 5b coefficients available)
            ax = axes[0, 1]
            if self.coefficients:
                coef_list = [(name, abs(data['value'])) for name, data in self.coefficients.items() 
                            if name != 'intercept']
                coef_list.sort(key=lambda x: x[1], reverse=True)
                top_features = coef_list[:10]
                
                names = [f[0] for f in top_features]
                values = [f[1] for f in top_features]
                
                y_pos = np.arange(len(names))
                ax.barh(y_pos, values, color='steelblue')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names, fontsize=9)
                ax.set_xlabel('|Coefficient| (transformed scale)')
                ax.set_title('Top 10 Features by Coefficient Magnitude')
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
            
            # 3. Model 5b comparison: R² over time
            ax = axes[1, 0]
            years = [2015, 2024]
            r2_values = [0.7998, self.metrics.get('r2_test', 0)]
            
            ax.plot(years, r2_values, 'o-', linewidth=2, markersize=10, color='darkgreen')
            ax.set_xlabel('Year')
            ax.set_ylabel('Test R²')
            ax.set_title('Model 5b Performance: 2015 vs 2024')
            ax.set_ylim([min(r2_values) - 0.05, max(r2_values) + 0.05])
            ax.grid(True, alpha=0.3)
            
            # Annotate points
            for year, r2 in zip(years, r2_values):
                ax.annotate(f'{r2:.4f}', xy=(year, r2), xytext=(0, 10),
                           textcoords='offset points', ha='center', fontsize=10)
            
            # 4. Feature specification table
            ax = axes[1, 1]
            ax.axis('off')
            
            table_data = [
                ['Feature Group', 'Count', 'Examples'],
                ['Living Settings', '5', 'ILSL, RH1-4'],
                ['Age Groups', '2', 'Age21-30, Age31+'],
                ['Behavioral Sum', '1', 'BSum'],
                ['Interactions', '3', 'FHFSum, SLFSum, SLBSum'],
                ['QSI Questions', '10', 'Q16, Q18, Q20, ...'],
                ['', '', ''],
                ['TOTAL', str(self.MODEL_5B_N_FEATURES), 'Model 5b Exact']
            ]
            
            table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                           colWidths=[0.35, 0.15, 0.50])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style header row
            for i in range(3):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Style total row
            for i in range(3):
                table[(7, i)].set_facecolor('#E8F5E9')
                table[(7, i)].set_text_props(weight='bold')
            
            ax.set_title('Model 5b Feature Specification\n(21 Features)', 
                        fontsize=11, fontweight='bold', pad=20)
            
            plt.suptitle('Model 1: Model 5b Re-Evaluation Diagnostics', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.output_dir / 'model1_specific_diagnostics.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model 1 specific diagnostic plots saved to {output_file}")


def main():
    """
    Run Model 1: Model 5b Re-Evaluation with 2024 data
    """
    # Initialize model with sqrt transform (Model 5b default)
    model = Model1Linear(use_sqrt_transform=True)
    
    # Run complete pipeline
    results = model.run_complete_pipeline(
        fiscal_year_start=2024,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL 1: MODEL 5B RE-EVALUATION (TAO & NIU 2015)")
    print("="*80)
    
    print(f"\nModel 5b Specification:")
    print(f"  • Features: {model.MODEL_5B_N_FEATURES} (exact match to 2015)")
    print(f"  • Transformation: {'Square-root' if model.use_sqrt_transform else 'None'}")
    print(f"  • Outlier Removal: {model.outlier_percentage}%")
    
    print(f"\nData Summary:")
    print(f"  • Total Records: {len(model.all_records)}")
    print(f"  • Training Records: {model.metrics.get('training_samples', 0)}")
    print(f"  • Test Records: {model.metrics.get('test_samples', 0)}")
    print(f"  • Outliers Removed: {model.n_outliers_removed} ({model.outlier_percentage}%)")
    
    print(f"\nModel 5b Comparison (2015 vs 2024):")
    print(f"  {'Metric':<20} {'2015 (Original)':<20} {'2024 (Current)':<20} {'Delta':<15}")
    print(f"  {'-'*75}")
    print(f"  {'R² Score':<20} {0.7998:<20.4f} {model.metrics.get('r2_test', 0):<20.4f} "
          f"{model.metrics.get('r2_delta_from_2015', 0):+.4f}")
    print(f"  {'RMSE':<20} ${30.82:<19.2f} ${model.metrics.get('rmse_test', 0):<19.2f} "
          f"${model.metrics.get('rmse_delta_from_2015', 0):+.2f}")
    print(f"  {'SBC':<20} {159394.3:<20,.1f} {model.calculate_sbc():<20,.1f} "
          f"{model.calculate_sbc() - 159394.3:+,.1f}")
    print(f"  {'Training N':<20} {23215:<20,} {model.metrics.get('training_samples', 0):<20,}")
    
    print(f"\nCurrent Performance (2024):")
    print(f"  • Training R²: {model.metrics.get('r2_train', 0):.4f}")
    print(f"  • Test R²: {model.metrics.get('r2_test', 0):.4f}")
    print(f"  • RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    print(f"  • MAE: ${model.metrics.get('mae_test', 0):,.2f}")
    print(f"  • MAPE: {model.metrics.get('mape_test', 0):.2f}%")
    print(f"  • 10-Fold CV R²: {model.metrics.get('cv_mean', 0):.4f} ± {model.metrics.get('cv_std', 0):.4f}")
    
    print(f"\nAccuracy Bands:")
    print(f"  • Within $5,000: {model.metrics.get('within_5k', 0):.1f}%")
    print(f"  • Within $10,000: {model.metrics.get('within_10k', 0):.1f}%")
    print(f"  • Within $20,000: {model.metrics.get('within_20k', 0):.1f}%")
    
    print("\nFiles Generated:")
    for file in sorted(model.output_dir.glob("*")):
        print(f"  • {file.name}")
    
    print("="*80)
    
    return model


if __name__ == "__main__":
    model = main()