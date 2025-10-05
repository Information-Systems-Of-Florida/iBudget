"""
model_3_robust.py
=================
Model 3: Robust Linear Regression with Huber Estimation
FIXED: Generates ALL LaTeX commands including model-specific ones
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import HuberRegressor
from scipy.stats import median_abs_deviation
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from pathlib import Path
import json

from base_model import BaseiBudgetModel, ConsumerRecord

logger = logging.getLogger(__name__)


class Model3Robust(BaseiBudgetModel):
    """Model 3: Robust Linear Regression with Huber Estimation"""
    
    def __init__(self, use_fy2024_only: bool = True):
        super().__init__(model_id=3, model_name="Robust Linear Regression")
        self.use_fy2024_only = use_fy2024_only
        self.fiscal_years_used = "2024" if use_fy2024_only else "2020-2021"
    
        self.model = None
        self.weights = None
        self.epsilon = 1.35
        self.scale_estimate = None
        self.num_iterations = None
        self.converged = False
        self.num_parameters = 23
        self.uses_all_data = True
        self.outlier_percentage = 0.0
        
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """Override to handle boolean test_size"""
        if isinstance(test_size, bool):
            test_size = 0.2 if test_size else 0.0
        
        if not self.all_records:
            raise ValueError("No records loaded. Call load_data() first.")
        
        n_total = len(self.all_records)
        n_test = int(n_total * test_size)
        n_train = n_total - n_test
        
        np.random.seed(random_state)
        indices = np.random.permutation(n_total)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        self.train_records = [self.all_records[i] for i in train_indices]
        self.test_records = [self.all_records[i] for i in test_indices]
        
        logger.info(f"Data split: {len(self.train_records)} training, {len(self.test_records)} test")
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """Prepare features using Model 5b structure"""
        if not records:
            return np.array([]).reshape(0, 22), []
        
        living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
        age_groups = ['Age21_30', 'Age31Plus']
        selected_qsi = [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]
        
        features_list = []
        
        for record in records:
            row_features = []
            
            for ls in living_settings:
                row_features.append(1.0 if record.living_setting == ls else 0.0)
            
            for age in age_groups:
                row_features.append(1.0 if record.age_group == age else 0.0)
            
            for q in selected_qsi:
                value = getattr(record, f'q{q}', 0)
                row_features.append(float(value) if value is not None else 0.0)
            
            row_features.append(float(record.bsum) if record.bsum is not None else 0.0)
            row_features.append(float(record.fsum) if record.fsum is not None else 0.0)
            
            dd = record.developmental_disability if hasattr(record, 'developmental_disability') else ""
            row_features.append(1.0 if 'autism' in str(dd).lower() else 0.0)
            row_features.append(1.0 if 'cerebral' in str(dd).lower() else 0.0)
            row_features.append(1.0 if 'down' in str(dd).lower() else 0.0)
            
            features_list.append(row_features)
        
        if not self.feature_names:
            feature_names = []
            for ls in living_settings:
                feature_names.append(f'living_{ls}')
            for age in age_groups:
                feature_names.append(f'age_{age}')
            for q in selected_qsi:
                feature_names.append(f'Q{q}')
            feature_names.extend(['BSum', 'FSum'])
            feature_names.extend(['DD_Autism', 'DD_Cerebral', 'DD_Down'])
            self.feature_names = feature_names
        
        X = np.array(features_list, dtype=np.float64)
        
        if len(features_list) > 0:
            self.num_parameters = X.shape[1] + 1
            logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} records")
        else:
            self.num_parameters = len(self.feature_names) + 1
        
        return X, self.feature_names
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Huber robust regression"""
        logger.info("Fitting Huber robust regression...")
        
        self.model = HuberRegressor(
            epsilon=self.epsilon,
            max_iter=100,
            alpha=0.0,
            tol=1e-5,
            fit_intercept=True
        )
        
        self.model.fit(X, y)
        
        predictions = self.model.predict(X)
        residuals = y - predictions
        
        self.scale_estimate = median_abs_deviation(residuals, scale='normal')
        self.weights = self._calculate_huber_weights(residuals)
        self.converged = self.model.n_iter_ < self.model.max_iter
        self.num_iterations = self.model.n_iter_
        
        logger.info(f"Model converged: {self.converged}")
        logger.info(f"Iterations: {self.num_iterations}")
        logger.info(f"Scale estimate: {self.scale_estimate:.4f}")
        logger.info(f"Mean weight: {np.mean(self.weights):.4f}")
    
    def _calculate_huber_weights(self, residuals: np.ndarray) -> np.ndarray:
        """Calculate Huber weights"""
        if self.scale_estimate == 0:
            return np.ones_like(residuals)
        
        std_residuals = np.abs(residuals / self.scale_estimate)
        weights = np.where(
            std_residuals <= self.epsilon,
            1.0,
            self.epsilon / std_residuals
        )
        return weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate model-specific metrics"""
        metrics = super().calculate_metrics()
        
        if self.weights is not None:
            metrics['mean_weight'] = float(np.mean(self.weights))
            metrics['min_weight'] = float(np.min(self.weights))
            metrics['max_weight'] = float(np.max(self.weights))
            full_weight_pct = np.mean(self.weights >= 0.99) * 100
            metrics['full_weight_pct'] = float(full_weight_pct)
            outliers_detected = np.sum(self.weights < 0.99)
            metrics['outliers_detected'] = int(outliers_detected)
            metrics['outlier_percentage'] = float(outliers_detected / len(self.weights) * 100)
        
        if self.converged is not None:
            metrics['converged'] = 'Yes' if self.converged else 'No'
            metrics['num_iterations'] = int(self.num_iterations) if self.num_iterations else 0
        
        if self.scale_estimate is not None:
            metrics['scale_estimate'] = float(self.scale_estimate)
            metrics['epsilon'] = float(self.epsilon)
        
        metrics['num_parameters'] = self.num_parameters
        
        return metrics
    
    def generate_latex_commands(self) -> None:
        """Override to add Model 3-specific LaTeX commands"""
        # Generate base commands first
        super().generate_latex_commands()
        
        # Now add Model 3-specific commands
        model_word = "Three"
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append command definitions to newcommands
        with open(newcommands_file, 'a') as f:
            f.write("\n% Model 3 Robust Regression Specific Commands\n")
            f.write(f"\\newcommand{{\\Model{model_word}Epsilon}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}OutliersDetected}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}OutlierPercentage}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}ScaleEstimate}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MeanWeight}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MinWeight}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MaxWeight}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}FullWeightPct}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Converged}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}NumIterations}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Parameters}}{{\\WarningRunPipeline}}\n")
        
        # Append actual values to renewcommands
        with open(renewcommands_file, 'a') as f:
            f.write("\n% Model 3 Robust Regression Specific Values\n")
            
            if self.epsilon is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}Epsilon}}{{{self.epsilon:.2f}}}\n")
            
            if 'outliers_detected' in self.metrics:
                f.write(f"\\renewcommand{{\\Model{model_word}OutliersDetected}}{{{self.metrics['outliers_detected']}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OutlierPercentage}}{{{self.metrics['outlier_percentage']:.1f}}}\n")
            
            if self.scale_estimate is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}ScaleEstimate}}{{{self.scale_estimate:.4f}}}\n")
            
            if 'mean_weight' in self.metrics:
                f.write(f"\\renewcommand{{\\Model{model_word}MeanWeight}}{{{self.metrics['mean_weight']:.4f}}}\n")
            
            if 'min_weight' in self.metrics:
                f.write(f"\\renewcommand{{\\Model{model_word}MinWeight}}{{{self.metrics['min_weight']:.4f}}}\n")
            
            if 'max_weight' in self.metrics:
                f.write(f"\\renewcommand{{\\Model{model_word}MaxWeight}}{{{self.metrics['max_weight']:.4f}}}\n")
            
            if 'full_weight_pct' in self.metrics:
                f.write(f"\\renewcommand{{\\Model{model_word}FullWeightPct}}{{{self.metrics['full_weight_pct']:.1f}}}\n")
            
            if self.converged is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}Converged}}{{{self.metrics['converged']}}}\n")
            
            if self.num_iterations is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}NumIterations}}{{{self.num_iterations}}}\n")
            
            if self.num_parameters is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}Parameters}}{{{self.num_parameters}}}\n")
        
        logger.info("Generated Model 3-specific LaTeX commands")
    
    def plot_diagnostics(self) -> None:
        """Create diagnostic plots"""
        if self.test_predictions is None or self.y_test is None:
            logger.warning("Cannot create diagnostic plots - no predictions available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model 3: Robust Linear Regression Diagnostics', fontsize=16, fontweight='bold')
        
        # 1. Predicted vs Actual
        ax = axes[0, 0]
        ax.scatter(self.y_test, self.test_predictions, alpha=0.3, s=10)
        ax.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual ?Cost')
        ax.set_ylabel('Predicted ?Cost')
        ax.set_title('Predicted vs Actual (Test Set)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Residual Plot with weights
        ax = axes[0, 1]
        train_residuals = self.y_train - self.model.predict(self.X_train)
        scatter = ax.scatter(self.model.predict(self.X_train), train_residuals,
                            c=self.weights, cmap='RdYlGn', alpha=0.5, s=10, vmin=0, vmax=1)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals (colored by Huber weight)')
        plt.colorbar(scatter, ax=ax, label='Weight')
        ax.grid(True, alpha=0.3)
        
        # 3. Weight Distribution
        ax = axes[0, 2]
        ax.hist(self.weights, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=np.mean(self.weights), color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(self.weights):.3f}')
        ax.axvline(x=0.99, color='orange', linestyle=':', linewidth=2,
                   label='Full Weight Cutoff')
        ax.set_xlabel('Huber Weight')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Weight Distribution ({np.mean(self.weights>=0.99)*100:.1f}% full)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Q-Q Plot
        ax = axes[1, 0]
        test_residuals = self.y_test - self.test_predictions
        stats.probplot(test_residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Test Residuals)')
        ax.grid(True, alpha=0.3)
        
        # 5. Residual Distribution
        ax = axes[1, 1]
        ax.hist(test_residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Test Residual Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Performance by Cost Quartile
        ax = axes[1, 2]
        quartiles = pd.qcut(self.y_test, q=4, labels=['Q1-Low', 'Q2', 'Q3', 'Q4-High'], duplicates='drop')
        quartile_r2 = []
        quartile_labels = []
        for q in ['Q1-Low', 'Q2', 'Q3', 'Q4-High']:
            mask = quartiles == q
            if mask.sum() > 0:
                q_r2 = r2_score(self.y_test[mask], self.test_predictions[mask])
                quartile_r2.append(q_r2)
                quartile_labels.append(q)
        
        colors = ['#d62728' if r2 < 0.7 else '#2ca02c' if r2 > 0.85 else '#ff7f0e' for r2 in quartile_r2]
        ax.bar(range(len(quartile_r2)), quartile_r2, edgecolor='black', color=colors)
        ax.set_xticks(range(len(quartile_r2)))
        ax.set_xticklabels(quartile_labels)
        ax.set_ylabel('R^2')
        ax.set_title('Performance by Cost Quartile')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Target: 0.80')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'diagnostic_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plots saved")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("MODEL 3: ROBUST LINEAR REGRESSION WITH HUBER ESTIMATION")
    print("="*80)
    
    model = Model3Robust()
    
    results = model.run_complete_pipeline(
        fiscal_year_start=2023,
        fiscal_year_end=2024,
        test_size=0.2,
        perform_cv=True,
        n_cv_folds=10
    )
    
    print(f"\nConfiguration:")
    print(f"  ? Method: Huber M-estimator")
    print(f"  ? Epsilon: {model.epsilon}")
    print(f"  ? Data Utilization: 100%")
    print(f"  ? Features: {len(model.feature_names)}")
    
    print(f"\nModel Fitting:")
    print(f"  ? Converged: {model.converged}")
    print(f"  ? Iterations: {model.num_iterations}")
    print(f"  ? Scale Estimate: {model.scale_estimate:.4f}")
    
    print(f"\nWeight Statistics:")
    print(f"  ? Mean: {model.metrics.get('mean_weight', 0):.4f}")
    print(f"  ? Min: {model.metrics.get('min_weight', 0):.4f}")
    print(f"  ? % Full: {model.metrics.get('full_weight_pct', 0):.1f}%")
    print(f"  ? Downweighted: {model.metrics.get('outliers_detected', 0)} ({model.metrics.get('outlier_percentage', 0):.1f}%)")
    
    print(f"\nPerformance:")
    print(f"  ? Test R^2: {model.metrics.get('r2_test', 0):.4f}")
    print(f"  ? RMSE: ${model.metrics.get('rmse_test', 0):,.2f}")
    print(f"  ? CV R^2: {model.metrics.get('cv_r2_mean', 0):.4f} +- {model.metrics.get('cv_r2_std', 0):.4f}")
    
    print("\nFiles Generated:")
    for file in sorted(model.output_dir.glob("*")):
        print(f"  ? {file.name}")
    
    print("="*80)
    
    return model


if __name__ == "__main__":
    model = main()
