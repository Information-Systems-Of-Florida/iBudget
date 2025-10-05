"""
model_8_bayesian.py
===================
Model 8: Bayesian Linear Regression with Robust Features
Conjugate prior approach with uncertainty quantification
Uses only validated features from Model 5b: 19 features total
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# For Bayesian implementation
from sklearn.linear_model import BayesianRidge

# Import base class - CRITICAL
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

class Model8Bayesian(BaseiBudgetModel):
    """
    Model 8: Bayesian Linear Regression
    
    Key features:
    - Conjugate Normal-Inverse-Gamma prior
    - Full posterior distributions
    - Credible intervals for predictions
    - Uses ONLY robust features from Model 5b (19 features)
    - Square-root transformation of costs
    
    Robust Features:
    - 5 Living Settings: ILSL, RH1, RH2, RH3, RH4 (FH as reference)
    - 2 Age Groups: Age21_30, Age31Plus (Age3_20 as reference)
    - 10 QSI Questions: Q16, Q18, Q20, Q21, Q23, Q28, Q33, Q34, Q36, Q43
    - 2 Summary Scores: BSum, FSum
    """
    
    def __init__(self, fiscal_year_start: int = 2023, fiscal_year_end: int = 2024, use_fy2024_only: bool = True):
        """Initialize Model 8 - follow Model 1 pattern exactly"""
        super().__init__(model_id=8, model_name="Bayesian Linear Regression")
        self.use_fy2024_only = use_fy2024_only
        self.fiscal_years_used = "2024" if use_fy2024_only else "2023-2024"
        
        # Store fiscal years for data loading
        self.fiscal_year_start = fiscal_year_start
        self.fiscal_year_end = fiscal_year_end
        
        # Bayesian model parameters
        self.alpha_1 = 1e-6  # Shape parameter for Gamma prior on alpha
        self.alpha_2 = 1e-6  # Rate parameter for Gamma prior on alpha
        self.lambda_1 = 1e-6  # Shape parameter for Gamma prior on lambda
        self.lambda_2 = 1e-6  # Rate parameter for Gamma prior on lambda
        
        # Initialize model
        self.model = None
        
        # Define robust features from Model 5b validation
        self.robust_qsi_questions = [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]
        self.use_living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
        self.use_age_groups = ['Age21_30', 'Age31Plus']
        self.use_summary_scores = ['bsum', 'fsum']
        
        # Posterior statistics
        self.posterior_mean = None
        self.posterior_std = None
        self.credible_intervals = {}
        
        # Model evidence (marginal likelihood)
        self.log_marginal_likelihood = None
        
        # Coefficients storage (posterior means)
        self.coefficients = {}
        self.num_parameters = 0
        
        # Hardcoded cost estimates (as per specification)
        self.implementation_cost = 165000  # $165,000
        self.annual_operating_cost = 35000  # $35,000
        self.three_year_tco = 270000  # $270,000
        
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Override split_data - COPY EXACTLY from Model 1
        """
        # Handle boolean test_size issue from base class
        if isinstance(test_size, bool):
            test_size = 0.2
        
        logger.info("Splitting data into train and test sets...")
        
        # Convert test_size to actual count
        n_test = int(len(self.all_records) * test_size)
        n_train = len(self.all_records) - n_test
        
        # Create indices for splitting
        indices = np.arange(len(self.all_records))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Split the records
        self.train_records = [self.all_records[i] for i in train_indices]
        self.test_records = [self.all_records[i] for i in test_indices]
        
        logger.info(f"Data split: {len(self.train_records)} training, {len(self.test_records)} test")
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features - ONLY ROBUST FEATURES from Model 5b validation
        Total of 19 validated features
        """
        if not records:
            return np.array([]), []
        
        features_list = []
        
        for record in records:
            row_features = []
            
            # 1. Living setting dummies (5 features, drop FH as reference)
            for setting in self.use_living_settings:
                value = 1.0 if record.living_setting == setting else 0.0
                row_features.append(value)
            
            # 2. Age group dummies (2 features, drop Age3_20 as reference)
            for age_group in self.use_age_groups:
                value = 1.0 if record.age_group == age_group else 0.0
                row_features.append(value)
            
            # 3. ONLY Robust QSI questions from Model 5b (10 features)
            for q_num in self.robust_qsi_questions:
                value = getattr(record, f'q{q_num}', 0)
                row_features.append(float(value))
            
            # 4. Summary scores (2 features)
            row_features.append(float(record.bsum))  # Behavioral sum
            row_features.append(float(record.fsum))  # Functional sum
            
            features_list.append(row_features)
        
        # Build feature names ONCE (follow Model 1 pattern)
        if not self.feature_names:
            feature_names = []
            
            # Living settings
            for setting in self.use_living_settings:
                feature_names.append(f'Live_{setting}')
            
            # Age groups
            for age_group in self.use_age_groups:
                feature_names.append(f'Age_{age_group}')
            
            # QSI questions
            for q_num in self.robust_qsi_questions:
                feature_names.append(f'Q{q_num}')
            
            # Summary scores
            feature_names.append('BSum')
            feature_names.append('FSum')
            
            self.feature_names = feature_names
        
        X = np.array(features_list, dtype=np.float64)
        self.num_parameters = len(self.feature_names) + 1  # +1 for intercept
        
        logger.info(f"Prepared ROBUST features matrix with shape {X.shape}")
        logger.info(f"Using {len(self.feature_names)} robust features validated in Model 5b")
        logger.info(f"Features: 5 living settings + 2 age groups + 10 QSI questions + 2 summary scores")
        
        return X, self.feature_names
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Bayesian Linear Regression model with conjugate priors
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (square-root transformed costs)
        """
        logger.info("Fitting Bayesian Linear Regression with robust features...")
        logger.info(f"Training samples: {len(y)}, Features: {X.shape[1]}")
        
        # Initialize Bayesian Ridge Regression (implements conjugate Normal-Inverse-Gamma)
        self.model = BayesianRidge(
            max_iter=300,
            tol=1e-3,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            compute_score=True,
            fit_intercept=True
        )
        
        # Fit model
        self.model.fit(X, y)
        
        # Extract posterior statistics
        self.posterior_mean = self.model.coef_
        
        # Calculate posterior standard deviations
        if hasattr(self.model, 'sigma_'):
            self.posterior_std = np.sqrt(np.diag(self.model.sigma_))
        else:
            # Approximate using the precision parameter
            self.posterior_std = np.sqrt(1 / (self.model.lambda_ * np.ones(len(self.model.coef_))))
        
        # Calculate 95% credible intervals
        for i, name in enumerate(self.feature_names):
            mean = self.posterior_mean[i]
            std = self.posterior_std[i] if i < len(self.posterior_std) else 0.1
            self.credible_intervals[name] = {
                'mean': mean,
                'std': std,
                'lower_95': mean - 1.96 * std,
                'upper_95': mean + 1.96 * std
            }
        
        # Store coefficients (posterior means) for compatibility
        self.coefficients = {
            'intercept': self.model.intercept_,
            **{name: coef for name, coef in zip(self.feature_names, self.model.coef_)}
        }
        
        # Log marginal likelihood (model evidence)
        if hasattr(self.model, 'scores_'):
            self.log_marginal_likelihood = self.model.scores_[-1]
        
        logger.info(f"Bayesian model fitted with {self.num_parameters} parameters")
        logger.info(f"Alpha (noise precision): {self.model.alpha_:.4f}")
        logger.info(f"Lambda (weights precision): {self.model.lambda_:.4f}")
        if self.log_marginal_likelihood:
            logger.info(f"Log marginal likelihood: {self.log_marginal_likelihood:.2f}")
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """
        Make predictions with uncertainty quantification
        
        Args:
            X: Feature matrix
            return_std: If True, return both mean and standard deviation
            
        Returns:
            Predictions (and std if return_std=True)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if return_std:
            # Return both mean and standard deviation
            return self.model.predict(X, return_std=True)
        else:
            return self.model.predict(X)
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation - follows Model 1 pattern
        """
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        # Prepare features if not already done
        if self.X_train is None or self.y_train is None:
            if self.train_records:
                self.X_train, _ = self.prepare_features(self.train_records)
                self.y_train = np.array([np.sqrt(r.total_cost) for r in self.train_records])
            else:
                raise ValueError("No training records available for cross-validation")
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        cv_log_likelihood = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train), 1):
            # Split data
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_val = self.y_train[val_idx]
            
            # Train Bayesian model
            fold_model = BayesianRidge(
                max_iter=300,
                tol=1e-3,
                alpha_1=self.alpha_1,
                alpha_2=self.alpha_2,
                lambda_1=self.lambda_1,
                lambda_2=self.lambda_2,
                compute_score=True,
                fit_intercept=True
            )
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Validate
            y_pred = fold_model.predict(X_fold_val)
            fold_r2 = r2_score(y_fold_val, y_pred)
            cv_scores.append(fold_r2)
            
            # Store log likelihood if available
            if hasattr(fold_model, 'scores_'):
                cv_log_likelihood.append(fold_model.scores_[-1])
            
            logger.info(f"Fold {fold}: R² = {fold_r2:.4f}")
        
        self.cv_scores = cv_scores
        
        # Return with EXACT keys expected by base class
        results = {
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
            'cv_r2_scores': cv_scores,
            'cv_log_likelihood_mean': np.mean(cv_log_likelihood) if cv_log_likelihood else None
        }
        
        logger.info(f"Cross-validation R²: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
        
        return results
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive metrics including Bayesian-specific measures
        """
        # Get base metrics
        metrics = super().calculate_metrics()
        
        if self.model is not None:
            # Add Bayesian specific metrics
            metrics['alpha'] = float(self.model.alpha_)
            metrics['lambda'] = float(self.model.lambda_)
            
            # Log marginal likelihood
            if self.log_marginal_likelihood is not None:
                metrics['log_marginal_likelihood'] = float(self.log_marginal_likelihood)
            
            # Effective number of parameters (from ARD)
            if hasattr(self.model, 'lambda_'):
                # Effective number of parameters based on precision
                effective_params = np.sum(self.model.alpha_ / (self.model.alpha_ + self.model.lambda_))
                metrics['effective_parameters'] = float(effective_params)
            
            # Credible interval widths (average)
            if self.credible_intervals:
                avg_width = np.mean([v['upper_95'] - v['lower_95'] 
                                    for v in self.credible_intervals.values()])
                metrics['avg_credible_interval_width'] = float(avg_width)
            
            # Add hardcoded costs (as specified)
            metrics['implementation_cost'] = self.implementation_cost
            metrics['annual_operating_cost'] = self.annual_operating_cost
            metrics['three_year_tco'] = self.three_year_tco
            
            # Add number of robust features
            metrics['n_robust_features'] = len(self.feature_names)
        
        return metrics
    
    def generate_diagnostic_plots(self) -> None:
        """
        Generate diagnostic plots including Bayesian-specific visualizations
        """
        # Call parent diagnostic plots first
        super().generate_diagnostic_plots()
        
        # Add Bayesian specific plots
        self.plot_posterior_distributions()
        self.plot_credible_intervals()
        self.plot_predictive_uncertainty()
    
    def plot_posterior_distributions(self) -> None:
        """
        Plot posterior distributions for key coefficients
        """
        if self.posterior_mean is None:
            logger.warning("No posterior distributions to plot")
            return
        
        # Select top 6 features by absolute posterior mean
        top_indices = np.argsort(np.abs(self.posterior_mean))[-6:]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (ax, feat_idx) in enumerate(zip(axes, top_indices)):
            if feat_idx < len(self.feature_names):
                feat_name = self.feature_names[feat_idx]
                mean = self.posterior_mean[feat_idx]
                std = self.posterior_std[feat_idx] if feat_idx < len(self.posterior_std) else 0.1
                
                # Plot posterior distribution
                x = np.linspace(mean - 4*std, mean + 4*std, 100)
                y = stats.norm.pdf(x, mean, std)
                ax.plot(x, y, 'b-', lw=2)
                ax.fill_between(x, 0, y, alpha=0.3)
                
                # Add credible interval
                ci = self.credible_intervals.get(feat_name, {})
                if ci:
                    ax.axvline(ci['lower_95'], color='r', linestyle='--', alpha=0.5, label='95% CI')
                    ax.axvline(ci['upper_95'], color='r', linestyle='--', alpha=0.5)
                
                # Add mean line
                ax.axvline(mean, color='g', linestyle='-', alpha=0.7, label='Mean')
                
                ax.set_title(f'{feat_name}', fontsize=10)
                ax.set_xlabel('Coefficient Value')
                ax.set_ylabel('Posterior Density')
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
        
        plt.suptitle('Posterior Distributions of Top 6 Robust Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = self.output_dir / 'posterior_distributions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Posterior distributions saved to {plot_path}")
    
    def plot_credible_intervals(self) -> None:
        """
        Plot 95% credible intervals for all coefficients
        """
        if not self.credible_intervals:
            logger.warning("No credible intervals to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        features = list(self.credible_intervals.keys())
        means = [self.credible_intervals[f]['mean'] for f in features]
        lowers = [self.credible_intervals[f]['lower_95'] for f in features]
        uppers = [self.credible_intervals[f]['upper_95'] for f in features]
        
        # Sort by absolute mean value
        sorted_idx = np.argsort(np.abs(means))[::-1]  # Descending
        features = [features[i] for i in sorted_idx]
        means = [means[i] for i in sorted_idx]
        lowers = [lowers[i] for i in sorted_idx]
        uppers = [uppers[i] for i in sorted_idx]
        
        # Plot intervals
        y_pos = range(len(features))
        errors = [np.array(means) - np.array(lowers), 
                  np.array(uppers) - np.array(means)]
        
        ax.errorbar(means, y_pos, xerr=errors, fmt='o', capsize=3, capthick=1.5, 
                   markersize=6, linewidth=1.5, color='steelblue')
        
        # Add zero line
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Zero Effect')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Coefficient Value (√$ scale)', fontsize=11, fontweight='bold')
        ax.set_title('95% Credible Intervals for Robust Features', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'credible_intervals.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Credible intervals plot saved to {plot_path}")
    
    def plot_predictive_uncertainty(self) -> None:
        """
        Plot predictions with uncertainty bands
        """
        if self.test_predictions is None or self.X_test is None:
            logger.warning("No test predictions available for uncertainty plot")
            return
        
        # Get predictions with uncertainty
        y_pred, y_std = self.predict(self.X_test, return_std=True)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Sort by predicted value for better visualization
        sorted_idx = np.argsort(y_pred)
        y_pred_sorted = y_pred[sorted_idx]
        y_std_sorted = y_std[sorted_idx]
        y_true_sorted = self.y_test[sorted_idx]
        
        x = range(len(y_pred_sorted))
        
        # Plot predictions with uncertainty bands
        ax.plot(x, y_pred_sorted, 'b-', label='Predictions', alpha=0.7, linewidth=1.5)
        ax.fill_between(x, 
                        y_pred_sorted - 1.96 * y_std_sorted,
                        y_pred_sorted + 1.96 * y_std_sorted,
                        alpha=0.3, label='95% Credible Interval', color='lightblue')
        
        # Plot actual values
        ax.scatter(x, y_true_sorted, c='red', s=2, alpha=0.4, label='Actual Values')
        
        ax.set_xlabel('Sample (sorted by prediction)', fontsize=11, fontweight='bold')
        ax.set_ylabel('√Cost', fontsize=11, fontweight='bold')
        ax.set_title('Predictive Uncertainty Quantification (Test Set)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'predictive_uncertainty.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Predictive uncertainty plot saved to {plot_path}")
    
    def generate_latex_commands(self) -> None:
        """
        Generate LaTeX commands including Bayesian-specific metrics
        """
        # Call parent method first
        super().generate_latex_commands()
        
        # Add Bayesian specific commands
        model_word = 'Eight'
        
        # Read existing files
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        # Append Bayesian-specific commands to newcommands
        with open(newcommands_file, 'a') as f:
            f.write("\n% Bayesian Specific Commands\n")
            f.write(f"\\newcommand{{\\Model{model_word}Alpha}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Lambda}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}EffectiveParams}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}AvgCredibleWidth}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}LogMarginalLikelihood}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}NRobustFeatures}}{{\\WarningRunPipeline}}\n")
            # Hardcoded costs
            f.write(f"\\newcommand{{\\Model{model_word}ImplementationCost}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}AnnualCost}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}ThreeYearTCO}}{{\\WarningRunPipeline}}\n")
        
        # Append actual values to renewcommands
        with open(renewcommands_file, 'a') as f:
            f.write("\n% Bayesian Specific Metrics\n")
            
            if self.model is not None:
                f.write(f"\\renewcommand{{\\Model{model_word}Alpha}}{{{self.model.alpha_:.4f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Lambda}}{{{self.model.lambda_:.4f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}NRobustFeatures}}{{{len(self.feature_names)}}}\n")
                
                if 'effective_parameters' in self.metrics:
                    f.write(f"\\renewcommand{{\\Model{model_word}EffectiveParams}}{{{self.metrics['effective_parameters']:.1f}}}\n")
                
                if 'avg_credible_interval_width' in self.metrics:
                    f.write(f"\\renewcommand{{\\Model{model_word}AvgCredibleWidth}}{{{self.metrics['avg_credible_interval_width']:.3f}}}\n")
                
                if self.log_marginal_likelihood is not None:
                    f.write(f"\\renewcommand{{\\Model{model_word}LogMarginalLikelihood}}{{{self.log_marginal_likelihood:.1f}}}\n")
                
                # Hardcoded costs (as specified)
                f.write(f"\\renewcommand{{\\Model{model_word}ImplementationCost}}{{\\${self.implementation_cost:,}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}AnnualCost}}{{\\${self.annual_operating_cost:,}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}ThreeYearTCO}}{{\\${self.three_year_tco:,}}}\n")
        
        logger.info(f"LaTeX commands generated in {self.output_dir}")

# Main execution - FOLLOW Model 1 pattern exactly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*60)
    logger.info("Model 8: Bayesian Linear Regression with Robust Features")
    logger.info("="*60)
    
    # Initialize model
    model = Model8Bayesian()
    
    # Display feature specification
    logger.info("\nRobust Feature Specification:")
    logger.info(f"  Living Settings: {model.use_living_settings}")
    logger.info(f"  Age Groups: {model.use_age_groups}")
    logger.info(f"  QSI Questions: {model.robust_qsi_questions}")
    logger.info(f"  Summary Scores: {model.use_summary_scores}")
    logger.info(f"  Total Features: {len(model.use_living_settings) + len(model.use_age_groups) + len(model.robust_qsi_questions) + len(model.use_summary_scores)}")
    
    # Run complete pipeline
    try:
        model.run_complete_pipeline(
            fiscal_year_start=2023,
            fiscal_year_end=2024
        )
        logger.info("\n" + "="*60)
        logger.info("Model 8 pipeline completed successfully!")
        logger.info("="*60)
        
        # Display key metrics
        logger.info("\n" + "="*60)
        logger.info("KEY PERFORMANCE METRICS")
        logger.info("="*60)
        
        metrics = model.metrics
        print(f"\nPrediction Performance:")
        print(f"  Test R²: {metrics.get('r2_test', 0):.4f}")
        print(f"  Test RMSE: ${metrics.get('rmse_test', 0):,.2f}")
        print(f"  Test MAE: ${metrics.get('mae_test', 0):,.2f}")
        print(f"  Test MAPE: {metrics.get('mape_test', 0):.2f}%")
        
        print(f"\nModel Characteristics:")
        print(f"  Robust Features Used: {len(model.feature_names)}")
        print(f"  Effective Parameters: {metrics.get('effective_parameters', 0):.1f}")
        print(f"  Alpha (noise precision): {metrics.get('alpha', 0):.4f}")
        print(f"  Lambda (weights precision): {metrics.get('lambda', 0):.4f}")
        
        print(f"\nCross-Validation:")
        print(f"  CV R²: {metrics.get('cv_r2_mean', 0):.4f} ± {metrics.get('cv_r2_std', 0):.4f}")
        
        if 'log_marginal_likelihood' in metrics:
            print(f"\nBayesian Evidence:")
            print(f"  Log Marginal Likelihood: {metrics.get('log_marginal_likelihood', 0):.2f}")
        
        if 'avg_credible_interval_width' in metrics:
            print(f"\nUncertainty Quantification:")
            print(f"  Avg Credible Interval Width: {metrics.get('avg_credible_interval_width', 0):.3f}")
        
        # Display costs (hardcoded as specified)
        print("\n" + "="*60)
        print("IMPLEMENTATION COSTS")
        print("="*60)
        print(f"  Implementation: ${model.implementation_cost:,}")
        print(f"  Annual Operating: ${model.annual_operating_cost:,}")
        print(f"  3-Year TCO: ${model.three_year_tco:,}")
        
        # Display top credible intervals
        if model.credible_intervals:
            print("\n" + "="*60)
            print("TOP 5 ROBUST FEATURES (95% Credible Intervals)")
            print("="*60)
            sorted_features = sorted(model.credible_intervals.items(), 
                                   key=lambda x: abs(x[1]['mean']), 
                                   reverse=True)[:5]
            for feat, ci in sorted_features:
                print(f"  {feat:<20} Mean: {ci['mean']:+.4f}  [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}]")
        
        print("\n" + "="*60)
        print(f"All outputs saved to: {model.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running Model 8 pipeline: {str(e)}", exc_info=True)
        raise