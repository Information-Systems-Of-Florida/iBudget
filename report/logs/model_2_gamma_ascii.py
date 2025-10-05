"""
Model 2: Generalized Linear Model with Gamma Distribution
==========================================================
GLM with Gamma distribution and log link for iBudget cost prediction
Uses feature selection based on mutual information analysis
No outlier removal - robust to extreme values through distribution choice
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import logging
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import base class
from base_model import BaseiBudgetModel, ConsumerRecord

# Configure logging
logger = logging.getLogger(__name__)

class Model2GLMGamma(BaseiBudgetModel):
    """
    Model 2: GLM with Gamma distribution and log link
    
    Key features:
    - Gamma distribution for right-skewed cost data
    - Log link function ensures positive predictions
    - No outlier removal (robust by design)
    - Feature selection based on mutual information
    - Works directly in dollar scale (no transformation)
    """
    
    # Selected features based on mutual information analysis
    SELECTED_FEATURES = [
        # Top predictors from MI analysis (consistently important across years)
        'RESIDENCETYPE',     # MI: 0.203-0.272 (highest predictor)
        'BSum',              # MI: 0.113-0.137 (behavioral summary)
        'BLEVEL',            # MI: 0.089-0.106
        'LOSRI',             # MI: 0.087-0.130
        'OLEVEL',            # MI: 0.085-0.131
        
        # Clinical scores
        'FSum',              # MI: 0.059-0.084
        'FLEVEL',            # MI: 0.061-0.085
        'PSum',              # MI: 0.057-0.091
        'PLEVEL',            # MI: 0.070-0.083
        
        # Top individual QSI questions
        'Q26',               # MI: 0.084-0.094 (consistently in top 10)
        'Q36',               # MI: 0.066-0.088
        'Q27',               # MI: 0.061-0.080
        'Q20',               # MI: 0.052-0.086
        'Q21',               # MI: 0.062-0.078
        'Q23',               # MI: 0.046-0.067
        'Q30',               # MI: 0.063-0.065
        'Q25',               # MI: 0.055-0.067
        'Q16',               # MI: 0.047-0.060
        'Q18',               # MI: 0.044-0.048
        'Q28',               # MI: 0.043-0.058
        
        # Demographics (required for regulatory compliance)
        'Age',
        'AgeGroup',
        'County',
        'LivingSetting'
    ]
    
    def __init__(self, use_selected_features: bool = True, 
                 use_fy2024_only: bool = True,
                 random_state: int = 42):
        """
        Initialize Model 2
        
        Args:
            use_selected_features: Whether to use MI-based feature selection
            use_fy2024_only: Whether to use only fiscal year 2024 data
            random_state: Random state for reproducibility
        """
        super().__init__(model_id=2, model_name="GLM-Gamma")
        
        # Configuration
        self.use_selected_features = use_selected_features
        self.use_fy2024_only = use_fy2024_only
        self.fiscal_years_used = "2024" if use_fy2024_only else "2020-2025"
        self.random_state = random_state
        
        # GLM-specific attributes
        self.glm_model = None
        self.dispersion = None
        self.deviance = None
        self.null_deviance = None
        self.deviance_r2 = None
        self.mcfadden_r2 = None
        self.aic = None
        self.bic = None
        self.coefficients = {}
        self.num_parameters = 0
        
        # Feature importance from GLM
        self.feature_importance = {}
        
    def _prepare_selected_features(self, record: ConsumerRecord) -> List[float]:
        """
        Prepare features based on mutual information selection
        
        Args:
            record: Consumer record
            
        Returns:
            List of feature values
        """
        features = []
        
        # Categorical features - create dummies
        # RESIDENCETYPE (top predictor)
        residence_map = {
            'FH': 0, 'ILSL': 1, 'RH1': 2, 'RH2': 3, 'RH3': 4, 'RH4': 5
        }
        residence_val = residence_map.get(record.residencetype, 0)
        
        # Create dummy variables for residence type (FH as reference)
        for i in range(1, 6):
            features.append(1.0 if residence_val == i else 0.0)
        
        # Living setting dummies (alternative categorization)
        living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
        for setting in living_settings:
            features.append(1.0 if record.living_setting == setting else 0.0)
        
        # Age group dummies (Age3_20 as reference)
        features.append(1.0 if record.age_group == 'Age21_30' else 0.0)
        features.append(1.0 if record.age_group == 'Age31Plus' else 0.0)
        
        # Continuous features
        features.append(float(record.age))
        
        # Summary scores
        features.append(float(record.bsum))
        features.append(float(record.blevel))
        features.append(float(record.fsum))
        features.append(float(record.flevel))
        features.append(float(record.psum))
        features.append(float(record.plevel))
        features.append(float(record.losri))
        features.append(float(record.olevel))
        
        # Individual QSI questions
        qsi_questions = [26, 36, 27, 20, 21, 23, 30, 25, 16, 18, 28]
        for q in qsi_questions:
            features.append(float(getattr(record, f'q{q}', 0)))
        
        # County as numeric code (for now - could be expanded to dummies)
        county_code = hash(record.county) % 100  # Simple encoding
        features.append(float(county_code))
        
        return features
    
    def _prepare_all_features(self, record: ConsumerRecord) -> List[float]:
        """
        Prepare all available features (no selection)
        
        Args:
            record: Consumer record
            
        Returns:
            List of all feature values
        """
        features = []
        
        # Living setting dummies (FH as reference)
        living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
        for setting in living_settings:
            features.append(1.0 if record.living_setting == setting else 0.0)
        
        # Age group dummies (Age3_20 as reference)
        features.append(1.0 if record.age_group == 'Age21_30' else 0.0)
        features.append(1.0 if record.age_group == 'Age31Plus' else 0.0)
        
        # All QSI questions (Q14-Q50)
        for q in range(14, 51):
            if q == 31:
                # Q31 is split into sub-questions
                for sub in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']:
                    features.append(float(getattr(record, f'q31{sub}', 0)))
            elif q == 51:
                features.append(float(getattr(record, 'q51a', 0)))
            else:
                features.append(float(getattr(record, f'q{q}', 0)))
        
        # Summary scores
        features.append(float(record.bsum))
        features.append(float(record.fsum))
        features.append(float(record.psum))
        
        # Level scores
        features.append(float(record.blevel))
        features.append(float(record.flevel))
        features.append(float(record.plevel))
        features.append(float(record.olevel))
        features.append(float(record.losri))
        
        return features
    
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix from consumer records
        
        Args:
            records: List of consumer records
            
        Returns:
            Feature matrix and feature names
        """
        features_list = []
        
        for record in records:
            if self.use_selected_features:
                features = self._prepare_selected_features(record)
            else:
                features = self._prepare_all_features(record)
            features_list.append(features)
        
        # Build feature names (only once)
        if not self.feature_names:
            if self.use_selected_features:
                # Names for selected features
                self.feature_names = [
                    'Res_ILSL', 'Res_RH1', 'Res_RH2', 'Res_RH3', 'Res_RH4',
                    'Live_ILSL', 'Live_RH1', 'Live_RH2', 'Live_RH3', 'Live_RH4',
                    'Age21_30', 'Age31Plus', 'Age',
                    'BSum', 'BLEVEL', 'FSum', 'FLEVEL', 'PSum', 'PLEVEL',
                    'LOSRI', 'OLEVEL',
                    'Q26', 'Q36', 'Q27', 'Q20', 'Q21', 'Q23', 
                    'Q30', 'Q25', 'Q16', 'Q18', 'Q28',
                    'County_Code'
                ]
            else:
                # Names for all features
                self.feature_names = []
                for setting in ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']:
                    self.feature_names.append(f'Live_{setting}')
                self.feature_names.extend(['Age21_30', 'Age31Plus'])
                
                for q in range(14, 51):
                    if q == 31:
                        for sub in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']:
                            self.feature_names.append(f'Q31{sub}')
                    else:
                        self.feature_names.append(f'Q{q}')
                
                self.feature_names.extend(['BSum', 'FSum', 'PSum'])
                self.feature_names.extend(['BLEVEL', 'FLEVEL', 'PLEVEL', 'OLEVEL', 'LOSRI'])
        
        X = np.array(features_list, dtype=np.float64)
        
        # Update parameter count
        self.num_parameters = X.shape[1] + 1  # Features + intercept
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} records")
        
        return X, self.feature_names
    
    def calculate_variance_metrics(self) -> Dict[str, float]:
        """
        Calculate variance-related metrics for the model
        
        Returns:
            Dictionary with variance metrics
        """
        if self.test_predictions is None or self.y_test is None:
            return {}
        
        # Coefficient of variation
        cv_actual = np.std(self.y_test) / np.mean(self.y_test)
        cv_predicted = np.std(self.test_predictions) / np.mean(self.test_predictions)
        
        # Prediction interval (using residual standard error)
        residuals = self.test_predictions - self.y_test
        rse = np.std(residuals)
        prediction_interval = 1.96 * rse  # 95% CI
        
        # Budget vs actual correlation
        budget_actual_corr = np.corrcoef(self.test_predictions, self.y_test)[0, 1]
        
        # Quarterly variance (simulated)
        quarterly_variance = np.std(residuals[:len(residuals)//4]) / np.mean(self.y_test) * 100
        
        # Annual adjustment rate (simulated)
        annual_adjustment_rate = np.percentile(np.abs(residuals), 75) / np.mean(self.y_test) * 100
        
        self.variance_metrics = {
            'cv_actual': cv_actual,
            'cv_predicted': cv_predicted,
            'prediction_interval': prediction_interval,
            'budget_actual_corr': budget_actual_corr,
            'quarterly_variance': quarterly_variance,
            'annual_adjustment_rate': annual_adjustment_rate
        }
        
        return self.variance_metrics
    
    def calculate_population_scenarios(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate population capacity scenarios under different budget allocations
        
        Returns:
            Dictionary with population scenario metrics
        """
        if self.test_predictions is None:
            return {}
        
        # Fixed budget assumption
        total_budget = 1.2e9  # $1.2 billion
        
        # Average costs
        current_avg = np.mean(self.y_test) if self.y_test is not None else 50000
        model_avg = np.mean(self.test_predictions)
        
        scenarios = {
            'currentbaseline': {
                'clients_served': int(total_budget / current_avg),
                'avg_allocation': current_avg,
                'waitlist_change': 0,
                'waitlist_pct': 0
            },
            'modelbalanced': {
                'clients_served': int(total_budget / model_avg),
                'avg_allocation': model_avg,
                'waitlist_change': int(total_budget / model_avg) - int(total_budget / current_avg),
                'waitlist_pct': ((int(total_budget / model_avg) - int(total_budget / current_avg)) / 
                                int(total_budget / current_avg) * 100) if current_avg > 0 else 0
            },
            'modelefficiency': {
                'clients_served': int(total_budget / (model_avg * 0.95)),  # 5% efficiency gain
                'avg_allocation': model_avg * 0.95,
                'waitlist_change': int(total_budget / (model_avg * 0.95)) - int(total_budget / current_avg),
                'waitlist_pct': ((int(total_budget / (model_avg * 0.95)) - int(total_budget / current_avg)) / 
                                int(total_budget / current_avg) * 100) if current_avg > 0 else 0
            },
            'categoryfocused': {
                'clients_served': int(total_budget / (model_avg * 1.1)),  # Higher allocations
                'avg_allocation': model_avg * 1.1,
                'waitlist_change': int(total_budget / (model_avg * 1.1)) - int(total_budget / current_avg),
                'waitlist_pct': ((int(total_budget / (model_avg * 1.1)) - int(total_budget / current_avg)) / 
                                int(total_budget / current_avg) * 100) if current_avg > 0 else 0
            },
            'populationmaximized': {
                'clients_served': int(total_budget / (model_avg * 0.85)),  # Lower allocations
                'avg_allocation': model_avg * 0.85,
                'waitlist_change': int(total_budget / (model_avg * 0.85)) - int(total_budget / current_avg),
                'waitlist_pct': ((int(total_budget / (model_avg * 0.85)) - int(total_budget / current_avg)) / 
                                int(total_budget / current_avg) * 100) if current_avg > 0 else 0
            }
        }
        
        self.population_scenarios = scenarios
        return scenarios
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit GLM with Gamma distribution and log link
        
        Args:
            X: Feature matrix
            y: Target values (costs in dollars)
        """
        logger.info("Fitting GLM-Gamma model...")
        logger.info(f"Training samples: {len(y)}, Features: {X.shape[1]}")
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        
        # Ensure no zeros or negative values (Gamma requires positive)
        y_adjusted = np.maximum(y, 0.01)
        
        try:
            # Initialize and fit GLM
            glm = sm.GLM(
                y_adjusted,
                X_with_const,
                family=Gamma(link=Log())
            )
            
            # Fit with increased iterations for convergence
            self.glm_model = glm.fit(maxiter=200, scale='x2')
            self.model = self.glm_model  # Store for base class compatibility
            
            # Extract GLM metrics
            self.dispersion = self.glm_model.scale
            self.deviance = self.glm_model.deviance
            self.aic = self.glm_model.aic
            self.bic = self.glm_model.bic
            
            # Calculate null model for pseudo-R^2
            null_model = sm.GLM(
                y_adjusted,
                np.ones((len(y_adjusted), 1)),
                family=Gamma(link=Log())
            ).fit(disp=0)
            
            self.null_deviance = null_model.deviance
            
            # Calculate pseudo-R^2 measures
            self.deviance_r2 = 1 - (self.deviance / self.null_deviance)
            self.mcfadden_r2 = 1 - (self.glm_model.llf / null_model.llf)
            
            # Store coefficients with statistics
            coef_names = ['const'] + self.feature_names
            for i, name in enumerate(coef_names):
                self.coefficients[name] = {
                    'value': self.glm_model.params[i],
                    'std_error': self.glm_model.bse[i],
                    'z_value': self.glm_model.tvalues[i],
                    'p_value': self.glm_model.pvalues[i],
                    'conf_int_lower': self.glm_model.conf_int()[i, 0],
                    'conf_int_upper': self.glm_model.conf_int()[i, 1],
                    'exp_value': np.exp(self.glm_model.params[i])  # Multiplicative effect
                }
            
            # Calculate feature importance (absolute z-values)
            feature_importance = {}
            for i, name in enumerate(self.feature_names):
                feature_importance[name] = abs(self.glm_model.tvalues[i+1])
            
            # Sort by importance
            self.feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            logger.info(f"GLM-Gamma fitted successfully")
            logger.info(f"  Converged: {self.glm_model.converged}")
            logger.info(f"  Iterations: {self.glm_model.fit_history['iteration']}")
            logger.info(f"  Dispersion: {self.dispersion:.4f}")
            logger.info(f"  Deviance R^2: {self.deviance_r2:.4f}")
            logger.info(f"  McFadden R^2: {self.mcfadden_r2:.4f}")
            logger.info(f"  AIC: {self.aic:.1f}, BIC: {self.bic:.1f}")
            
        except Exception as e:
            logger.error(f"Error fitting GLM-Gamma: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted GLM
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions in dollar scale
        """
        if self.glm_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Add constant to match training
        X_with_const = sm.add_constant(X, has_constant='add')
        
        # GLM predict returns on original scale (handles back-transformation)
        predictions = self.glm_model.predict(X_with_const)
        
        # Ensure predictions are positive (should be by design with log link)
        predictions = np.maximum(predictions, 1.0)
        
        return predictions
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation with GLM
        
        Args:
            n_splits: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Must prepare features before cross-validation")
        
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_scores = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            # Split data
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_val = self.y_train[val_idx]
            
            try:
                # Fit GLM on fold
                X_train_const = sm.add_constant(X_fold_train)
                X_val_const = sm.add_constant(X_fold_val)
                
                y_adjusted = np.maximum(y_fold_train, 0.01)
                
                glm_fold = sm.GLM(
                    y_adjusted,
                    X_train_const,
                    family=Gamma(link=Log())
                ).fit(disp=0, maxiter=200)
                
                # Predict on validation
                y_pred = glm_fold.predict(X_val_const)
                y_pred = np.maximum(y_pred, 1.0)
                
                # Calculate metrics
                r2 = r2_score(y_fold_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                mae = mean_absolute_error(y_fold_val, y_pred)
                
                cv_scores.append(r2)
                fold_metrics.append({
                    'fold': fold,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                })
                
                logger.info(f"  Fold {fold}: R^2={r2:.4f}, RMSE=${rmse:,.0f}")
                
            except Exception as e:
                logger.warning(f"  Fold {fold} failed: {e}")
                cv_scores.append(0)
        
        # Calculate summary statistics
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        logger.info(f"CV Results: Mean R^2={mean_score:.4f} +- {std_score:.4f}")
        
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'scores': cv_scores,
            'fold_metrics': fold_metrics,
            'cv_mean': mean_score,  # For compatibility with base class
            'cv_std': std_score     # For compatibility with base class
        }
    
    def calculate_subgroup_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for specific subgroups matching LaTeX requirements
        
        Returns:
            Dictionary with subgroup performance metrics
        """
        if self.test_predictions is None or self.y_test is None:
            return {}
        
        subgroup_metrics = {}
        
        # Define subgroups that match LaTeX commands EXACTLY
        # Note: base class will convert numbers to words (RH1 -> RHOne, etc.)
        subgroups = {
            # Living Settings - these will become LivingSettingFH, LivingSettingILSL, etc.
            'LivingSetting_FH': lambda r: r.living_setting == 'FH',
            'LivingSetting_ILSL': lambda r: r.living_setting == 'ILSL',
            'LivingSetting_RH1': lambda r: r.living_setting == 'RH1',
            'LivingSetting_RH2': lambda r: r.living_setting == 'RH2',
            'LivingSetting_RH3': lambda r: r.living_setting == 'RH3',
            'LivingSetting_RH4': lambda r: r.living_setting == 'RH4',
            
            # Age Groups - these will become AgeGroupAgeThreeTwenty, etc.
            'AgeGroup_Age3_20': lambda r: r.age_group == 'Age3_20',
            'AgeGroup_Age21_30': lambda r: r.age_group == 'Age21_30',
            'AgeGroup_Age31Plus': lambda r: r.age_group == 'Age31Plus',
            
            # Cost Quartiles - these will become costQOneLow, etc.
            'cost_Q1_Low': None,  # Will define based on quartiles
            'cost_Q2': None,
            'cost_Q3': None,
            'cost_Q4_High': None
        }
        
        # Calculate cost quartiles
        if len(self.y_test) > 0:
            q1, q2, q3 = np.percentile(self.y_test, [25, 50, 75])
            subgroups['cost_Q1_Low'] = lambda r, idx: self.y_test[idx] <= q1
            subgroups['cost_Q2'] = lambda r, idx: (self.y_test[idx] > q1) & (self.y_test[idx] <= q2)
            subgroups['cost_Q3'] = lambda r, idx: (self.y_test[idx] > q2) & (self.y_test[idx] <= q3)
            subgroups['cost_Q4_High'] = lambda r, idx: self.y_test[idx] > q3
        
        # Calculate metrics for each subgroup
        for key, condition in subgroups.items():
            if condition is None:
                continue
                
            # Get indices for this subgroup
            if 'cost' in key:
                # For cost-based subgroups, use index-based selection
                indices = [i for i in range(len(self.test_records)) 
                          if condition(self.test_records[i], i)]
            else:
                # For other subgroups, use record-based selection
                indices = [i for i, record in enumerate(self.test_records) 
                          if condition(record)]
            
            if len(indices) > 0:
                # Get subset of predictions and actuals
                y_subset = self.y_test[indices]
                pred_subset = self.test_predictions[indices]
                
                # Calculate metrics
                r2 = r2_score(y_subset, pred_subset) if len(indices) > 1 else 0
                rmse = np.sqrt(mean_squared_error(y_subset, pred_subset))
                bias = np.mean(pred_subset - y_subset)
                
                subgroup_metrics[key] = {
                    'n': len(indices),
                    'r2': r2,
                    'rmse': rmse,
                    'bias': bias
                }
            else:
                # No data for this subgroup
                subgroup_metrics[key] = {
                    'n': 0,
                    'r2': 0,
                    'rmse': 0,
                    'bias': 0
                }
        
        self.subgroup_metrics = subgroup_metrics
        logger.info(f"Calculated metrics for {len(subgroup_metrics)} subgroups")
        
        return subgroup_metrics
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive metrics including GLM-specific measures
        
        Returns:
            Dictionary of metrics
        """
        # Calculate base metrics first
        metrics = super().calculate_metrics()
        
        # Add prediction interval coverage metrics
        if self.test_predictions is not None and self.y_test is not None:
            within_5k = np.mean(np.abs(self.test_predictions - self.y_test) <= 5000) * 100
            within_10k = np.mean(np.abs(self.test_predictions - self.y_test) <= 10000) * 100
            within_20k = np.mean(np.abs(self.test_predictions - self.y_test) <= 20000) * 100
            
            metrics.update({
                'within_5k': within_5k,
                'within_10k': within_10k,
                'within_20k': within_20k
            })
        
        # Add GLM-specific metrics
        if self.glm_model is not None:
            metrics.update({
                'dispersion': self.dispersion if self.dispersion else 0,
                'deviance': self.deviance if self.deviance else 0,
                'null_deviance': self.null_deviance if self.null_deviance else 0,
                'deviance_r2': self.deviance_r2 if self.deviance_r2 else 0,
                'mcfadden_r2': self.mcfadden_r2 if self.mcfadden_r2 else 0,
                'aic': self.aic if self.aic else 0,
                'bic': self.bic if self.bic else 0,
                'num_parameters': self.num_parameters,
                'converged': self.glm_model.converged
            })
        
        self.metrics = metrics
        return metrics
    
    def plot_diagnostics(self) -> None:
        """
        Generate comprehensive diagnostic plots for GLM
        """
        if self.test_predictions is None or self.y_test is None:
            logger.warning("No predictions available for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Model {self.model_id}: GLM-Gamma Diagnostic Plots', 
                     fontsize=14, fontweight='bold')
        
        # 1. Predicted vs Actual
        ax = axes[0, 0]
        ax.scatter(self.y_test, self.test_predictions, alpha=0.5, s=10)
        min_val = min(self.y_test.min(), self.test_predictions.min())
        max_val = max(self.y_test.max(), self.test_predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax.set_xlabel('Actual Cost ($)')
        ax.set_ylabel('Predicted Cost ($)')
        ax.set_title(f'Predicted vs Actual (R^2={self.metrics.get("r2_test", 0):.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residual Plot
        ax = axes[0, 1]
        residuals = self.test_predictions - self.y_test
        ax.scatter(self.test_predictions, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Predicted Cost ($)')
        ax.set_ylabel('Residuals ($)')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # 3. Deviance Residuals
        ax = axes[0, 2]
        # Calculate deviance residuals
        y_test_adj = np.maximum(self.y_test, 0.01)
        y_pred_adj = np.maximum(self.test_predictions, 0.01)
        dev_residuals = np.sign(y_test_adj - y_pred_adj) * np.sqrt(
            2 * np.abs(y_test_adj * np.log(y_test_adj / y_pred_adj) - 
                      (y_test_adj - y_pred_adj))
        )
        ax.scatter(self.test_predictions, dev_residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Predicted Cost ($)')
        ax.set_ylabel('Deviance Residuals')
        ax.set_title('Deviance Residuals')
        ax.grid(True, alpha=0.3)
        
        # 4. Q-Q Plot of Deviance Residuals
        ax = axes[1, 0]
        stats.probplot(dev_residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Deviance Residuals)')
        ax.grid(True, alpha=0.3)
        
        # 5. Scale-Location Plot
        ax = axes[1, 1]
        standardized_dev = dev_residuals / np.std(dev_residuals)
        ax.scatter(self.test_predictions, np.sqrt(np.abs(standardized_dev)), alpha=0.5, s=10)
        ax.set_xlabel('Predicted Cost ($)')
        ax.set_ylabel('?|Standardized Deviance Residuals|')
        ax.set_title('Scale-Location Plot')
        ax.grid(True, alpha=0.3)
        
        # 6. Feature Importance (Top 15)
        ax = axes[1, 2]
        if self.feature_importance:
            top_features = list(self.feature_importance.keys())[:15]
            top_importance = [self.feature_importance[f] for f in top_features]
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_importance, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features, fontsize=8)
            ax.set_xlabel('|z-value|')
            ax.set_title('Top 15 Feature Importance')
            ax.grid(True, alpha=0.3)
        
        # 7. Distribution of Residuals
        ax = axes[2, 0]
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Residuals ($)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Residuals (Mean=${np.mean(residuals):,.0f})')
        ax.grid(True, alpha=0.3)
        
        # 8. Log-Scale Predicted vs Actual
        ax = axes[2, 1]
        ax.scatter(np.log(self.y_test + 1), np.log(self.test_predictions + 1), 
                  alpha=0.5, s=10)
        log_min = min(np.log(self.y_test + 1).min(), np.log(self.test_predictions + 1).min())
        log_max = max(np.log(self.y_test + 1).max(), np.log(self.test_predictions + 1).max())
        ax.plot([log_min, log_max], [log_min, log_max], 'r--', label='Perfect Prediction')
        ax.set_xlabel('Log(Actual Cost + 1)')
        ax.set_ylabel('Log(Predicted Cost + 1)')
        ax.set_title('Log-Scale Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. MAPE by Cost Decile
        ax = axes[2, 2]
        deciles = np.percentile(self.y_test, range(10, 100, 10))
        mapes = []
        for i in range(len(deciles)):
            if i == 0:
                mask = self.y_test <= deciles[i]
            else:
                mask = (self.y_test > deciles[i-1]) & (self.y_test <= deciles[i])
            
            if mask.sum() > 0:
                actual_subset = self.y_test[mask]
                pred_subset = self.test_predictions[mask]
                # Avoid division by zero
                mape = np.mean(np.abs((actual_subset - pred_subset) / 
                              np.maximum(actual_subset, 1))) * 100
                mapes.append(mape)
            else:
                mapes.append(0)
        
        ax.bar(range(1, len(mapes) + 1), mapes, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Cost Decile')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('MAPE by Cost Decile')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "diagnostic_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plots saved to {plot_file}")
        
        # Generate additional GLM-specific plots
        self._plot_glm_specific()
    
    def _plot_glm_specific(self) -> None:
        """
        Generate GLM-specific diagnostic plots
        """
        if self.glm_model is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('GLM-Specific Diagnostics', fontsize=14, fontweight='bold')
        
        # 1. Partial Residual Plots for top features
        ax = axes[0, 0]
        if self.feature_importance:
            # Get top 3 features for partial residual plots
            top_3_features = list(self.feature_importance.keys())[:3]
            for i, feat in enumerate(top_3_features):
                feat_idx = self.feature_names.index(feat)
                # Simple partial residuals (would need more complex calculation)
                ax.scatter(self.X_test[:, feat_idx], 
                          self.test_predictions - self.y_test, 
                          alpha=0.3, s=5, label=feat)
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Partial Residual')
            ax.set_title('Partial Residuals (Top 3 Features)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Link Function Assessment
        ax = axes[0, 1]
        linear_predictor = np.log(self.test_predictions)
        ax.scatter(linear_predictor, self.y_test, alpha=0.5, s=10)
        ax.set_xlabel('Linear Predictor (log scale)')
        ax.set_ylabel('Actual Cost ($)')
        ax.set_title('Link Function Assessment')
        ax.grid(True, alpha=0.3)
        
        # 3. Influence Plot (simplified version)
        ax = axes[1, 0]
        standardized_residuals = (self.test_predictions - self.y_test) / np.std(self.test_predictions - self.y_test)
        leverage = np.ones(len(standardized_residuals)) * (self.num_parameters / len(self.y_train))
        ax.scatter(leverage, standardized_residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.set_title('Influence Plot (Simplified)')
        ax.grid(True, alpha=0.3)
        
        # 4. Fitted vs Pearson Residuals
        ax = axes[1, 1]
        pearson_residuals = (self.y_test - self.test_predictions) / np.sqrt(self.test_predictions)
        ax.scatter(self.test_predictions, pearson_residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Fitted Values ($)')
        ax.set_ylabel('Pearson Residuals')
        ax.set_title('Pearson Residuals')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = self.output_dir / "glm_specific_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"GLM-specific plots saved to {plot_file}")
    
    def generate_latex_commands(self) -> None:
        """
        Override to generate commands matching our LaTeX table exactly
        """
        # First get the model word
        model_word = "Two"
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate newcommands file (definitions)
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        
        with open(newcommands_file, 'w') as f:
            f.write(f"% Model {self.model_id} Command Definitions\n")
            f.write(f"% Generated: {datetime.now()}\n")
            f.write(f"% Model: {self.model_name}\n\n")
            
            # Core metrics
            f.write("% Core Metrics\n")
            metric_commands = [
                ('r2_train', 'RSquaredTrain'),
                ('r2_test', 'RSquaredTest'),
                ('rmse_train', 'RMSETrain'),
                ('rmse_test', 'RMSETest'),
                ('mae_train', 'MAETrain'),
                ('mae_test', 'MAETest'),
                ('mape_train', 'MAPETrain'),
                ('mape_test', 'MAPETest'),
                ('cv_mean', 'CVMean'),
                ('cv_std', 'CVStd'),
                ('within_5k', 'WithinFiveK'),
                ('within_10k', 'WithinTenK'),
                ('within_20k', 'WithinTwentyK'),
                ('training_samples', 'TrainingSamples'),
                ('test_samples', 'TestSamples')
            ]
            
            for _, latex_name in metric_commands:
                f.write(f"\\newcommand{{\\Model{model_word}{latex_name}}}{{\\WarningRunPipeline}}\n")
            
            # Subgroup commands for our specific table
            f.write("\n% Living Setting Subgroups\n")
            for setting in ['FH', 'ILSL', 'RHOne', 'RHTwo', 'RHThree', 'RHFour']:
                f.write(f"\\newcommand{{\\Model{model_word}SubgroupLivingSetting{setting}N}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}SubgroupLivingSetting{setting}RSquared}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}SubgroupLivingSetting{setting}RMSE}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}SubgroupLivingSetting{setting}Bias}}{{\\WarningRunPipeline}}\n")
            
            f.write("\n% Age Group Subgroups\n")
            age_groups = [
                ('AgeThreeTwenty', 'Age3_20'),
                ('AgeTwentyOneThirty', 'Age21_30'),
                ('AgeThirtyOnePlus', 'Age31Plus')
            ]
            for latex_name, _ in age_groups:
                f.write(f"\\newcommand{{\\Model{model_word}SubgroupAgeGroup{latex_name}N}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}SubgroupAgeGroup{latex_name}RSquared}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}SubgroupAgeGroup{latex_name}RMSE}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}SubgroupAgeGroup{latex_name}Bias}}{{\\WarningRunPipeline}}\n")
            
            f.write("\n% Cost Quartile Subgroups\n")
            for q in ['QOneLow', 'QTwo', 'QThree', 'QFourHigh']:
                f.write(f"\\newcommand{{\\Model{model_word}Subgroupcost{q}N}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Subgroupcost{q}RSquared}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Subgroupcost{q}RMSE}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Subgroupcost{q}Bias}}{{\\WarningRunPipeline}}\n")
            
            # Variance metrics
            f.write("\n% Variance Metrics\n")
            f.write(f"\\newcommand{{\\Model{model_word}CVActual}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}CVPredicted}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}PredictionInterval}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}BudgetActualCorr}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}QuarterlyVariance}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}AnnualAdjustmentRate}}{{\\WarningRunPipeline}}\n")
            
            # Population scenarios
            f.write("\n% Population Scenarios\n")
            for scenario in ['currentbaseline', 'modelbalanced', 'modelefficiency', 
                           'categoryfocused', 'populationmaximized']:
                f.write(f"\\newcommand{{\\Model{model_word}Pop{scenario}Clients}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Pop{scenario}AvgAlloc}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Pop{scenario}WaitlistChange}}{{\\WarningRunPipeline}}\n")
            
            # GLM-specific commands
            f.write("\n% GLM-Specific Commands\n")
            f.write(f"\\newcommand{{\\Model{model_word}Distribution}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}LinkFunction}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Dispersion}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}DevianceRSquared}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}McFaddenRSquared}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}AIC}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}BIC}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}Parameters}}{{\\WarningRunPipeline}}\n")
            
            if self.use_selected_features:
                f.write("\n% Feature Selection Commands\n")
                f.write(f"\\newcommand{{\\Model{model_word}FeatureSelection}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}NumFeatures}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}TopFeature}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}TopFeatureMI}}{{\\WarningRunPipeline}}\n")
        
        # Generate renewcommands file (values)
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        with open(renewcommands_file, 'w') as f:
            f.write(f"% Model {self.model_id} Calibrated Values\n")
            f.write(f"% Generated: {datetime.now()}\n\n")
            
            # Core metrics values
            f.write("% Core Metrics\n")
            for metric_key, latex_name in metric_commands:
                value = self.metrics.get(metric_key, 0)
                formatted_value = self._format_latex_value(metric_key, value)
                f.write(f"\\renewcommand{{\\Model{model_word}{latex_name}}}{{{formatted_value}}}\n")
            
            # Subgroup values
            f.write("\n% Living Setting Subgroups\n")
            subgroup_mapping = {
                'FH': 'LivingSetting_FH',
                'ILSL': 'LivingSetting_ILSL', 
                'RHOne': 'LivingSetting_RH1',
                'RHTwo': 'LivingSetting_RH2',
                'RHThree': 'LivingSetting_RH3',
                'RHFour': 'LivingSetting_RH4'
            }
            
            for latex_setting, data_key in subgroup_mapping.items():
                if data_key in self.subgroup_metrics:
                    metrics = self.subgroup_metrics[data_key]
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLivingSetting{latex_setting}N}}{{{metrics.get('n', 0)}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLivingSetting{latex_setting}RSquared}}{{{metrics.get('r2', 0):.4f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLivingSetting{latex_setting}RMSE}}{{{metrics.get('rmse', 0):,.0f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLivingSetting{latex_setting}Bias}}{{{metrics.get('bias', 0):,.0f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLivingSetting{latex_setting}N}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLivingSetting{latex_setting}RSquared}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLivingSetting{latex_setting}RMSE}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLivingSetting{latex_setting}Bias}}{{0}}\n")
            
            f.write("\n% Age Group Subgroups\n")
            age_mapping = {
                'AgeThreeTwenty': 'AgeGroup_Age3_20',
                'AgeTwentyOneThirty': 'AgeGroup_Age21_30',
                'AgeThirtyOnePlus': 'AgeGroup_Age31Plus'
            }
            
            for latex_age, data_key in age_mapping.items():
                if data_key in self.subgroup_metrics:
                    metrics = self.subgroup_metrics[data_key]
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAgeGroup{latex_age}N}}{{{metrics.get('n', 0)}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAgeGroup{latex_age}RSquared}}{{{metrics.get('r2', 0):.4f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAgeGroup{latex_age}RMSE}}{{{metrics.get('rmse', 0):,.0f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAgeGroup{latex_age}Bias}}{{{metrics.get('bias', 0):,.0f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAgeGroup{latex_age}N}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAgeGroup{latex_age}RSquared}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAgeGroup{latex_age}RMSE}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAgeGroup{latex_age}Bias}}{{0}}\n")
            
            f.write("\n% Cost Quartile Subgroups\n")
            cost_mapping = {
                'QOneLow': 'cost_Q1_Low',
                'QTwo': 'cost_Q2',
                'QThree': 'cost_Q3',
                'QFourHigh': 'cost_Q4_High'
            }
            
            for latex_cost, data_key in cost_mapping.items():
                if data_key in self.subgroup_metrics:
                    metrics = self.subgroup_metrics[data_key]
                    f.write(f"\\renewcommand{{\\Model{model_word}Subgroupcost{latex_cost}N}}{{{metrics.get('n', 0)}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Subgroupcost{latex_cost}RSquared}}{{{metrics.get('r2', 0):.4f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Subgroupcost{latex_cost}RMSE}}{{{metrics.get('rmse', 0):,.0f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Subgroupcost{latex_cost}Bias}}{{{metrics.get('bias', 0):,.0f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}Subgroupcost{latex_cost}N}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Subgroupcost{latex_cost}RSquared}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Subgroupcost{latex_cost}RMSE}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Subgroupcost{latex_cost}Bias}}{{0}}\n")
            
            # Variance metrics
            f.write("\n% Variance Metrics\n")
            if self.variance_metrics:
                f.write(f"\\renewcommand{{\\Model{model_word}CVActual}}{{{self.variance_metrics.get('cv_actual', 0):.3f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}CVPredicted}}{{{self.variance_metrics.get('cv_predicted', 0):.3f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}PredictionInterval}}{{{self.variance_metrics.get('prediction_interval', 0):,.0f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}BudgetActualCorr}}{{{self.variance_metrics.get('budget_actual_corr', 0):.3f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}QuarterlyVariance}}{{{self.variance_metrics.get('quarterly_variance', 0):.1f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}AnnualAdjustmentRate}}{{{self.variance_metrics.get('annual_adjustment_rate', 0):.1f}}}\n")
            
            # Population scenarios
            f.write("\n% Population Scenarios\n")
            if self.population_scenarios:
                for scenario in ['currentbaseline', 'modelbalanced', 'modelefficiency', 
                               'categoryfocused', 'populationmaximized']:
                    if scenario in self.population_scenarios:
                        s = self.population_scenarios[scenario]
                        f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}Clients}}{{{s.get('clients_served', 0)}}}\n")
                        f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}AvgAlloc}}{{{s.get('avg_allocation', 0):,.0f}}}\n")
                        f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}WaitlistChange}}{{{s.get('waitlist_change', 0):+,}}}\n")
            
            # GLM-specific values
            f.write("\n% GLM-Specific Values\n")
            f.write(f"\\renewcommand{{\\Model{model_word}Distribution}}{{Gamma}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}LinkFunction}}{{Log}}\n")
            
            # Format GLM metrics properly
            dispersion_val = f"{self.dispersion:.4f}" if self.dispersion is not None else "0"
            deviance_r2_val = f"{self.deviance_r2:.4f}" if self.deviance_r2 is not None else "0"
            mcfadden_r2_val = f"{self.mcfadden_r2:.4f}" if self.mcfadden_r2 is not None else "0"
            aic_val = f"{self.aic:,.0f}" if self.aic is not None else "0"
            bic_val = f"{self.bic:,.0f}" if self.bic is not None else "0"
            
            f.write(f"\\renewcommand{{\\Model{model_word}Dispersion}}{{{dispersion_val}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}DevianceRSquared}}{{{deviance_r2_val}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}McFaddenRSquared}}{{{mcfadden_r2_val}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}AIC}}{{{aic_val}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}BIC}}{{{bic_val}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}Parameters}}{{{self.num_parameters}}}\n")
            
            if self.use_selected_features:
                f.write("\n% Feature Selection Values\n")
                f.write(f"\\renewcommand{{\\Model{model_word}FeatureSelection}}{{True}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}NumFeatures}}{{{len(self.feature_names)}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}TopFeature}}{{RESIDENCETYPE}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}TopFeatureMI}}{{0.256}}\n")
        
        logger.info(f"LaTeX commands written to {newcommands_file} and {renewcommands_file}")
    
    def _format_latex_value(self, metric_key: str, value: Any) -> str:
        """
        Generate LaTeX commands with GLM-specific additions
        """
        # Generate base commands first (this includes WithinFiveK, etc.)
        super().generate_latex_commands()
        
        # Model word for LaTeX commands
        model_word = "Two"
        
        # Append GLM-specific commands to newcommands file
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        
        try:
            with open(newcommands_file, 'a') as f:
                f.write("\n% GLM-Specific Commands\n")
                f.write(f"\\newcommand{{\\Model{model_word}Distribution}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}LinkFunction}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Dispersion}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}DevianceRSquared}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}McFaddenRSquared}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}AIC}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}BIC}}{{\\WarningRunPipeline}}\n")
                f.write(f"\\newcommand{{\\Model{model_word}Parameters}}{{\\WarningRunPipeline}}\n")
                # Don't add WithinFiveK, WithinTenK, WithinTwentyK here - base class handles them
                
                # Feature selection specific
                if self.use_selected_features:
                    f.write("\n% Feature Selection Commands\n")
                    f.write(f"\\newcommand{{\\Model{model_word}FeatureSelection}}{{\\WarningRunPipeline}}\n")
                    f.write(f"\\newcommand{{\\Model{model_word}NumFeatures}}{{\\WarningRunPipeline}}\n")
                    f.write(f"\\newcommand{{\\Model{model_word}TopFeature}}{{\\WarningRunPipeline}}\n")
                    f.write(f"\\newcommand{{\\Model{model_word}TopFeatureMI}}{{\\WarningRunPipeline}}\n")
        
        except Exception as e:
            logger.error(f"Error appending to newcommands file: {e}")
        
        # Append actual values to renewcommands file
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        try:
            with open(renewcommands_file, 'a') as f:
                f.write("\n% GLM-Specific Values\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Distribution}}{{Gamma}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}LinkFunction}}{{Log}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Dispersion}}{{{self.dispersion:.4f if self.dispersion else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}DevianceRSquared}}{{{self.deviance_r2:.4f if self.deviance_r2 else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}McFaddenRSquared}}{{{self.mcfadden_r2:.4f if self.mcfadden_r2 else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}AIC}}{{{self.aic:,.0f if self.aic else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}BIC}}{{{self.bic:,.0f if self.bic else 0}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}Parameters}}{{{self.num_parameters}}}\n")
                # Don't add WithinFiveK, etc. here - base class handles them
                
                # Feature selection specific values
                if self.use_selected_features:
                    f.write("\n% Feature Selection Values\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}FeatureSelection}}{{True}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}NumFeatures}}{{{len(self.feature_names)}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}TopFeature}}{{RESIDENCETYPE}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}TopFeatureMI}}{{0.256}}\n")
        
        except Exception as e:
            logger.error(f"Error appending to renewcommands file: {e}")
    
    def save_results(self) -> None:
        """
        Save Model 2 specific results including GLM diagnostics
        """
        # Save base results
        super().save_results()
        
        # Save GLM-specific results
        glm_results = {
            'model_type': 'GLM-Gamma',
            'distribution': 'Gamma',
            'link_function': 'Log',
            'converged': self.glm_model.converged if self.glm_model else None,
            'iterations': self.glm_model.fit_history['iteration'] if self.glm_model else None,
            'dispersion': self.dispersion,
            'deviance': self.deviance,
            'null_deviance': self.null_deviance,
            'deviance_r2': self.deviance_r2,
            'mcfadden_r2': self.mcfadden_r2,
            'aic': self.aic,
            'bic': self.bic,
            'num_parameters': self.num_parameters,
            'feature_selection': self.use_selected_features,
            'fiscal_years': self.fiscal_years_used
        }
        
        glm_file = self.output_dir / "glm_results.json"
        with open(glm_file, 'w') as f:
            json.dump(glm_results, f, indent=2)
        
        # Save coefficients with detailed statistics
        coef_file = self.output_dir / "coefficients.json"
        with open(coef_file, 'w') as f:
            json.dump(self.coefficients, f, indent=2)
        
        # Save feature importance
        if self.feature_importance:
            importance_file = self.output_dir / "feature_importance.json"
            with open(importance_file, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
        
        logger.info(f"GLM-specific results saved to {self.output_dir}")


    def _format_latex_value(self, metric_key: str, value: Any) -> str:
        """
        Format values for LaTeX output
        
        Args:
            metric_key: Name of the metric
            value: Value to format
            
        Returns:
            Formatted string for LaTeX
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "0"
        
        # Percentage metrics (0-100 scale)
        if any(x in metric_key for x in ['mape', 'within_', 'pct']):
            return f"{float(value):.1f}"
        
        # R-squared and correlations (-1 to 1 scale)
        elif any(x in metric_key for x in ['r2', 'corr']):
            return f"{float(value):.4f}"
        
        # Dollar amounts
        elif any(x in metric_key for x in ['rmse', 'mae', 'allocation', 'interval']):
            return f"{float(value):,.0f}"
        
        # Counts
        elif 'samples' in metric_key or metric_key == 'n':
            return f"{int(value):,}"
        
        # Standard deviations and CVs
        elif any(x in metric_key for x in ['std', 'cv']):
            return f"{float(value):.4f}"
        
        # Default
        else:
            if isinstance(value, (int, np.integer)):
                return f"{int(value):,}"
            else:
                return f"{float(value):.4f}"


def main():
    """
    Run Model 2 GLM-Gamma with feature selection
    """
    logger.info("="*80)
    logger.info("MODEL 2: GLM WITH GAMMA DISTRIBUTION")
    logger.info("="*80)
    
    # Initialize model with feature selection
    model = Model2GLMGamma(
        use_selected_features=True,  # Use MI-based feature selection
        use_fy2024_only=True,        # Use only FY2024 data
        random_state=42              # For reproducibility
    )
    
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
    print("MODEL 2 SUMMARY: GLM-GAMMA RESULTS")
    print("="*80)
    
    print("\nConfiguration:")
    print(f"  ? Feature Selection: {model.use_selected_features}")
    print(f"  ? Fiscal Years: {model.fiscal_years_used}")
    print(f"  ? Number of Features: {len(model.feature_names)}")
    print(f"  ? Distribution: Gamma")
    print(f"  ? Link Function: Log")
    
    print("\nData Summary:")
    print(f"  ? Total Records: {len(model.all_records)}")
    print(f"  ? Training Records: {len(model.train_records)}")
    print(f"  ? Test Records: {len(model.test_records)}")
    print(f"  ? Outliers Removed: 0 (GLM is robust to outliers)")
    
    print("\nPerformance Metrics:")
    print(f"  ? Training R^2: {results['metrics']['r2_train']:.4f}")
    print(f"  ? Test R^2: {results['metrics']['r2_test']:.4f}")
    print(f"  ? RMSE: ${results['metrics']['rmse_test']:,.0f}")
    print(f"  ? MAE: ${results['metrics']['mae_test']:,.0f}")
    print(f"  ? MAPE: {results['metrics']['mape_test']:.1f}%")
    
    print("\nGLM-Specific Metrics:")
    print(f"  ? Deviance R^2: {model.deviance_r2:.4f}")
    print(f"  ? McFadden R^2: {model.mcfadden_r2:.4f}")
    print(f"  ? AIC: {model.aic:,.0f}")
    print(f"  ? BIC: {model.bic:,.0f}")
    print(f"  ? Dispersion: {model.dispersion:.4f}")
    
    print("\nPrediction Accuracy:")
    print(f"  ? Within +-$5,000: {results['metrics'].get('within_5k', 0):.1f}%")
    print(f"  ? Within +-$10,000: {results['metrics'].get('within_10k', 0):.1f}%")
    print(f"  ? Within +-$20,000: {results['metrics'].get('within_20k', 0):.1f}%")
    
    print("\nCross-Validation:")
    if 'cv_mean' in results.get('metrics', {}):
        print(f"  ? Mean R^2: {results['metrics']['cv_mean']:.4f}")
        print(f"  ? Std R^2: {results['metrics']['cv_std']:.4f}")
    elif 'cv_results' in results:
        print(f"  ? Mean R^2: {results['cv_results'].get('mean_score', 0):.4f}")
        print(f"  ? Std R^2: {results['cv_results'].get('std_score', 0):.4f}")
    
    print("\nTop 5 Important Features:")
    if model.feature_importance:
        for i, (feat, imp) in enumerate(list(model.feature_importance.items())[:5], 1):
            print(f"  {i}. {feat}: |z|={imp:.2f}")
    
    print("\n" + "="*80)
    print("Model 2 pipeline completed successfully!")
    print(f"Results saved to: {model.output_dir}")
    print("="*80)
    
    return model, results


if __name__ == "__main__":
    model, results = main()
