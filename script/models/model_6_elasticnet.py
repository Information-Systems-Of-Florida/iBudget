"""
Model 6: Elastic Net Regression
Combined L1 and L2 regularization for feature selection and stability
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import logging
import warnings
warnings.filterwarnings('ignore')

from base_model import BaseiBudgetModel, ConsumerRecord

logger = logging.getLogger(__name__)

class Model6ElasticNet(BaseiBudgetModel):
    """
    Model 6: Elastic Net Regression
    
    Key features:
    - Combined L1 (Lasso) and L2 (Ridge) regularization
    - Automatic feature selection through L1 penalty
    - Stability through L2 penalty
    - Cross-validated parameter selection for alpha and l1_ratio
    """
    
    def __init__(self):
        """Initialize Model 6"""
        super().__init__(model_id=6, model_name="Elastic Net")
        
        # Model-specific parameters
        self.model = None
        self.scaler = StandardScaler()
        
        # Elastic Net parameters
        self.alpha = None
        self.l1_ratio = None
        self.l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        self.n_alphas = 100
        
        # Feature selection tracking
        self.coefficients = {}
        self.selected_features = []
        self.dropped_features = []
        self.n_features_selected = 0
        self.n_features_dropped = 0
        
        # Store scaled data for analysis
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        # Cross-validation path
        self.cv_alphas = None
        self.cv_scores = None
    
    def load_data(self, fiscal_year_start: int = 2024, fiscal_year_end: int = 2024) -> List[ConsumerRecord]:
        """
        Override to load ALL data, not just usable records
        
        Args:
            fiscal_year_start: First fiscal year to load
            fiscal_year_end: Last fiscal year to load
            
        Returns:
            List of ALL consumer records with positive costs
        """
        data_dir = Path("data/cached")
        all_records = []
        
        for fiscal_year in range(fiscal_year_start, fiscal_year_end + 1):
            pickle_file = data_dir / f"fy{fiscal_year}.pkl"
            
            if not pickle_file.exists():
                logger.warning(f"Data file not found: {pickle_file}")
                continue
            
            # Load pickle file
            with open(pickle_file, 'rb') as f:
                data_dict = pickle.load(f)
            
            records_data = data_dict.get('data', [])
            logger.info(f"Loaded {len(records_data)} records from FY{fiscal_year}")
            
            # Convert to ConsumerRecord objects
            for row_dict in records_data:
                # Helper function to safely get values
                def get_val(key, default=0, type_func=float):
                    val = row_dict.get(key, default)
                    if val is None or val == '' or (isinstance(val, str) and val.lower() in ['nan', 'none']):
                        return default
                    try:
                        return type_func(val)
                    except (ValueError, TypeError):
                        return default
                
                # Parse TotalCost
                total_cost_str = str(row_dict.get('TotalCost', '0'))
                total_cost_str = total_cost_str.replace('$', '').replace(',', '').strip()
                try:
                    total_cost = float(total_cost_str)
                except (ValueError, TypeError):
                    total_cost = 0
                
                # Skip records with zero or negative cost
                if total_cost <= 0:
                    continue
                
                # Create record with all necessary fields
                record = ConsumerRecord(
                    case_no=int(get_val('CASENO', 0, int)),
                    fiscal_year=int(get_val('FiscalYear', 2020, int)),
                    age=int(get_val('Age', 0, int)),
                    gender=str(row_dict.get('GENDER', 'U')),
                    race=str(row_dict.get('RACE', 'Unknown')),
                    ethnicity=str(row_dict.get('Ethnicity', 'Unknown')),
                    county=str(row_dict.get('County', 'Unknown')),
                    district=str(row_dict.get('District', 'Unknown')),
                    region=str(row_dict.get('Region', 'Unknown')),
                    primary_diagnosis=str(row_dict.get('PrimaryDiagnosis', 'Unknown')),
                    secondary_diagnosis=str(row_dict.get('SecondaryDiagnosis', 'Unknown')),
                    residencetype=str(row_dict.get('RESIDENCETYPE', 'Unknown')),
                    living_setting=str(row_dict.get('LivingSetting', 'FH')),
                    age_group=str(row_dict.get('AgeGroup', 'Age3_20')),
                    
                    # QSI Questions  
                    q14=get_val('Q14', 0, float),
                    q15=get_val('Q15', 0, float),
                    q16=get_val('Q16', 0, float),
                    q17=get_val('Q17', 0, float),
                    q18=get_val('Q18', 0, float),
                    q19=get_val('Q19', 0, float),
                    q20=get_val('Q20', 0, float),
                    q21=get_val('Q21', 0, float),
                    q22=get_val('Q22', 0, float),
                    q23=get_val('Q23', 0, float),
                    q24=get_val('Q24', 0, float),
                    q25=get_val('Q25', 0, float),
                    q26=get_val('Q26', 0, float),
                    q27=get_val('Q27', 0, float),
                    q28=get_val('Q28', 0, float),
                    q29=get_val('Q29', 0, float),
                    q30=get_val('Q30', 0, float),
                    q32=get_val('Q32', 0, float),
                    q33=get_val('Q33', 0, float),
                    q34=get_val('Q34', 0, float),
                    q35=get_val('Q35', 0, float),
                    q36=get_val('Q36', 0, float),
                    q37=get_val('Q37', 0, float),
                    q38=get_val('Q38', 0, float),
                    q39=get_val('Q39', 0, float),
                    q40=get_val('Q40', 0, float),
                    q41=get_val('Q41', 0, float),
                    q42=get_val('Q42', 0, float),
                    q43=get_val('Q43', 0, float),
                    q44=get_val('Q44', 0, float),
                    q45=get_val('Q45', 0, float),
                    q46=get_val('Q46', 0, float),
                    q47=get_val('Q47', 0, float),
                    q48=get_val('Q48', 0, float),
                    q49=get_val('Q49', 0, float),
                    q50=get_val('Q50', 0, float),
                    
                    # Summary scores
                    fsum=get_val('FSum', 0, float),
                    bsum=get_val('BSum', 0, float),
                    psum=get_val('PSum', 0, float),
                    
                    # Cost and usage
                    total_cost=total_cost,
                    service_days=int(get_val('ServiceDays', 0, int)),
                    unique_procedures=int(get_val('UniqueProcedures', 0, int)),
                    unique_providers=int(get_val('UniqueProviders', 0, int)),
                    
                    # Data quality - NOT USED FOR FILTERING
                    days_in_system=int(get_val('DaysInSystem', 0, int)),
                    usable=bool(row_dict.get('Usable', 1))  # Default to True
                )
                
                # Store developmental disability separately (not part of ConsumerRecord)
                record.developmental_disability = str(row_dict.get('DevelopmentalDisability', ''))
                
                all_records.append(record)
        
        logger.info(f"Total records loaded: {len(all_records)}")
        logger.info(f"Using ALL {len(all_records)} records (100.0%) - No filtering applied")
        
        self.all_records = all_records
        return all_records
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Override split_data to ensure proper train/test split
        
        Args:
            test_size: Proportion for test set  
            random_state: Random seed
        """
        # CRITICAL: Handle boolean test_size (base class sometimes passes True)
        if isinstance(test_size, bool):
            test_size = 0.2 if test_size else 0.0
        
        np.random.seed(random_state)
        n_records = len(self.all_records)
        n_test = int(n_records * test_size)
        n_train = n_records - n_test
        
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
        Prepare features for Elastic Net model
        22 features total, including disability indicators
        
        Returns:
            Tuple of (feature matrix, feature names)
        """
        if not records:
            return np.array([]), []
        
        features_list = []
        feature_names = []
        
        for record in records:
            row_features = []
            
            # 1. Living setting dummies (5 features, drop FH as reference)
            living_settings = ['ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
            for setting in living_settings:
                value = 1.0 if record.living_setting == setting else 0.0
                row_features.append(value)
            
            # 2. Age group dummies (2 features, drop Age3_20 as reference)
            age_groups = ['Age21_30', 'Age31Plus']
            for age_group in age_groups:
                value = 1.0 if record.age_group == age_group else 0.0
                row_features.append(value)
            
            # 3. Selected QSI questions (10 features)
            selected_qsi = [16, 18, 20, 21, 23, 28, 33, 34, 36, 43]
            for q_num in selected_qsi:
                value = getattr(record, f'q{q_num}', 0)
                row_features.append(float(value))
            
            # 4. Summary scores (2 features)
            row_features.append(float(record.bsum))  # Behavioral sum
            row_features.append(float(record.fsum))  # Functional sum
            
            # 5. Developmental Disability indicators (3 features)
            # Based on DevelopmentalDisability field
            disability = getattr(record, 'developmental_disability', '')
            row_features.append(1.0 if 'Intellectual' in str(disability) else 0.0)
            row_features.append(1.0 if 'Autism' in str(disability) else 0.0)
            row_features.append(1.0 if 'Cerebral' in str(disability) else 0.0)
            
            features_list.append(row_features)
        
        # Build feature names (only once)
        if not self.feature_names:
            # Living setting dummies
            for setting in living_settings:
                feature_names.append(f'Live_{setting}')
            
            # Age group dummies
            for age_group in age_groups:
                feature_names.append(f'Age_{age_group}')
            
            # Selected QSI questions
            for q_num in selected_qsi:
                feature_names.append(f'Q{q_num}')
            
            # Summary scores
            feature_names.append('BSum')
            feature_names.append('FSum')
            
            # Developmental disability indicators
            feature_names.append('DD_Intellectual')
            feature_names.append('DD_Autism')
            feature_names.append('DD_CerebralPalsy')
            
            self.feature_names = feature_names
        
        # Convert to numpy array
        X = np.array(features_list, dtype=np.float64)
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} records")
        
        return X, feature_names
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Elastic Net model with cross-validated parameter selection
        
        Args:
            X: Feature matrix
            y: Target values (square-root transformed costs)
        """
        logger.info("Fitting Elastic Net model with cross-validated parameter selection")
        
        # Scale features for regularization
        self.X_train_scaled = self.scaler.fit_transform(X)
        
        # Use ElasticNetCV for automatic parameter selection
        self.model = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            n_alphas=self.n_alphas,
            cv=10,
            max_iter=2000,
            tol=1e-4,
            random_state=42,
            selection='cyclic'
        )
        
        # Fit model
        self.model.fit(self.X_train_scaled, y)
        
        # Store optimal parameters
        self.alpha = self.model.alpha_
        self.l1_ratio = self.model.l1_ratio_
        
        # Store CV results
        self.cv_alphas = self.model.alphas_
        self.cv_scores = self.model.mse_path_
        
        logger.info(f"Optimal alpha: {self.alpha:.6f}")
        logger.info(f"Optimal l1_ratio: {self.l1_ratio:.2f}")
        
        # Analyze feature selection
        self._analyze_feature_selection()
    
    def _analyze_feature_selection(self) -> None:
        """Analyze which features were selected or dropped"""
        coef = self.model.coef_
        
        # Store coefficients with feature names
        for i, name in enumerate(self.feature_names):
            self.coefficients[name] = {
                'value': coef[i],
                'selected': abs(coef[i]) > 1e-10,
                'abs_value': abs(coef[i])
            }
        
        # Identify selected and dropped features
        self.selected_features = [name for name, info in self.coefficients.items() 
                                 if info['selected']]
        self.dropped_features = [name for name, info in self.coefficients.items() 
                                if not info['selected']]
        
        self.n_features_selected = len(self.selected_features)
        self.n_features_dropped = len(self.dropped_features)
        
        logger.info(f"Features selected: {self.n_features_selected}/{len(self.feature_names)}")
        logger.info(f"Features dropped: {self.n_features_dropped}")
        
        # Log dropped features
        if self.dropped_features:
            logger.info(f"Dropped features: {', '.join(self.dropped_features)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted Elastic Net model
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values (in square-root space)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, Any]:
        """
        Perform cross-validation for model evaluation
        
        Args:
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        # Prepare features if not already done
        if self.X_train is None or self.y_train is None:
            if self.train_records:
                self.X_train, _ = self.prepare_features(self.train_records)
                self.y_train = np.array([np.sqrt(r.total_cost) for r in self.train_records])
            else:
                raise ValueError("No training records available")
        
        logger.info(f"Performing {n_splits}-fold cross-validation")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.X_train)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
            X_fold_train = X_scaled[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = X_scaled[val_idx]
            y_fold_val = self.y_train[val_idx]
            
            # Fit Elastic Net for this fold
            fold_model = ElasticNetCV(
                l1_ratio=self.l1_ratios,
                n_alphas=50,  # Fewer alphas for speed in CV
                cv=3,  # Inner CV
                max_iter=2000,
                random_state=42
            )
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Evaluate
            y_pred = fold_model.predict(X_fold_val)
            score = r2_score(y_fold_val, y_pred)
            cv_scores.append(score)
            
            logger.info(f"Fold {fold}: R^2 = {score:.4f}, alpha = {fold_model.alpha_:.6f}, "
                       f"l1_ratio = {fold_model.l1_ratio_:.2f}")
        
        # Store CV results
        self.cv_scores = cv_scores
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logger.info(f"Cross-validation R^2: {cv_mean:.4f} +- {cv_std:.4f}")
        
        return {
            'scores': cv_scores,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std
        }
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary of metrics
        """
        if self.train_predictions is None or self.test_predictions is None:
            # Make predictions if not already done
            self.train_predictions = self.predict(self.X_train)
            self.test_predictions = self.predict(self.X_test)
        
        # Transform predictions back to dollar scale
        train_pred_dollars = self.train_predictions ** 2
        test_pred_dollars = self.test_predictions ** 2
        train_actual_dollars = self.y_train ** 2
        test_actual_dollars = self.y_test ** 2
        
        # Calculate metrics
        metrics = {
            # Training metrics
            'r2_train': r2_score(train_actual_dollars, train_pred_dollars),
            'rmse_train': np.sqrt(mean_squared_error(train_actual_dollars, train_pred_dollars)),
            'mae_train': mean_absolute_error(train_actual_dollars, train_pred_dollars),
            
            # Test metrics
            'r2_test': r2_score(test_actual_dollars, test_pred_dollars),
            'rmse_test': np.sqrt(mean_squared_error(test_actual_dollars, test_pred_dollars)),
            'mae_test': mean_absolute_error(test_actual_dollars, test_pred_dollars),
            
            # Sample sizes
            'n_train': len(self.y_train),
            'n_test': len(self.y_test),
            
            # Model-specific metrics
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'n_features_selected': self.n_features_selected,
            'n_features_dropped': self.n_features_dropped,
            'sparsity_percent': (self.n_features_selected / len(self.feature_names)) * 100
        }
        
        # MAPE calculation
        train_mape = np.mean(np.abs((train_actual_dollars - train_pred_dollars) / 
                                   train_actual_dollars)) * 100
        test_mape = np.mean(np.abs((test_actual_dollars - test_pred_dollars) / 
                                  test_actual_dollars)) * 100
        metrics['mape_train'] = train_mape
        metrics['mape_test'] = test_mape
        
        # Within threshold metrics
        thresholds = [1000, 2000, 5000, 10000, 20000]
        for threshold in thresholds:
            train_within = np.mean(np.abs(train_actual_dollars - train_pred_dollars) <= threshold) * 100
            test_within = np.mean(np.abs(test_actual_dollars - test_pred_dollars) <= threshold) * 100
            
            # Use words for LaTeX compatibility
            threshold_word = {1000: 'one_k', 2000: 'two_k', 5000: 'five_k', 
                            10000: 'ten_k', 20000: 'twenty_k'}[threshold]
            metrics[f'within_{threshold_word}_train'] = train_within
            metrics[f'within_{threshold_word}_test'] = test_within
        
        # Find most and least important features
        sorted_features = sorted(self.coefficients.items(), 
                               key=lambda x: x[1]['abs_value'], 
                               reverse=True)
        
        if self.selected_features:
            # Most important is the one with largest absolute coefficient
            metrics['most_important_feature'] = sorted_features[0][0]
            
            # Least important non-zero feature
            selected_sorted = [(name, info) for name, info in sorted_features 
                             if info['selected']]
            if selected_sorted:
                metrics['least_important_feature'] = selected_sorted[-1][0]
        
        # Add CV metrics if available
        if hasattr(self, 'cv_scores') and self.cv_scores is not None and len(self.cv_scores) > 0:
            metrics['cv_r2_mean'] = np.mean(self.cv_scores)
            metrics['cv_r2_std'] = np.std(self.cv_scores)
        
        self.metrics = metrics
        return metrics
    
    def calculate_subgroup_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for demographic subgroups"""
        subgroup_metrics = {}
        
        # Transform predictions to dollar scale
        test_pred_dollars = self.test_predictions ** 2
        test_actual_dollars = np.array([r.total_cost for r in self.test_records])
        
        # Living setting subgroups
        living_settings = ['FH', 'ILSL', 'RH1', 'RH2', 'RH3', 'RH4']
        for setting in living_settings:
            mask = np.array([r.living_setting == setting for r in self.test_records])
            if np.sum(mask) > 0:
                subgroup_metrics[f'living_{setting}'] = self._compute_subgroup_metrics(
                    test_actual_dollars[mask], test_pred_dollars[mask]
                )
        
        # Age group subgroups
        age_groups = ['Age3_20', 'Age21_30', 'Age31Plus']
        for age_group in age_groups:
            mask = np.array([r.age_group == age_group for r in self.test_records])
            if np.sum(mask) > 0:
                subgroup_metrics[f'age_{age_group}'] = self._compute_subgroup_metrics(
                    test_actual_dollars[mask], test_pred_dollars[mask]
                )
        
        # Cost quartiles
        quartiles = np.percentile(test_actual_dollars, [25, 50, 75])
        quartile_names = ['q1_low', 'q2', 'q3', 'q4_high']
        
        for i, name in enumerate(quartile_names):
            if i == 0:
                mask = test_actual_dollars <= quartiles[0]
            elif i == 3:
                mask = test_actual_dollars > quartiles[2]
            else:
                mask = (test_actual_dollars > quartiles[i-1]) & (test_actual_dollars <= quartiles[i])
            
            if np.sum(mask) > 0:
                subgroup_metrics[f'cost_{name}'] = self._compute_subgroup_metrics(
                    test_actual_dollars[mask], test_pred_dollars[mask]
                )
        
        self.subgroup_metrics = subgroup_metrics
        return subgroup_metrics
    
    def _compute_subgroup_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Helper to compute metrics for a subgroup"""
        if len(y_true) == 0:
            return {'n': 0, 'r2': 0, 'rmse': 0, 'bias': 0}
        
        return {
            'n': len(y_true),
            'r2': r2_score(y_true, y_pred) if len(y_true) > 1 else 0,
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'bias': np.mean(y_pred - y_true)
        }
    
    def calculate_variance_metrics(self) -> Dict[str, float]:
        """Calculate variance-related metrics"""
        # Get predictions in dollar scale
        test_pred_dollars = self.test_predictions ** 2
        test_actual_dollars = np.array([r.total_cost for r in self.test_records])
        
        variance_metrics = {
            'cv_actual': np.std(test_actual_dollars) / np.mean(test_actual_dollars),
            'cv_predicted': np.std(test_pred_dollars) / np.mean(test_pred_dollars),
            'prediction_interval': 1.96 * np.std(test_pred_dollars - test_actual_dollars),
            'budget_actual_corr': np.corrcoef(test_actual_dollars, test_pred_dollars)[0, 1],
            'quarterly_variance': np.std(test_pred_dollars) / 4,  # Simplified
            'annual_adjustment_rate': 0.05  # Placeholder
        }
        
        self.variance_metrics = variance_metrics
        return variance_metrics
    
    def calculate_population_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Calculate population impact scenarios"""
        # Fixed budget amount
        total_budget = 1.2e9  # $1.2 billion
        
        # Get all predictions in dollar scale
        all_predictions = np.concatenate([
            self.train_predictions ** 2,
            self.test_predictions ** 2
        ])
        
        scenarios = {}
        
        # Current baseline (use actual costs)
        all_actual = np.concatenate([
            self.y_train ** 2,
            self.y_test ** 2
        ])
        
        scenarios['currentbaseline'] = {
            'clients_served': int(total_budget / np.mean(all_actual)),
            'avg_allocation': np.mean(all_actual),
            'waitlist_change': 0,
            'waitlist_pct': 0
        }
        
        # Model balanced (use predictions)
        base_clients = scenarios['currentbaseline']['clients_served']
        
        scenarios['modelbalanced'] = {
            'clients_served': int(total_budget / np.mean(all_predictions)),
            'avg_allocation': np.mean(all_predictions),
            'waitlist_change': int(total_budget / np.mean(all_predictions)) - base_clients,
            'waitlist_pct': ((int(total_budget / np.mean(all_predictions)) - base_clients) / 
                           base_clients * 100)
        }
        
        # Model efficiency (reduce high-cost allocations)
        efficient_predictions = np.clip(all_predictions, None, np.percentile(all_predictions, 90))
        scenarios['modelefficiency'] = {
            'clients_served': int(total_budget / np.mean(efficient_predictions)),
            'avg_allocation': np.mean(efficient_predictions),
            'waitlist_change': int(total_budget / np.mean(efficient_predictions)) - base_clients,
            'waitlist_pct': ((int(total_budget / np.mean(efficient_predictions)) - base_clients) / 
                           base_clients * 100)
        }
        
        # Category focused (prioritize certain groups)
        focused_predictions = all_predictions * 0.95  # 5% reduction
        scenarios['categoryfocused'] = {
            'clients_served': int(total_budget / np.mean(focused_predictions)),
            'avg_allocation': np.mean(focused_predictions),
            'waitlist_change': int(total_budget / np.mean(focused_predictions)) - base_clients,
            'waitlist_pct': ((int(total_budget / np.mean(focused_predictions)) - base_clients) / 
                           base_clients * 100)
        }
        
        # Population maximized
        maximized_predictions = all_predictions * 0.90  # 10% reduction
        scenarios['populationmaximized'] = {
            'clients_served': int(total_budget / np.mean(maximized_predictions)),
            'avg_allocation': np.mean(maximized_predictions),
            'waitlist_change': int(total_budget / np.mean(maximized_predictions)) - base_clients,
            'waitlist_pct': ((int(total_budget / np.mean(maximized_predictions)) - base_clients) / 
                           base_clients * 100)
        }
        
        self.population_scenarios = scenarios
        return scenarios
    
    def generate_diagnostic_plots(self) -> None:
        """Generate comprehensive diagnostic plots including sparsity visualization"""
        # First generate standard 6-panel diagnostic plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Model {self.model_id}: {self.model_name} Diagnostic Plots', fontsize=14)
        
        # Get predictions in dollar scale
        train_pred_dollars = self.train_predictions ** 2
        test_pred_dollars = self.test_predictions ** 2
        train_actual_dollars = self.y_train ** 2
        test_actual_dollars = self.y_test ** 2
        
        # 1. Predicted vs Actual (Test)
        ax = axes[0, 0]
        ax.scatter(test_actual_dollars, test_pred_dollars, alpha=0.5, s=10)
        ax.plot([0, max(test_actual_dollars)], [0, max(test_actual_dollars)], 'r--', lw=2)
        ax.set_xlabel('Actual Cost ($)')
        ax.set_ylabel('Predicted Cost ($)')
        ax.set_title(f'Predicted vs Actual (Test R^2={self.metrics["r2_test"]:.3f})')
        ax.grid(True, alpha=0.3)
        
        # 2. Residual Plot
        ax = axes[0, 1]
        residuals = test_actual_dollars - test_pred_dollars
        ax.scatter(test_pred_dollars, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Cost ($)')
        ax.set_ylabel('Residuals ($)')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # 3. Feature Selection Summary
        ax = axes[0, 2]
        selected_count = self.n_features_selected
        dropped_count = self.n_features_dropped
        ax.bar(['Selected', 'Dropped'], [selected_count, dropped_count], 
               color=['green', 'red'], alpha=0.7)
        ax.set_ylabel('Number of Features')
        ax.set_title(f'Feature Selection (α={self.alpha:.4f}, L1={self.l1_ratio:.2f})')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Q-Q Plot
        ax = axes[1, 0]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # 5. Histogram of Residuals
        ax = axes[1, 1]
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residuals ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Performance by Cost Quartile
        ax = axes[1, 2]
        quartiles = ['Q1\nLow', 'Q2', 'Q3', 'Q4\nHigh']
        r2_values = []
        for q in ['cost_q1_low', 'cost_q2', 'cost_q3', 'cost_q4_high']:
            if q in self.subgroup_metrics:
                r2_values.append(self.subgroup_metrics[q]['r2'])
            else:
                r2_values.append(0)
        
        ax.bar(quartiles, r2_values, color='steelblue', alpha=0.7)
        ax.set_ylabel('R^2 Score')
        ax.set_title('Performance by Cost Quartile')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save diagnostic plots
        diagnostic_file = self.output_dir / 'diagnostic_plots.png'
        plt.savefig(diagnostic_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plots saved to {diagnostic_file}")
        
        # Generate separate sparsity visualization
        self.generate_sparsity_plot()
    
    def generate_sparsity_plot(self) -> None:
        """Generate sparsity pattern visualization"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Model {self.model_id}: Elastic Net Sparsity Analysis', fontsize=14)
        
        # 1. Coefficient Magnitude Bar Chart
        ax = axes[0]
        
        # Sort features by absolute coefficient value
        sorted_features = sorted(self.coefficients.items(), 
                               key=lambda x: x[1]['abs_value'], 
                               reverse=True)
        
        names = [self._clean_feature_name(name) for name, _ in sorted_features]
        values = [info['value'] for _, info in sorted_features]
        colors = ['green' if info['selected'] else 'red' 
                 for _, info in sorted_features]
        
        positions = np.arange(len(names))
        bars = ax.barh(positions, values, color=colors, alpha=0.7)
        
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Coefficient Value')
        ax.set_title(f'Feature Coefficients ({self.n_features_selected}/{len(self.feature_names)} selected)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Selected'),
                          Patch(facecolor='red', alpha=0.7, label='Dropped')]
        ax.legend(handles=legend_elements, loc='best')
        
        # 2. Regularization Path
        ax = axes[1]
        
        # Generate path for different alpha values
        alphas_to_plot = np.logspace(np.log10(0.0001), np.log10(1.0), 50)
        coef_paths = []
        
        # Fit models with different alphas
        X_scaled = self.scaler.transform(self.X_train)
        
        for alpha in alphas_to_plot:
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=self.l1_ratio,
                max_iter=2000,
                random_state=42
            )
            model.fit(X_scaled, self.y_train)
            coef_paths.append(model.coef_)
        
        coef_paths = np.array(coef_paths).T
        
        # Plot paths for top features
        top_features_idx = [i for i, (name, info) in enumerate(sorted_features[:10])]
        
        for idx in top_features_idx:
            feature_name = self._clean_feature_name(self.feature_names[idx])
            ax.plot(alphas_to_plot, coef_paths[idx], label=feature_name, linewidth=2)
        
        ax.set_xscale('log')
        ax.set_xlabel('Regularization Parameter (α)')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'Regularization Path (L1 Ratio = {self.l1_ratio:.2f})')
        ax.axvline(x=self.alpha, color='red', linestyle='--', 
                  label=f'Optimal α={self.alpha:.4f}')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save sparsity plot
        sparsity_file = self.output_dir / 'sparsity_analysis.png'
        plt.savefig(sparsity_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sparsity analysis plot saved to {sparsity_file}")
    
    def _clean_feature_name(self, name: str) -> str:
        """Clean feature name for display"""
        # Shorten long names
        name = name.replace('DD_', '')
        name = name.replace('Live_', '')
        name = name.replace('Age_', '')
        name = name.replace('Age31Plus', 'Age31+')
        name = name.replace('Age21_30', 'Age21-30')
        name = name.replace('CerebralPalsy', 'CP')
        return name
    
    def save_metrics(self) -> None:
        """Save metrics to JSON file"""
        metrics_file = self.output_dir / 'metrics.json'
        
        # Prepare metrics for JSON serialization
        metrics_to_save = {}
        for key, value in self.metrics.items():
            if isinstance(value, (int, float)):
                metrics_to_save[key] = float(value)
            else:
                metrics_to_save[key] = str(value)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
    
    def save_predictions(self) -> None:
        """Save predictions to NPZ file"""
        predictions_file = self.output_dir / 'predictions.npz'
        np.savez_compressed(
            predictions_file,
            train_predictions=self.train_predictions,
            test_predictions=self.test_predictions,
            y_train=self.y_train,
            y_test=self.y_test,
            selected_features=self.selected_features,
            dropped_features=self.dropped_features,
            coefficients=list(self.coefficients.values())
        )
        logger.info(f"Predictions saved to {predictions_file}")
    
    def generate_latex_commands(self) -> None:
        """Generate LaTeX commands including Elastic Net specific ones"""
        # First generate base commands from parent class
        super().generate_latex_commands()
        
        # Add Model 6 specific commands
        model_word = 'Six'
        
        # Add to newcommands file
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        with open(newcommands_file, 'a') as f:
            f.write("\n% Elastic Net Specific Commands\n")
            f.write(f"\\newcommand{{\\Model{model_word}Alpha}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}LOneRatio}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}FeaturesSelected}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}FeaturesDropped}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}SparsityPercent}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}MostImportant}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}LeastImportant}}{{\\WarningRunPipeline}}\n")
        
        # Add actual values to renewcommands file
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        with open(renewcommands_file, 'a') as f:
            f.write("\n% Elastic Net Specific Metrics\n")
            f.write(f"\\renewcommand{{\\Model{model_word}Alpha}}{{{self.alpha:.6f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}LOneRatio}}{{{self.l1_ratio:.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}FeaturesSelected}}{{{self.n_features_selected}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}FeaturesDropped}}{{{self.n_features_dropped}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}SparsityPercent}}{{{self.metrics.get('sparsity_percent', 0):.1f}}}\n")
            
            # Most and least important features
            most_important = self.metrics.get('most_important_feature', 'N/A')
            least_important = self.metrics.get('least_important_feature', 'N/A')
            
            # Clean feature names for LaTeX display
            most_important_clean = self._clean_feature_name(most_important) if most_important != 'N/A' else 'N/A'
            least_important_clean = self._clean_feature_name(least_important) if least_important != 'N/A' else 'N/A'
            
            f.write(f"\\renewcommand{{\\Model{model_word}MostImportant}}{{{most_important_clean}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}LeastImportant}}{{{least_important_clean}}}\n")
    
    def run_complete_pipeline(self) -> None:
        """Run the complete model pipeline"""
        logger.info(f"Starting complete pipeline for {self.model_name}")
        
        # Load ALL data
        self.all_records = self.load_data()
        
        # Split data
        self.split_data(test_size=0.2, random_state=42)
        
        # Prepare features BEFORE cross-validation
        self.X_train, _ = self.prepare_features(self.train_records)
        self.y_train = np.array([np.sqrt(r.total_cost) for r in self.train_records])
        
        self.X_test, _ = self.prepare_features(self.test_records)
        self.y_test = np.array([np.sqrt(r.total_cost) for r in self.test_records])
        
        # Perform cross-validation
        cv_results = self.perform_cross_validation(n_splits=10)
        
        # Fit model on full training set
        self.fit(self.X_train, self.y_train)
        
        # Make predictions
        self.train_predictions = self.predict(self.X_train)
        self.test_predictions = self.predict(self.X_test)
        
        # Calculate metrics
        self.metrics = self.calculate_metrics()
        
        # Calculate subgroup metrics
        self.subgroup_metrics = self.calculate_subgroup_metrics()
        
        # Calculate variance metrics
        self.variance_metrics = self.calculate_variance_metrics()
        
        # Calculate population scenarios
        self.population_scenarios = self.calculate_population_scenarios()
        
        # Generate outputs
        self.save_metrics()
        self.save_predictions()
        self.generate_diagnostic_plots()
        self.generate_latex_commands()
        
        # Print summary
        self.print_summary()
        
        logger.info(f"Pipeline completed for {self.model_name}")
    
    def print_summary(self) -> None:
        """Print model summary"""
        print("\n" + "="*60)
        print(f"MODEL {self.model_id}: {self.model_name.upper()} RESULTS")
        print("="*60)
        
        print("\nData Summary:")
        print(f"  Total Records Used: {len(self.all_records)} (100% - no filtering)")
        print(f"  Training Set: {self.metrics.get('n_train', 0)}")
        print(f"  Test Set: {self.metrics.get('n_test', 0)}")
        
        print("\nElastic Net Parameters:")
        print(f"  Optimal Alpha: {self.alpha:.6f}")
        print(f"  L1 Ratio: {self.l1_ratio:.2f}")
        print(f"  Features Selected: {self.n_features_selected}/{len(self.feature_names)}")
        print(f"  Sparsity: {self.metrics.get('sparsity_percent', 0):.1f}%")
        
        print("\nPerformance Metrics:")
        print(f"  Training R^2: {self.metrics.get('r2_train', 0):.4f}")
        print(f"  Test R^2: {self.metrics.get('r2_test', 0):.4f}")
        print(f"  Test RMSE: ${self.metrics.get('rmse_test', 0):,.2f}")
        print(f"  Test MAE: ${self.metrics.get('mae_test', 0):,.2f}")
        print(f"  Test MAPE: {self.metrics.get('mape_test', 0):.2f}%")
        
        if hasattr(self, 'cv_scores') and self.cv_scores is not None and len(self.cv_scores) > 0:
            print("\nCross-Validation:")
            print(f"  Mean R^2: {self.metrics.get('cv_r2_mean', 0):.4f} +- {self.metrics.get('cv_r2_std', 0):.4f}")
        
        print("\nFeature Selection:")
        if self.selected_features:
            print(f"  Selected Features: {', '.join(self.selected_features[:5])}...")
        if self.dropped_features:
            print(f"  Dropped Features: {', '.join(self.dropped_features)}")
        
        print("\nAccuracy Thresholds (Test Set):")
        print(f"  Within $1,000: {self.metrics.get('within_one_k_test', 0):.1f}%")
        print(f"  Within $5,000: {self.metrics.get('within_five_k_test', 0):.1f}%")
        print(f"  Within $10,000: {self.metrics.get('within_ten_k_test', 0):.1f}%")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run model
    model = Model6ElasticNet()
    model.run_complete_pipeline()