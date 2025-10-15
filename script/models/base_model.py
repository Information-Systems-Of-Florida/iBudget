"""
base_model.py
=============
Simplified Base Class for iBudget Models

FIXED: Type conversion issues for ConsumerRecord creation
- ConsumerID: int -> str
- BLEVEL/FLEVEL/PLEVEL: str -> float  
- QSI questions: None -> 0.0
- Enhanced error logging
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
import logging
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ConsumerRecord:
    """Data structure for consumer records"""
    consumer_id: str
    fiscal_year: int
    age: int
    age_group: str
    living_setting: str
    total_cost: float
    
    # QSI questions (q14-q50)
    q14: float = 0
    q15: float = 0
    q16: float = 0
    q17: float = 0
    q18: float = 0
    q19: float = 0
    q20: float = 0
    q21: float = 0
    q22: float = 0
    q23: float = 0
    q24: float = 0
    q25: float = 0
    q26: float = 0
    q27: float = 0
    q28: float = 0
    q29: float = 0
    q30: float = 0
    q31: float = 0
    q32: float = 0
    q33: float = 0
    q34: float = 0
    q35: float = 0
    q36: float = 0
    q37: float = 0
    q38: float = 0
    q39: float = 0
    q40: float = 0
    q41: float = 0
    q42: float = 0
    q43: float = 0
    q44: float = 0
    q45: float = 0
    q46: float = 0
    q47: float = 0
    q48: float = 0
    q49: float = 0
    q50: float = 0
    
    # Summary scores
    bsum: float = 0
    fsum: float = 0
    psum: float = 0
    
    # Support levels
    blevel: float = 0
    flevel: float = 0
    plevel: float = 0
    losri: float = 0
    olevel: float = 0
    
    # Other fields
    gender: str = ""
    county: str = ""
    primary_diagnosis: str = ""
    
    def __post_init__(self):
        """Ensure numeric fields are float"""
        for field in ['age', 'total_cost', 'bsum', 'fsum', 'psum', 
                      'blevel', 'flevel', 'plevel', 'losri', 'olevel']:
            if hasattr(self, field):
                value = getattr(self, field)
                if value is None:
                    setattr(self, field, 0.0)
                else:
                    setattr(self, field, float(value))


class BaseiBudgetModel(ABC):
    """
    Base class for iBudget prediction models
    
    Handles all common functionality:
    - Data loading/splitting
    - Outlier removal (studentized residuals)
    - Transformations (sqrt/log/none)
    - Cross-validation
    - Metrics calculation
    - LaTeX generation
    - Diagnostic plots
    
    Child classes implement:
    - prepare_features(records) -> (X, feature_names)
    - _fit_core(X, y) -> fit model
    - _predict_core(X) -> predict in fitted scale
    """
    
    def __init__(self, model_id: int, model_name: str, 
                    transformation: str = 'none',
                    use_outlier_removal: bool = False,
                    outlier_threshold: float = 3,
                    random_seed: int = 42,
                    log_suffix: str = None):
        """
        Initialize base model
        
        Args:
            model_id: Model number (1-10)
            model_name: Descriptive name
            use_outlier_removal: Whether to remove outliers
            outlier_threshold: Threshold for studentized residuals (default: 1.645 for ~10%)
            transformation: 'sqrt', 'log', or 'none'
            random_seed: Random seed for reproducibility
        """
        self.model_id = model_id
        self.model_name = model_name
        
        # Outlier configuration
        self.use_outlier_removal = use_outlier_removal
        self.outlier_threshold = outlier_threshold
        self.outlier_diagnostics = {}
        
        # Transformation configuration
        self.transformation = transformation
        
        # Random seed
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Data storage
        self.all_records: List[ConsumerRecord] = []
        self.train_records: List[ConsumerRecord] = []
        self.test_records: List[ConsumerRecord] = []
        
        # Features and targets
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        
        # Predictions
        self.train_predictions: Optional[np.ndarray] = None
        self.test_predictions: Optional[np.ndarray] = None
        
        # Model and metrics
        self.model = None
        self.metrics: Dict[str, float] = {}
        self.cv_results: Dict[str, Any] = {}
        self.subgroup_metrics: Dict[str, Dict[str, float]] = {}
        self.variance_metrics: Dict[str, float] = {}
        self.population_scenarios: Dict[str, Dict[str, float]] = {}
        
        # Output directory
        # Make path relative to script location
        script_dir = Path(__file__).parent
        self.output_dir_relative = Path("../../report/models") / f"model_{model_id}"
        self.output_dir = (script_dir / self.output_dir_relative).resolve()  # Keep absolute for operations
        self.output_dir.mkdir(parents=True, exist_ok=True)     
        
        # Set up logging with optional suffix
        self.log_suffix = log_suffix
        self._setup_logging()
                
        # Log initialization
        self.log_section(f"INITIALIZING MODEL {self.model_id}: {self.model_name.upper()}", "=")
        self.logger.info(f"Output directory: {self.output_dir_relative}")  # Log the relative path        
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Outlier removal: {use_outlier_removal}")
        if use_outlier_removal:
            self.logger.info(f"  - Outlier threshold: +-{outlier_threshold} (studentized residuals)")
        self.logger.info(f"  - Transformation: {transformation}")
        self.logger.info(f"  - Random seed: {random_seed}")
    
    def _setup_logging(self):
        """Set up model-specific logging"""
        log_dir = Path("../../report/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        # Build log filename with optional suffix
        if self.log_suffix:
            log_filename = log_dir / f"model_{self.model_id}_log_{self.log_suffix}.txt"
        else:
            log_filename = log_dir / f"model_{self.model_id}_log.txt"        
        
        self.logger = logging.getLogger(f"MODEL_{self.model_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
    
        # Prevent duplicate output, e.g. from Orchestrator.py
        self.logger.propagate = False 

        fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_section(self, title: str, char: str = "-"):
        """Log section header"""
        self.logger.info("")
        self.logger.info(char * 60)
        self.logger.info(title.upper())
        self.logger.info(char * 60)
    
    def log_metrics_summary(self):
        """Log formatted metrics summary"""
        self.logger.info("")
        self.logger.info("-" * 60)
        self.logger.info("PERFORMANCE METRICS SUMMARY")
        self.logger.info("-" * 60)
        
        if self.metrics:
            self.logger.info(f"Training R^2: {self.metrics.get('r2_train', 0):.4f}")
            self.logger.info(f"Test R^2: {self.metrics.get('r2_test', 0):.4f}")
            self.logger.info(f"RMSE: ${self.metrics.get('rmse_test', 0):,.2f}")
            self.logger.info(f"MAE: ${self.metrics.get('mae_test', 0):,.2f}")
            self.logger.info(f"MAPE: {self.metrics.get('mape_test', 0):.2f}%")
            
            if 'cv_mean' in self.metrics:
                self.logger.info(f"CV R^2 (mean +- std): {self.metrics['cv_mean']:.4f} +- "
                               f"{self.metrics.get('cv_std', 0):.4f}")
        
        self.logger.info("-" * 60)

    def log_feature_importance(self, feature_stats, model_type="Model"):
            """
            Generic method to log feature importance/coefficients for any model.
            
            Args:
                feature_stats: Dict or list where each item has at minimum 'name' and a value metric
                            Examples:
                            - GLM: {'name': 'feat1', 'coefficient': 1.2, 'p_value': 0.01, ...}
                            - RF: {'name': 'feat1', 'importance': 0.15}
                            - XGBoost: {'name': 'feat1', 'gain': 0.23}
                model_type: String describing the model (e.g., "GLM-Gamma", "Random Forest")
            """
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"{model_type} Feature Analysis (All {len(feature_stats)} Features)")
            self.logger.info("=" * 80)
            
            # Convert dict to list if needed
            if isinstance(feature_stats, dict):
                feature_list = []
                for name, data in feature_stats.items():
                    if isinstance(data, dict):
                        feature_list.append({'name': name, **data})
                    else:
                        # Simple value (like RF importance)
                        feature_list.append({'name': name, 'value': data})
            else:
                feature_list = feature_stats
            
            # Determine the main metric to sort by
            if feature_list:
                first_item = feature_list[0]
                if 'coefficient' in first_item:
                    # GLM-style: sort by absolute coefficient
                    feature_list.sort(key=lambda x: abs(x.get('coefficient', 0)), reverse=True)
                    metric_name = "coefficient"
                elif 'importance' in first_item:
                    # RandomForest-style
                    feature_list.sort(key=lambda x: x.get('importance', 0), reverse=True)
                    metric_name = "importance"
                elif 'gain' in first_item:
                    # XGBoost-style
                    feature_list.sort(key=lambda x: x.get('gain', 0), reverse=True)
                    metric_name = "gain"
                else:
                    # Generic: use first numeric value found
                    metric_name = "value"
                    feature_list.sort(key=lambda x: abs(x.get('value', 0)), reverse=True)
                
                # Check if we have p-values for significance separation
                has_pvalues = 'p_value' in first_item
                
                if has_pvalues:
                    # Separate by significance
                    sig_features = [f for f in feature_list if f.get('p_value', 1) < 0.05]
                    non_sig_features = [f for f in feature_list if f.get('p_value', 1) >= 0.05]
                    
                    if sig_features:
                        self.logger.info(f"Significant Features (p < 0.05): {len(sig_features)} features")
                        self.logger.info("-" * 60)
                        for idx, feat in enumerate(sig_features, 1):
                            self._format_feature_line(idx, feat, metric_name)
                    
                    if non_sig_features:
                        self.logger.info(f"Non-Significant Features (p >= 0.05): {len(non_sig_features)} features")
                        self.logger.info("-" * 60)
                        for idx, feat in enumerate(non_sig_features, 1):
                            self._format_feature_line(idx, feat, metric_name)
                else:
                    # No p-values: just list all features
                    self.logger.info(f"Features sorted by {metric_name}:")
                    self.logger.info("-" * 60)
                    for idx, feat in enumerate(feature_list, 1):
                        self._format_feature_line(idx, feat, metric_name)
            
            self.logger.info("=" * 80)
        
    def _format_feature_line(self, idx, feat, primary_metric):
        """Format a single feature line based on available metrics."""
        name = feat.get('name', 'unknown')
        parts = [f"  {idx:3d}. {name:30s}"]
        
        # Add the primary metric
        if primary_metric == 'coefficient' and 'coefficient' in feat:
            parts.append(f"Beta={feat['coefficient']:8.4f}")
        elif primary_metric == 'importance' and 'importance' in feat:
            parts.append(f"importance={feat['importance']:8.4f}")
        elif primary_metric == 'gain' and 'gain' in feat:
            parts.append(f"gain={feat['gain']:8.4f}")
        elif 'value' in feat:
            parts.append(f"value={feat['value']:8.4f}")
        
        # Add additional stats if available
        if 'std_error' in feat:
            parts.append(f"SE={feat['std_error']:7.4f}")
        if 'p_value' in feat:
            if feat['p_value'] < 0.0001:
                parts.append("p<0.0001")
            else:
                parts.append(f"p={feat['p_value']:7.4f}")
        if 'effect_pct' in feat:
            parts.append(f"effect={feat['effect_pct']:+8.1f}%")
        if 'z_value' in feat:
            parts.append(f"z={feat['z_value']:6.2f}")
        
        self.logger.info(" ".join(parts))
        
    def log_all_features(self, features_data, title="Sorted Features (by coefficient magnitude)"):
        """Log all model features with their statistics.
        
        Args:
            features_data: Dict or list of dicts with feature statistics
            title: Section title for the feature listing
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(title)
        self.logger.info("=" * 80)
        
        # Convert to list of dicts if needed
        if isinstance(features_data, dict):
            # If it's a simple dict of feature_name: value
            if features_data and not isinstance(next(iter(features_data.values())), dict):
                features_list = [{'name': k, 'value': v} for k, v in features_data.items()]
            else:
                # If it's a dict of feature_name: dict_of_stats
                features_list = [{'name': k, **v} for k, v in features_data.items()]
        else:
            features_list = features_data
        
        # Sort by absolute coefficient if available
        if features_list and 'coefficient' in features_list[0]:
            features_list.sort(key=lambda x: abs(x.get('coefficient', 0)), reverse=True)
        elif features_list and 'importance' in features_list[0]:
            features_list.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        # Separate by significance if p-values exist
        if features_list and 'p_value' in features_list[0]:
            sig_features = [f for f in features_list if f.get('p_value', 1) < 0.05]
            non_sig_features = [f for f in features_list if f.get('p_value', 1) >= 0.05]
            
            # Log significant features
            if sig_features:
                self.logger.info(f"Significant Features (p < 0.05): {len(sig_features)} features")
                self.logger.info("-" * 60)
                for idx, feat in enumerate(sig_features, 1):
                    self._log_single_feature(idx, feat)
            
            # Log non-significant features
            if non_sig_features:
                self.logger.info("")
                self.logger.info(f"Non-Significant Features (p >= 0.05): {len(non_sig_features)} features")
                self.logger.info("-" * 60)
                for idx, feat in enumerate(non_sig_features, 1):
                    self._log_single_feature(idx, feat)
        else:
            # Just log all features if no p-values
            for idx, feat in enumerate(features_list, 1):
                self._log_single_feature(idx, feat)
        
        self.logger.info("=" * 80)
    
    def _log_single_feature(self, idx, feature):
        """Helper to log a single feature's statistics."""
        name = feature.get('name', 'unknown')
        parts = [f"  {idx:3d}. {name:30s}"]
        
        if 'coefficient' in feature:
            parts.append(f"Beta={feature['coefficient']:8.4f}")
        if 'std_error' in feature:
            parts.append(f"SE={feature['std_error']:7.4f}")
        if 'p_value' in feature:
            if feature['p_value'] < 0.0001:
                parts.append("p<0.0001")
            else:
                parts.append(f"p={feature['p_value']:7.4f}")
        if 'effect_pct' in feature:
            parts.append(f"effect={feature['effect_pct']:+8.1f}%")
        if 'importance' in feature:
            parts.append(f"importance={feature['importance']:8.4f}")
        
        self.logger.info(" ".join(parts))
            
    def log_final_summary(self, summary_dict):
        """Log the final model summary consistently."""
        self.log_section(f"MODEL {self.model_id} FINAL SUMMARY", "=")
        
        for key, value in summary_dict.items():
            if isinstance(value, dict):
                self.logger.info(f"{key}:") # \n
                for sub_key, sub_value in value.items():
                    self.logger.info(f"     {sub_key}: {sub_value}")
            elif isinstance(value, list):
                self.logger.info(f"{key}:") # \n
                for item in value:
                    self.logger.info(f"     {item}")
            else:
                self.logger.info(f"{key}: {value}")
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"Model {self.model_id} pipeline complete!")
        self.logger.info(f"Results saved to: {self.output_dir_relative}")
        self.logger.info("=" * 80)
            
    # ========================================================================
    # TRANSFORMATION METHODS
    # ========================================================================
    
    def apply_transformation(self, y: np.ndarray) -> np.ndarray:
        """Apply transformation to target variable"""
        if self.transformation == 'sqrt':
            return np.sqrt(np.maximum(0, y))
        elif self.transformation == 'log':
            return np.log(np.maximum(1, y))
        else:
            return y
    
    def inverse_transformation(self, y_transformed: np.ndarray) -> np.ndarray:
        """Inverse transform predictions back to original scale"""
        if self.transformation == 'sqrt':
            return y_transformed ** 2
        elif self.transformation == 'log':
            return np.exp(y_transformed)
        else:
            return y_transformed
    
    # ========================================================================
    # OUTLIER DETECTION (Studentized Residuals Method)
    # ========================================================================
    
    def calculate_leverage(self, X: np.ndarray) -> np.ndarray:
        """Calculate leverage values (diagonal of hat matrix)"""
        n_samples = X.shape[0]
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        
        try:
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            h_diag = np.sum(X_with_intercept * (X_with_intercept @ XtX_inv), axis=1)
        except np.linalg.LinAlgError:
            self.logger.warning("Singular matrix in leverage calculation, using pseudo-inverse")
            XtX_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
            h_diag = np.sum(X_with_intercept * (X_with_intercept @ XtX_inv), axis=1)
        
        h_diag = np.clip(h_diag, 1e-6, 1 - 1e-6)
        return h_diag
    
    def remove_outliers_studentized(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Remove outliers using studentized residuals (Tao & Niu 2015)"""
        from sklearn.linear_model import LinearRegression
        
        self.logger.info(f"Applying studentized residuals outlier detection...")
        self.logger.info(f"  Threshold: |t_i| >= {self.outlier_threshold}")
        
        # Apply transformation
        y_transformed = self.apply_transformation(y)
        
        # Fit preliminary model
        prelim_model = LinearRegression()
        prelim_model.fit(X, y_transformed)
        
        # Calculate residuals
        y_pred_transformed = prelim_model.predict(X)
        residuals = y_transformed - y_pred_transformed
        
        # Calculate leverage
        h_diag = self.calculate_leverage(X)
        
        # Calculate sigma_hat
        n = len(y)
        p = X.shape[1] + 1
        sigma_hat = np.std(residuals, ddof=p)
        
        # Calculate studentized residuals
        studentized_residuals = residuals / (sigma_hat * np.sqrt(1 - h_diag))
        
        # Identify outliers
        outlier_mask = np.abs(studentized_residuals) < self.outlier_threshold
        outlier_indices = np.where(~outlier_mask)[0]
        
        # Clean data
        X_clean = X[outlier_mask]
        y_clean = y[outlier_mask]
        
        n_removed = len(y) - len(y_clean)
        pct_removed = (n_removed / len(y)) * 100
        
        # Calculate diagnostics
        diagnostics = {
            'n_removed': n_removed,
            'pct_removed': pct_removed,
            'outlier_indices': outlier_indices.tolist(),
            'studentized_residuals': studentized_residuals,
            'leverage_values': h_diag,
            'mean_ti': float(np.mean(studentized_residuals)),
            'std_ti': float(np.std(studentized_residuals)),
            'pct_within_threshold': float(np.mean(outlier_mask) * 100),
            'leverage_mean': float(np.mean(h_diag)),
            'leverage_max': float(np.max(h_diag)),
            'high_leverage_threshold': 2 * p / n,
            'high_leverage_count': int(np.sum(h_diag > (2 * p / n)))
        }
        
        # Log diagnostics
        self.logger.info(f"Outlier Detection Results:")
        self.logger.info(f"  Removed: {n_removed:,} observations ({pct_removed:.2f}%)")
        self.logger.info(f"  Studentized residuals - Mean: {diagnostics['mean_ti']:.4f}, "
                        f"Std: {diagnostics['std_ti']:.4f}")
        self.logger.info(f"  % within threshold: {diagnostics['pct_within_threshold']:.1f}%")
        
        return X_clean, y_clean, diagnostics
    
    # ========================================================================
    # DATA LOADING AND SPLITTING
    # ========================================================================
    
    def _map_field_names(self, record_dict: dict) -> dict:
        """Map PascalCase field names from cache to snake_case for ConsumerRecord"""
        field_map = {
            'ConsumerID': 'consumer_id',
            'FiscalYear': 'fiscal_year',
            'Age': 'age',
            'AgeGroup': 'age_group',
            'LivingSetting': 'living_setting',
            'TotalCost': 'total_cost',
            'GENDER': 'gender',
            'County': 'county',
            'PrimaryDiagnosis': 'primary_diagnosis',
            'BSum': 'bsum',
            'FSum': 'fsum',
            'PSum': 'psum',
            'BLEVEL': 'blevel',
            'FLEVEL': 'flevel',
            'PLEVEL': 'plevel',
            'LOSRI': 'losri',
            'OLEVEL': 'olevel',
        }
        
        # QSI questions Q14-Q50
        for i in range(14, 51):
            field_map[f'Q{i}'] = f'q{i}'
        
        # Map fields with type conversions
        mapped = {}
        for cache_key, record_key in field_map.items():
            if cache_key in record_dict:
                value = record_dict[cache_key]
                
                # CRITICAL: Convert ConsumerID from int to str
                if cache_key == 'ConsumerID':
                    try:
                        value = str(value)
                    except:
                        value = "0"
                
                # Parse TotalCost from string to float
                elif cache_key == 'TotalCost':
                    if isinstance(value, str):
                        try:
                            value = float(value.replace(',', '').replace('$', ''))
                        except (ValueError, AttributeError):
                            value = 0.0
                    elif value is None:
                        value = 0.0
                    else:
                        try:
                            value = float(value)
                        except:
                            value = 0.0
                
                # Parse sum fields (bsum, fsum, psum) to float
                elif record_key in ['bsum', 'fsum', 'psum']:
                    if value is None:
                        value = 0.0
                    elif isinstance(value, str):
                        try:
                            value = float(value)
                        except (ValueError, AttributeError):
                            value = 0.0
                    else:
                        try:
                            value = float(value)
                        except:
                            value = 0.0
                
                # CRITICAL: Convert support levels (BLEVEL, FLEVEL, PLEVEL, etc.) from str to float
                elif record_key in ['blevel', 'flevel', 'plevel', 'losri', 'olevel']:
                    if value is None:
                        value = 0.0
                    elif isinstance(value, str):
                        try:
                            value = float(value)
                        except (ValueError, AttributeError):
                            value = 0.0
                    else:
                        try:
                            value = float(value)
                        except:
                            value = 0.0
                
                # CRITICAL: Convert QSI questions - handle None values
                elif record_key.startswith('q') and record_key[1:].isdigit():
                    if value is None:
                        value = 0.0
                    else:
                        try:
                            value = float(value)
                        except:
                            value = 0.0
                
                # Age needs to be int then float
                elif cache_key == 'Age':
                    if value is None:
                        value = 0
                    else:
                        try:
                            value = int(value)
                        except:
                            value = 0
                
                # FiscalYear needs to be int
                elif cache_key == 'FiscalYear':
                    if value is None:
                        value = 2024
                    else:
                        try:
                            value = int(value)
                        except:
                            value = 2024
                
                mapped[record_key] = value
        
        return mapped
    
    def load_data(self, fiscal_year_start: int = 2024, fiscal_year_end: int = 2024) -> List[ConsumerRecord]:
        """Load data from cached pickle files"""
        self.log_section(f"LOADING DATA: FY{fiscal_year_start}-{fiscal_year_end}")
        
        all_records = []
        
        for year in range(fiscal_year_start, fiscal_year_end + 1):
            cache_file = Path(f"data/cached/fy{year}.pkl")
            
            if not cache_file.exists():
                self.logger.warning(f"Cache file not found: {cache_file}")
                continue
            
            self.logger.info(f"Loading data from {cache_file}")
            
            with open(cache_file, 'rb') as f:
                year_data = pickle.load(f)
            
            # Extract records list from dict structure
            if isinstance(year_data, dict) and 'data' in year_data:
                records_list = year_data['data']
                self.logger.info(f"Found {len(records_list)} records in 'data' key")
            elif isinstance(year_data, list):
                records_list = year_data
                self.logger.info(f"Found {len(records_list)} records (direct list)")
            else:
                self.logger.error(f"Unexpected data structure: {type(year_data)}")
                continue
            
            # Process records
            records_processed = 0
            records_skipped_unusable = 0
            records_skipped_nocost = 0
            records_skipped_error = 0
            
            for i, record_data in enumerate(records_list):
                try:
                    if not isinstance(record_data, dict):
                        records_skipped_error += 1
                        if records_skipped_error <= 3:
                            self.logger.error(f"Record {i}: Not a dict, got {type(record_data)}")
                        continue
                    
                    # Filter by Usable flag
                    #if record_data.get('Usable', 0) != 1:
                    #    records_skipped_unusable += 1
                    #    continue
                    
                    # Map field names
                    mapped_data = self._map_field_names(record_data)
                    
                    # Create ConsumerRecord
                    record = ConsumerRecord(**mapped_data)
                    
                    # Only include records with positive costs
                    if record.total_cost > 0:
                        all_records.append(record)
                        records_processed += 1
                    else:
                        records_skipped_nocost += 1
                        
                except Exception as e:
                    records_skipped_error += 1
                    if records_skipped_error <= 3:
                        self.logger.error(f"Record {i} error: {type(e).__name__}: {str(e)}")
                        self.logger.error(f"  ConsumerID: {record_data.get('ConsumerID', 'N/A')}")
                        self.logger.error(f"  Sample keys: {list(record_data.keys())[:5]}")
            
            self.logger.info(f"FY{year} Summary:")
            self.logger.info(f"  - Processed: {records_processed:,}")
            self.logger.info(f"  - Skipped (Usable=0): {records_skipped_unusable:,}")
            self.logger.info(f"  - Skipped (no cost): {records_skipped_nocost:,}")
            self.logger.info(f"  - Skipped (errors): {records_skipped_error:,}")
        
        self.logger.info(f"Total loaded: {len(all_records):,} usable records")
        
        if len(all_records) == 0:
            self.logger.error("No records loaded!")
        
        self.all_records = all_records
        return all_records
    
    def split_data(self, test_size: float = 0.2, random_state: Optional[int] = None) -> None:
        """Split data into train and test sets"""
        if random_state is None:
            random_state = self.random_seed
        
        self.log_section("DATA SPLIT")
        
        np.random.seed(random_state)
        n_records = len(self.all_records)
        n_test = int(n_records * test_size)
        
        indices = np.arange(n_records)
        np.random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        self.train_records = [self.all_records[i] for i in train_indices]
        self.test_records = [self.all_records[i] for i in test_indices]
        
        self.logger.info(f"Training samples: {len(self.train_records):,}")
        self.logger.info(f"Test samples: {len(self.test_records):,}")
        self.logger.info(f"Split ratio: {test_size * 100:.1f}%")
    
    # ========================================================================
    # ABSTRACT METHODS - Child classes must implement
    # ========================================================================
    
    @abstractmethod
    def prepare_features(self, records: List[ConsumerRecord]) -> Tuple[np.ndarray, List[str]]:
        """Prepare features from consumer records"""
        pass
    
    @abstractmethod
    def _fit_core(self, X: np.ndarray, y: np.ndarray) -> None:
        """Core model fitting logic (child implements)"""
        pass
    
    @abstractmethod
    def _predict_core(self, X: np.ndarray) -> np.ndarray:
        """Core prediction logic (child implements)"""
        pass
    

    # ========================================================================
    # DYNAMIC FEATURE PREPARATION
    # ========================================================================
    def prepare_features_from_spec(self, 
                                records: List[ConsumerRecord],
                                feature_config: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        Generic feature preparation from configuration dictionary with comprehensive logging
        """
        features_list = []
        feature_names = []
        feature_counts = {}  # Track counts by category
        
        for i, record in enumerate(records):
            row_features = []
            
            # Numeric features
            if 'numeric' in feature_config:
                for field in feature_config['numeric']:
                    value = getattr(record, field, 0) or 0
                    row_features.append(float(value))
                    if i == 0:  # Build names only once
                        feature_names.append(field)
                if i == 0:
                    feature_counts['numeric'] = len(feature_config['numeric'])
            
            # Categorical features (one-hot encoding)
            if 'categorical' in feature_config:
                cat_count = 0
                for field_name, config in feature_config['categorical'].items():
                    field_value = getattr(record, field_name, '')
                    for category in config['categories']:
                        row_features.append(1.0 if field_value == category else 0.0)
                        if i == 0:
                            feature_names.append(category)
                            cat_count += 1
                if i == 0:
                    feature_counts['categorical'] = cat_count
            
            # Binary features from lambdas
            if 'binary' in feature_config:
                for feature_name, condition_func in feature_config['binary'].items():
                    row_features.append(1.0 if condition_func(record) else 0.0)
                    if i == 0:
                        feature_names.append(feature_name)
                if i == 0:
                    feature_counts['binary'] = len(feature_config['binary'])
            
            # QSI features
            if 'qsi' in feature_config:
                for q_num in feature_config['qsi']:
                    value = getattr(record, f'q{q_num}', 0) or 0
                    row_features.append(float(value))
                    if i == 0:
                        feature_names.append(f'Q{q_num}')
                if i == 0:
                    feature_counts['qsi'] = len(feature_config['qsi'])
            
            # Interaction terms
            if 'interactions' in feature_config:
                for name, func in feature_config['interactions']:
                    row_features.append(func(record))
                    if i == 0:
                        feature_names.append(name)
                if i == 0:
                    feature_counts['interactions'] = len(feature_config['interactions'])
            
            features_list.append(row_features)
        
        # Comprehensive logging
        total_features = len(feature_names)
        
        # Summary logging (INFO level)
        self.logger.info(f"Features prepared: {total_features} total features from {len(records)} records")
        
        # Detailed breakdown (INFO level)
        self.logger.info("Feature breakdown:")
        
        # Categorical features detail
        if 'categorical' in feature_config:
            for field_name, config in feature_config['categorical'].items():
                ref = config.get('reference', 'None')
                cats = ', '.join(config['categories'])
                self.logger.info(f"  {field_name}: {len(config['categories'])} categories [{cats}] (reference: {ref})")
        
        # Binary features detail  
        if 'binary' in feature_counts:
            binary_names = ', '.join(feature_config['binary'].keys())
            self.logger.info(f"  Binary: {feature_counts['binary']} features [{binary_names}]")
        
        # Numeric features detail
        if 'numeric' in feature_counts:
            numeric_names = ', '.join(feature_config['numeric'])
            self.logger.info(f"  Numeric: {feature_counts['numeric']} features [{numeric_names}]")
        
        # QSI features detail
        if 'qsi' in feature_counts:
            qsi_list = feature_config['qsi']
            if len(qsi_list) > 10:
                qsi_str = f"Q{qsi_list[0]}-Q{qsi_list[-1]} ({len(qsi_list)} items)"
            else:
                qsi_str = ', '.join([f"Q{q}" for q in qsi_list])
            self.logger.info(f"  QSI: {feature_counts['qsi']} features [{qsi_str}]")
        
        # Interaction terms detail
        if 'interactions' in feature_counts:
            interaction_names = ', '.join([name for name, _ in feature_config['interactions']])
            self.logger.info(f"  Interactions: {feature_counts['interactions']} features [{interaction_names}]")
        
        # Feature counts summary
        self.logger.info("Feature counts by type:")
        for category, count in feature_counts.items():
            self.logger.info(f"  {category}: {count}")
        
        # Debug level - list ALL features
        if self.logger.level <= logging.DEBUG:
            self.logger.debug("Complete feature list:")
            for i, name in enumerate(feature_names, 1):
                self.logger.debug(f"  {i:3d}. {name}")
        
        # Validation checks
        if len(feature_names) != len(row_features):
            self.logger.error(f"Feature count mismatch: names={len(feature_names)}, values={len(row_features)}")
        
        # Check for any missing data warnings
        if len(records) > 0:
            first_row = features_list[0]
            nan_count = sum(1 for x in first_row if np.isnan(x) or x is None)
            if nan_count > 0:
                self.logger.warning(f"Found {nan_count} missing values in first record")
        
        return np.array(features_list), feature_names

    # ========================================================================
    # TEMPLATE METHODS
    # ========================================================================
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model with preprocessing (template method)"""
        # Remove outliers if enabled
        if self.use_outlier_removal:
            X, y, self.outlier_diagnostics = self.remove_outliers_studentized(X, y)
            # FIX: Update stored training data to match what was actually used
            self.X_train = X
            self.y_train = y        
            
        # Apply transformation
        y_fit = self.apply_transformation(y)
        
        # Call child's core fitting logic
        self._fit_core(X, y_fit)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with back-transformation (template method)"""
        # Get predictions in fitted scale
        y_pred = self._predict_core(X)
        
        # Inverse transform
        y_pred = self.inverse_transformation(y_pred)
        
        # Ensure non-negative
        y_pred = np.maximum(0, y_pred)
        
        return y_pred
    
    # ========================================================================
    # CROSS-VALIDATION
    # ========================================================================

    def predict_original(self, X: np.ndarray) -> np.ndarray:
        """
        Unified hook to get predictions in ORIGINAL dollar scale for ANY model.

        Default behavior:
        1) Call _predict_core(X) -> fitted-scale predictions
        2) Inverse-transform to original dollars
        3) Enforce non-negativity

        Child models that ALREADY return original-scale predictions inside
        _predict_core() should override this method to just return their predictions.
        """
        y_pred_fitted = self._predict_core(X)
        y_pred_original = self.inverse_transformation(y_pred_fitted)
        return np.maximum(0.0, y_pred_original)
    
    def perform_cross_validation(self, n_splits: int = 10) -> Dict[str, Any]:
        """Perform k-fold CV with proper preprocessing per fold"""
        self.log_section(f"{n_splits}-FOLD CROSS-VALIDATION")
        
        if self.X_train is None or self.y_train is None:
            self.logger.warning("No training data for cross-validation")
            return {'cv_mean': 0, 'cv_std': 0, 'scores': []}
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_val_original = self.y_train[val_idx]
            
            # Remove outliers if enabled
            if self.use_outlier_removal:
                X_fold_train, y_fold_train, fold_diagnostics = self.remove_outliers_studentized(
                    X_fold_train, y_fold_train
                )
                self.logger.info(f"  Fold {fold}: Removed {fold_diagnostics['n_removed']} outliers "
                               f"({fold_diagnostics['pct_removed']:.2f}%)")
            
            # Transform
            y_fold_train_fit = self.apply_transformation(y_fold_train)
            
            # Fit core model
            self._fit_core(X_fold_train, y_fold_train_fit)
            
            # Predict
            ##y_fold_pred_fitted = self._predict_core(X_fold_val)
            ## Inverse transform
            ##y_fold_pred_original = self.inverse_transformation(y_fold_pred_fitted)
            ##y_fold_pred_original = np.maximum(0, y_fold_pred_original)
            y_fold_pred_original = self.predict_original(X_fold_val)
            
            # Evaluate on original scale
            score = r2_score(y_fold_val_original, y_fold_pred_original)
            cv_scores.append(score)
            
            self.logger.info(f"  Fold {fold}: R^2 = {score:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        self.logger.info("")
        self.logger.info("Cross-validation summary:")
        self.logger.info(f"  Mean R^2: {cv_mean:.4f}")
        self.logger.info(f"  Std R^2: {cv_std:.4f}")
        self.logger.info(f"  Min R^2: {min(cv_scores):.4f} (Fold {cv_scores.index(min(cv_scores))+1})")
        self.logger.info(f"  Max R^2: {max(cv_scores):.4f} (Fold {cv_scores.index(max(cv_scores))+1})")
        self.logger.info(f"  95% CI: [{cv_mean - 1.96*cv_std:.4f}, {cv_mean + 1.96*cv_std:.4f}]")
        
        return {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'scores': cv_scores
        }
    
    # ========================================================================
    # METRICS CALCULATION
    # ========================================================================
    
    def calculate_metricsX(self) -> Dict[str, float]:
        """Calculate performance metrics on original scale"""
        self.log_section("CALCULATING METRICS")
        
        metrics = {}  # Build new metrics
        
        # Training metrics
        if self.train_predictions is not None and self.y_train is not None:
            metrics['r2_train'] = r2_score(self.y_train, self.train_predictions)
            metrics['rmse_train'] = np.sqrt(mean_squared_error(self.y_train, self.train_predictions))
            metrics['mae_train'] = mean_absolute_error(self.y_train, self.train_predictions)
            
            mape_mask = self.y_train > 100
            if np.sum(mape_mask) > 0:
                metrics['mape_train'] = mean_absolute_percentage_error(
                    self.y_train[mape_mask], self.train_predictions[mape_mask]
                ) * 100
            else:
                metrics['mape_train'] = 0.0
        
        # Test metrics
        if self.test_predictions is not None and self.y_test is not None:
            metrics['r2_test'] = r2_score(self.y_test, self.test_predictions)
            metrics['rmse_test'] = np.sqrt(mean_squared_error(self.y_test, self.test_predictions))
            metrics['mae_test'] = mean_absolute_error(self.y_test, self.test_predictions)
            
            mape_mask = self.y_test > 100
            if np.sum(mape_mask) > 0:
                metrics['mape_test'] = mean_absolute_percentage_error(
                    self.y_test[mape_mask], self.test_predictions[mape_mask]
                ) * 100
            else:
                metrics['mape_test'] = 0.0
        
        # Sample sizes
        metrics['training_samples'] = len(self.train_records)
        metrics['test_samples'] = len(self.test_records)
        
        # Accuracy bands
        if self.test_predictions is not None and self.y_test is not None:
            errors = np.abs(self.test_predictions - self.y_test)
            for threshold in [1000, 2000, 5000, 10000, 20000]:
                pct = np.mean(errors <= threshold) * 100
                metrics[f'within_{threshold//1000}k'] = pct
        
        # UPDATE existing metrics instead of replacing
        self.metrics.update(metrics)  # Changed from: self.metrics = metrics
        
        self.log_metrics_summary()
        
        return self.metrics  # Changed from: return metrics

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        self.log_section("CALCULATING METRICS")
        
        if self.test_predictions is None or self.y_test is None:
            raise ValueError("Must run fit() and predict() before calculating metrics")
        
        # Get predictions in transformed scale (sqrt)
        y_train_transformed = self.apply_transformation(self.y_train)
        y_test_transformed = self.apply_transformation(self.y_test)
        X_train_transformed = self.X_train
        X_test_transformed = self.X_test
        
        # Get predictions in transformed scale
        train_predictions_transformed = self._predict_core(X_train_transformed)
        test_predictions_transformed = self._predict_core(X_test_transformed)
        
        # Calculate RMSE in TRANSFORMED (sqrt) scale - for fair comparison with 2015
        rmse_train_sqrt = np.sqrt(mean_squared_error(y_train_transformed, train_predictions_transformed))
        rmse_test_sqrt = np.sqrt(mean_squared_error(y_test_transformed, test_predictions_transformed))
        
        # Calculate RMSE in ORIGINAL scale (already have self.test_predictions in original scale)
        rmse_train_original = np.sqrt(mean_squared_error(self.y_train, self.train_predictions))
        rmse_test_original = np.sqrt(mean_squared_error(self.y_test, self.test_predictions))
        
        # R^2 scores
        r2_train = r2_score(self.y_train, self.train_predictions)
        r2_test = r2_score(self.y_test, self.test_predictions)
        
        # MAE (original scale only)
        mae_train = mean_absolute_error(self.y_train, self.train_predictions)
        mae_test = mean_absolute_error(self.y_test, self.test_predictions)
        
        # MAPE (original scale only) - but cap it to avoid extreme values
        mape_train = np.mean(np.abs((self.y_train - self.train_predictions) / (self.y_train + 1e-10))) * 100
        mape_test = np.mean(np.abs((self.y_test - self.test_predictions) / (self.y_test + 1e-10))) * 100
        
        # Accuracy bands (original scale)
        for threshold in [1000, 2000, 5000, 10000, 20000]:
            within = np.mean(np.abs(self.test_predictions - self.y_test) <= threshold) * 100
            threshold_name = f'{threshold//1000}k' if threshold >= 1000 else str(threshold)
            self.metrics[f'within_{threshold_name}'] = within
        
        # Store both scales
        self.metrics.update({
            'cv_mean': self.cv_results.get('cv_mean', self.cv_results.get('mean_score', 0)),
            'cv_std': self.cv_results.get('cv_std', self.cv_results.get('std_score', 0)),
            'r2_train': r2_train,
            'r2_test': r2_test,
            # ORIGINAL SCALE RMSE (for absolute dollar comparison)
            'rmse_train': rmse_train_original,
            'rmse_test': rmse_test_original,
            # SQRT SCALE RMSE (for fair comparison with 2015 Model 5b)
            'rmse_train_sqrt': rmse_train_sqrt,
            'rmse_test_sqrt': rmse_test_sqrt,
            # Other metrics
            'mae_train': mae_train,
            'mae_test': mae_test,
            'mape_train': mape_train,
            'mape_test': mape_test,
            'training_samples': len(self.y_train),
            'test_samples': len(self.y_test),
        })
        
        self.logger.info("")
        self.logger.info("------------------------------------------------------------")
        self.logger.info("PERFORMANCE METRICS SUMMARY")
        self.logger.info("------------------------------------------------------------")
        self.logger.info(f"Training R^2: {r2_train:.4f}")
        self.logger.info(f"Test R^2: {r2_test:.4f}")
        self.logger.info(f"RMSE (original): ${rmse_test_original:,.2f}")
        self.logger.info(f"RMSE (sqrt scale): ${rmse_test_sqrt:.2f}")  # ADD THIS LINE
        self.logger.info(f"MAE: ${mae_test:,.2f}")
        self.logger.info(f"MAPE: {mape_test:.2f}%")
        if 'cv_mean' in self.metrics and 'cv_std' in self.metrics:
            self.logger.info(f"CV R^2 (mean +- std): {self.metrics['cv_mean']:.4f} +- {self.metrics['cv_std']:.4f}")
        self.logger.info("------------------------------------------------------------")
        
        return self.metrics
    
    def calculate_subgroup_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics by subgroup"""
        self.log_section("SUBGROUP ANALYSIS")
        
        if self.test_predictions is None or self.y_test is None:
            return {}
        
        subgroup_metrics = {}
        
        # By living setting
        for setting in ['FH', 'ILSL', 'RH1-4']:
            if setting == 'RH1-4':
                mask = np.array([r.living_setting in ['RH1', 'RH2', 'RH3', 'RH4'] 
                               for r in self.test_records])
            else:
                mask = np.array([r.living_setting == setting for r in self.test_records])
            
            if np.sum(mask) > 0:
                y_sub = self.y_test[mask]
                pred_sub = self.test_predictions[mask]
                r2 = r2_score(y_sub, pred_sub)
                if r2 < -10:
                    r2 = -10.0
                
                subgroup_metrics[f'living_{setting}'] = {
                    'n': int(np.sum(mask)),
                    'r2': float(r2),
                    'rmse': float(np.sqrt(mean_squared_error(y_sub, pred_sub))),
                    'bias': float(np.mean(pred_sub - y_sub))
                }
        
        # By age group
        for age_group, name in [('Age3_20', 'AgeUnderTwentyOne'), 
                                 ('Age21_30', 'AgeTwentyOneToThirty'),
                                 ('Age31Plus', 'AgeThirtyOnePlus')]:
            mask = np.array([r.age_group == age_group for r in self.test_records])
            
            if np.sum(mask) > 0:
                y_sub = self.y_test[mask]
                pred_sub = self.test_predictions[mask]
                r2 = r2_score(y_sub, pred_sub)
                if r2 < -10:
                    r2 = -10.0
                
                subgroup_metrics[f'age_{name}'] = {
                    'n': int(np.sum(mask)),
                    'r2': float(r2),
                    'rmse': float(np.sqrt(mean_squared_error(y_sub, pred_sub))),
                    'bias': float(np.mean(pred_sub - y_sub))
                }
        
        # By cost quartile
        quartiles = np.percentile(self.y_test, [25, 50, 75])
        for i, (q_name, bounds) in enumerate([
            ('QOneLow', (0, quartiles[0])),
            ('QTwo', (quartiles[0], quartiles[1])),
            ('QThree', (quartiles[1], quartiles[2])),
            ('QFourHigh', (quartiles[2], np.inf))
        ]):
            mask = (self.y_test >= bounds[0]) & (self.y_test < bounds[1])
            
            if np.sum(mask) > 0:
                y_sub = self.y_test[mask]
                pred_sub = self.test_predictions[mask]
                r2 = r2_score(y_sub, pred_sub)
                if r2 < -10:
                    r2 = -10.0
                
                subgroup_metrics[f'cost_{q_name}'] = {
                    'n': int(np.sum(mask)),
                    'r2': float(r2),
                    'rmse': float(np.sqrt(mean_squared_error(y_sub, pred_sub))),
                    'bias': float(np.mean(pred_sub - y_sub))
                }
        
        self.subgroup_metrics = subgroup_metrics
        self.logger.info(f"Calculated metrics for {len(subgroup_metrics)} subgroups")
        
        # Log detailed subgroup results
        self.logger.info("")
        for subgroup_name, metrics in subgroup_metrics.items():
            self.logger.info(f"  {subgroup_name}:")
            self.logger.info(f"    N: {metrics['n']:,}")
            self.logger.info(f"    R^2: {metrics['r2']:.4f}")
            self.logger.info(f"    RMSE: ${metrics['rmse']:,.2f}")
            self.logger.info(f"    Bias: ${metrics['bias']:+,.2f}")
        
        return subgroup_metrics
        
    def calculate_variance_metrics(self) -> Dict[str, float]:
        """Calculate variance and stability metrics"""
        self.log_section("VARIANCE ANALYSIS")
        
        if self.test_predictions is None or self.y_test is None:
            return {}
        
        variance_metrics = {
            'cv_actual': float(np.std(self.y_test) / np.mean(self.y_test)),
            'cv_predicted': float(np.std(self.test_predictions) / np.mean(self.test_predictions)),
            'prediction_interval': float(1.96 * np.std(self.test_predictions - self.y_test)),
            'budget_actual_corr': float(np.corrcoef(self.test_predictions, self.y_test)[0, 1])
        }
        
        self.variance_metrics = variance_metrics
        self.logger.info("Variance metrics calculated")
        
        # Log detailed variance metrics
        self.logger.info("")
        self.logger.info(f"  Coefficient of Variation (Actual): {variance_metrics['cv_actual']:.4f}")
        self.logger.info(f"  Coefficient of Variation (Predicted): {variance_metrics['cv_predicted']:.4f}")
        self.logger.info(f"  95% Prediction Interval Width: ${variance_metrics['prediction_interval']:,.2f}")
        self.logger.info(f"  Actual-Predicted Correlation: {variance_metrics['budget_actual_corr']:.4f}")
        
        return variance_metrics
    
    def calculate_population_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Calculate population impact scenarios"""
        self.log_section("POPULATION SCENARIOS")
        
        total_budget = 1_200_000_000
        
        if self.test_predictions is not None and len(self.test_predictions) > 0:
            avg_allocation = np.mean(self.test_predictions)
        else:
            avg_allocation = 40000
        
        base_clients = int(total_budget / avg_allocation) if avg_allocation > 0 else 30000
        
        scenarios = {
            'currentbaseline': {
                'clients_served': base_clients,
                'avg_allocation': avg_allocation,
                'waitlist_change': 0,
                'waitlist_pct': 0.0
            },
            'modelbalanced': {
                'clients_served': int(base_clients * 1.02),
                'avg_allocation': avg_allocation * 0.98,
                'waitlist_change': int(base_clients * 0.02),
                'waitlist_pct': 2.0
            },
            'modelefficiency': {
                'clients_served': int(base_clients * 1.05),
                'avg_allocation': avg_allocation * 0.95,
                'waitlist_change': int(base_clients * 0.05),
                'waitlist_pct': 5.0
            },
            'categoryfocused': {
                'clients_served': int(base_clients * 0.85),
                'avg_allocation': avg_allocation * 1.18,
                'waitlist_change': int(base_clients * -0.15),
                'waitlist_pct': -15.0
            }
        }
        
        self.population_scenarios = scenarios
        self.logger.info(f"Calculated {len(scenarios)} population scenarios")
        
        # Log detailed scenario results
        self.logger.info("")
        self.logger.info(f"  Base budget: $1,200,000,000")
        for scenario_name, metrics in scenarios.items():
            self.logger.info(f"  {scenario_name}:")
            self.logger.info(f"    Clients served: {metrics['clients_served']:,}")
            self.logger.info(f"    Avg allocation: ${metrics['avg_allocation']:,.2f}")
            self.logger.info(f"    Waitlist change: {metrics['waitlist_change']:+,}")
            self.logger.info(f"    Waitlist %: {metrics['waitlist_pct']:+.1f}%")
        
        return scenarios
    
    # ========================================================================
    # PIPELINE ORCHESTRATION
    # ========================================================================
    
    def run_complete_pipeline(self,
                             fiscal_year_start: int = 2024,
                             fiscal_year_end: int = 2024,
                             test_size: float = 0.2,
                             perform_cv: bool = True,
                             n_cv_folds: int = 10) -> Dict[str, Any]:
        """Run complete modeling pipeline"""
        self.log_section(f"STARTING PIPELINE: {self.model_name}", "=")
        
        # Load data
        self.load_data(fiscal_year_start, fiscal_year_end)
        
        if len(self.all_records) == 0:
            self.logger.error("No records loaded!")
            return {}
        
        # Split data
        self.split_data(test_size=test_size)
        
        # Prepare features
        self.log_section("FEATURE PREPARATION")
        self.X_train, self.feature_names = self.prepare_features(self.train_records)
        self.y_train = np.array([r.total_cost for r in self.train_records])
        self.X_test, _ = self.prepare_features(self.test_records)
        self.y_test = np.array([r.total_cost for r in self.test_records])
        self.logger.info(f"Features prepared: {len(self.feature_names)} features")
        self.logger.info(f"  Training shape: {self.X_train.shape}")
        self.logger.info(f"  Test shape: {self.X_test.shape}")
        self.logger.info(f"  Target range: ${self.y_train.min():,.2f} - ${self.y_train.max():,.2f}")
        self.logger.info(f"  Target mean: ${self.y_train.mean():,.2f}")
        self.logger.info(f"  Target median: ${np.median(self.y_train):,.2f}")
        
        # Cross-validation
        if perform_cv:
            self.cv_results = self.perform_cross_validation(n_splits=n_cv_folds)  # Store in self.cv_results
            self.metrics['cv_mean'] = self.cv_results.get('cv_mean', 0)
            self.metrics['cv_std'] = self.cv_results.get('cv_std', 0)
            
        # Fit model
        self.log_section("MODEL TRAINING")
        self.fit(self.X_train, self.y_train)
        self.logger.info("Model training complete")
        
        # Make predictions
        self.log_section("MAKING PREDICTIONS")
        self.train_predictions = self.predict(self.X_train)
        self.test_predictions = self.predict(self.X_test)
        self.logger.info("Predictions complete")
        
        # Calculate metrics
        self.calculate_metrics()
        self.calculate_subgroup_metrics()
        self.calculate_variance_metrics()
        self.calculate_population_scenarios()
        
        # Generate outputs
        self.log_section("GENERATING OUTPUTS")
        self.generate_latex_commands()
        self.save_results()
        self.plot_diagnostics()
        
        # Final summary
        self.log_section(f"PIPELINE COMPLETE: {self.model_name}", "=")
        self.log_metrics_summary()
        
        # print(f"DEBUG: metrics keys = {list(self.metrics.keys())}")
        # print(f"DEBUG: subgroup_metrics keys = {list(self.subgroup_metrics.keys())}")
        # print(f"DEBUG: variance_metrics keys = {list(self.variance_metrics.keys())}")
        # print(f"DEBUG: population_scenarios keys = {list(self.population_scenarios.keys())}")
        
        return {
            'metrics': self.metrics,
            'subgroup_metrics': self.subgroup_metrics,
            'variance_metrics': self.variance_metrics,
            'population_scenarios': self.population_scenarios
        }
    
    # ========================================================================
    # OUTPUT GENERATION
    # ========================================================================
    
    def _number_to_word(self, num: int) -> str:
        """Convert number to word for LaTeX commands"""
        words = {
            1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five',
            6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten'
        }
        return words.get(num, str(num))
    
    def _clean_latex_command_name(self, name: str) -> str:
        """Clean name for LaTeX command"""
        for num in range(50):
            name = name.replace(str(num), self._number_to_word(num))
        parts = name.split('_')
        return ''.join(word.capitalize() for word in parts)
    
    def generate_latex_commands(self) -> None:
        """Generate standard LaTeX commands for all models"""
        self.log_section("LATEX GENERATION")
        #print(f"DEBUG: Starting LaTeX generation for model {self.model_id}") 
        
        model_word = self._number_to_word(self.model_id)
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        #print(f"DEBUG: Writing to {renewcommands_file}")
        
        # ========================================================================
        # NEWCOMMANDS (Placeholders)
        # ========================================================================
        with open(newcommands_file, 'w') as f:
            f.write(f"% Model {self.model_id} LaTeX Commands (Placeholders)\n")
            f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic performance metrics
            #print("DEBUG: Writing basic metrics newcommands")
            for metric in ['RSquaredTrain', 'RSquaredTest', 'RMSETrain', 'RMSETest',
                        'RMSETrainSqrt', 'RMSETestSqrt',  
                        'MAETrain', 'MAETest', 'MAPETrain', 'MAPETest',
                        'CVMean', 'CVStd', 'CVCILower', 'CVCIUpper',
                        'TrainingSamples', 'TestSamples']:
                f.write(f"\\newcommand{{\\Model{model_word}{metric}}}{{\\WarningRunPipeline}}\n")
            
            # Accuracy bands
            #print("DEBUG: Accuracy bands")
            for threshold in ['OneK', 'TwoK', 'FiveK', 'TenK', 'TwentyK']:
                f.write(f"\\newcommand{{\\Model{model_word}Within{threshold}}}{{\\WarningRunPipeline}}\n")
            
            # Subgroup: Living settings
            #print("DEBUG: Subgroup: Living settings")
            for setting in ['FH', 'ILSL', 'RHOneFour']:
                for metric in ['N', 'RSquared', 'RMSE', 'Bias']:
                    f.write(f"\\newcommand{{\\Model{model_word}SubgroupLiving{setting}{metric}}}{{\\WarningRunPipeline}}\n")
            
            # Subgroup: Age groups
            #print("DEBUG: Subgroup: Age groups")
            for age in ['AgeUnderTwentyOne', 'AgeTwentyOneToThirty', 'AgeThirtyOnePlus']:
                for metric in ['N', 'RSquared', 'RMSE', 'Bias']:
                    f.write(f"\\newcommand{{\\Model{model_word}SubgroupAge{age}{metric}}}{{\\WarningRunPipeline}}\n")
            
            # Subgroup: Cost quartiles
            #print("DEBUG: Subgroup: Cost quartiles")
            for quartile in ['QOneLow', 'QTwo', 'QThree', 'QFourHigh']:
                for metric in ['N', 'RSquared', 'RMSE', 'Bias']:
                    f.write(f"\\newcommand{{\\Model{model_word}SubgroupCost{quartile}{metric}}}{{\\WarningRunPipeline}}\n")
            
            # Variance metrics
            for metric in ['CVActual', 'CVPredicted', 'PredictionInterval', 'BudgetActualCorr']:
                f.write(f"\\newcommand{{\\Model{model_word}{metric}}}{{\\WarningRunPipeline}}\n")
            
            # Population scenarios
            for scenario in ['currentbaseline', 'modelbalanced', 'modelefficiency', 'categoryfocused']:
                for metric in ['Clients', 'AvgAlloc', 'WaitlistChange', 'WaitlistPct']:
                    f.write(f"\\newcommand{{\\Model{model_word}Pop{scenario}{metric}}}{{\\WarningRunPipeline}}\n")
        
            # Outlier diagnostics (common to all models with outlier removal)
            f.write(f"\\newcommand{{\\Model{model_word}StudentizedResidualsMean}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}StudentizedResidualsStd}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}PctWithinThreshold}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}OutliersRemoved}}{{\\WarningRunPipeline}}\n")
            f.write(f"\\newcommand{{\\Model{model_word}OutlierPct}}{{\\WarningRunPipeline}}\n")
            
            # Number of features (common to all models)
            f.write(f"\\newcommand{{\\Model{model_word}NumFeatures}}{{\\WarningRunPipeline}}\n")
        
        # ========================================================================
        # RENEWCOMMANDS (Actual Values)
        # ========================================================================
        with open(renewcommands_file, 'w') as f:
            f.write(f"% Model {self.model_id} Actual Values\n")
            f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic metrics
            f.write(f"\\renewcommand{{\\Model{model_word}RSquaredTrain}}{{{self.metrics.get('r2_train', 0):.4f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}RSquaredTest}}{{{self.metrics.get('r2_test', 0):.4f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}RMSETrain}}{{{self.metrics.get('rmse_train', 0):,.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}RMSETest}}{{{self.metrics.get('rmse_test', 0):,.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}RMSETrainSqrt}}{{{self.metrics.get('rmse_train_sqrt', 0):.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}RMSETestSqrt}}{{{self.metrics.get('rmse_test_sqrt', 0):.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}MAETrain}}{{{self.metrics.get('mae_train', 0):,.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}MAETest}}{{{self.metrics.get('mae_test', 0):,.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}MAPETrain}}{{{self.metrics.get('mape_train', 0):.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}MAPETest}}{{{self.metrics.get('mape_test', 0):.2f}}}\n")
            
            # CV metrics
            f.write(f"\\renewcommand{{\\Model{model_word}CVMean}}{{{self.metrics.get('cv_mean', 0):.4f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}CVStd}}{{{self.metrics.get('cv_std', 0):.4f}}}\n")
            
            # CV confidence interval
            if 'cv_mean' in self.metrics and 'cv_std' in self.metrics:
                ci_lower = self.metrics['cv_mean'] - 1.96 * self.metrics['cv_std']
                ci_upper = self.metrics['cv_mean'] + 1.96 * self.metrics['cv_std']
                f.write(f"\\renewcommand{{\\Model{model_word}CVCILower}}{{{ci_lower:.4f}}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}CVCIUpper}}{{{ci_upper:.4f}}}\n")
            else:
                f.write(f"\\renewcommand{{\\Model{model_word}CVCILower}}{{0.0000}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}CVCIUpper}}{{0.0000}}\n")
            
            # Sample sizes
            f.write(f"\\renewcommand{{\\Model{model_word}TrainingSamples}}{{{self.metrics.get('training_samples', 0):,}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}TestSamples}}{{{self.metrics.get('test_samples', 0):,}}}\n")
            
            # Accuracy bands
            for threshold_k, threshold_name in [(1, 'OneK'), (2, 'TwoK'), (5, 'FiveK'), (10, 'TenK'), (20, 'TwentyK')]:
                pct = self.metrics.get(f'within_{threshold_k}k', 0)
                f.write(f"\\renewcommand{{\\Model{model_word}Within{threshold_name}}}{{{pct:.2f}}}\n")
            
            # Subgroups: Living settings
            subgroup_living_map = {
                'living_FH': 'FH',
                'living_ILSL': 'ILSL',
                'living_RH1-4': 'RHOneFour'
            }
            for key, name in subgroup_living_map.items():
                if key in self.subgroup_metrics:
                    sg = self.subgroup_metrics[key]
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLiving{name}N}}{{{sg['n']:,}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLiving{name}RSquared}}{{{sg['r2']:.4f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLiving{name}RMSE}}{{{sg['rmse']:,.2f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLiving{name}Bias}}{{{sg['bias']:,.2f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLiving{name}N}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLiving{name}RSquared}}{{---}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLiving{name}RMSE}}{{---}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupLiving{name}Bias}}{{---}}\n")
            
            # Subgroups: Age
            subgroup_age_map = {
                'age_AgeUnderTwentyOne': 'AgeUnderTwentyOne',
                'age_AgeTwentyOneToThirty': 'AgeTwentyOneToThirty',
                'age_AgeThirtyOnePlus': 'AgeThirtyOnePlus'
            }
            for key, name in subgroup_age_map.items():
                if key in self.subgroup_metrics:
                    sg = self.subgroup_metrics[key]
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAge{name}N}}{{{sg['n']:,}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAge{name}RSquared}}{{{sg['r2']:.4f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAge{name}RMSE}}{{{sg['rmse']:,.2f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAge{name}Bias}}{{{sg['bias']:,.2f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAge{name}N}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAge{name}RSquared}}{{---}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAge{name}RMSE}}{{---}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupAge{name}Bias}}{{---}}\n")
            
            # Subgroups: Cost quartiles
            subgroup_cost_map = {
                'cost_QOneLow': 'QOneLow',
                'cost_QTwo': 'QTwo',
                'cost_QThree': 'QThree',
                'cost_QFourHigh': 'QFourHigh'
            }
            for key, name in subgroup_cost_map.items():
                if key in self.subgroup_metrics:
                    sg = self.subgroup_metrics[key]
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupCost{name}N}}{{{sg['n']:,}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupCost{name}RSquared}}{{{sg['r2']:.4f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupCost{name}RMSE}}{{{sg['rmse']:,.2f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupCost{name}Bias}}{{{sg['bias']:,.2f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupCost{name}N}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupCost{name}RSquared}}{{---}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupCost{name}RMSE}}{{---}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}SubgroupCost{name}Bias}}{{---}}\n")
            
            # Variance metrics
            f.write(f"\\renewcommand{{\\Model{model_word}CVActual}}{{{self.variance_metrics.get('cv_actual', 0):.4f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}CVPredicted}}{{{self.variance_metrics.get('cv_predicted', 0):.4f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}PredictionInterval}}{{{self.variance_metrics.get('prediction_interval', 0):,.2f}}}\n")
            f.write(f"\\renewcommand{{\\Model{model_word}BudgetActualCorr}}{{{self.variance_metrics.get('budget_actual_corr', 0):.4f}}}\n")
            
            # Population scenarios
            for scenario in ['currentbaseline', 'modelbalanced', 'modelefficiency', 'categoryfocused']:
                if scenario in self.population_scenarios:
                    sc = self.population_scenarios[scenario]
                    f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}Clients}}{{{sc['clients_served']:,}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}AvgAlloc}}{{{sc['avg_allocation']:,.2f}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}WaitlistChange}}{{{sc['waitlist_change']:,}}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}WaitlistPct}}{{{sc['waitlist_pct']:.1f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}Clients}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}AvgAlloc}}{{0.00}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}WaitlistChange}}{{0}}\n")
                    f.write(f"\\renewcommand{{\\Model{model_word}Pop{scenario}WaitlistPct}}{{0.0}}\n")
            
            # Outlier diagnostics (if outlier removal was used)
            if self.outlier_diagnostics:
                f.write(f"\n% Outlier Diagnostics\n")
                if 'mean_ti' in self.outlier_diagnostics:
                    f.write(f"\\renewcommand{{\\Model{model_word}StudentizedResidualsMean}}{{{self.outlier_diagnostics['mean_ti']:.4f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}StudentizedResidualsMean}}{{N/A}}\n")
                
                if 'std_ti' in self.outlier_diagnostics:
                    f.write(f"\\renewcommand{{\\Model{model_word}StudentizedResidualsStd}}{{{self.outlier_diagnostics['std_ti']:.4f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}StudentizedResidualsStd}}{{N/A}}\n")
                
                if 'pct_within_threshold' in self.outlier_diagnostics:
                    f.write(f"\\renewcommand{{\\Model{model_word}PctWithinThreshold}}{{{self.outlier_diagnostics['pct_within_threshold']:.1f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}PctWithinThreshold}}{{N/A}}\n")
                    
                if 'n_removed' in self.outlier_diagnostics:
                    f.write(f"\\renewcommand{{\\Model{model_word}OutliersRemoved}}{{{self.outlier_diagnostics['n_removed']:,}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}OutliersRemoved}}{{0}}\n")
                
                if 'pct_removed' in self.outlier_diagnostics:
                    f.write(f"\\renewcommand{{\\Model{model_word}OutlierPct}}{{{self.outlier_diagnostics['pct_removed']:.2f}}}\n")
                else:
                    f.write(f"\\renewcommand{{\\Model{model_word}OutlierPct}}{{0.00}}\n")                            
            else:
                # No outlier removal - set to N/A
                f.write(f"\n% Outlier Diagnostics (not used)\n")
                f.write(f"\\renewcommand{{\\Model{model_word}StudentizedResidualsMean}}{{N/A}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}StudentizedResidualsStd}}{{N/A}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}PctWithinThreshold}}{{N/A}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OutliersRemoved}}{{0}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}OutlierPct}}{{0.00}}\n")            
            # Number of features
            f.write(f"\n% Model Configuration\n")
            f.write(f"\\renewcommand{{\\Model{model_word}NumFeatures}}{{{len(self.feature_names)}}}\n")
                        
        # Count commands generated
        with open(newcommands_file, 'r') as f:
            newcommands = f.readlines()
            new_count = len([l for l in newcommands if l.strip().startswith('\\newcommand')])
        
        with open(renewcommands_file, 'r') as f:
            renewcommands = f.readlines()
            renew_count = len([l for l in renewcommands if l.strip().startswith('\\renewcommand')])
        
        self.logger.info("LaTeX commands generated:")
        self.logger.info(f"  - Newcommands file: {self.output_dir_relative / f'model_{self.model_id}_newcommands.tex'}")
        self.logger.info(f"    ({new_count} placeholder commands)")
        self.logger.info(f"  - Renewcommands file: {self.output_dir_relative / f'model_{self.model_id}_renewcommands.tex'}")
        self.logger.info(f"    ({renew_count} value commands)")
    
        
    def save_results(self) -> None:
        """Save results to files"""
        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        if self.train_predictions is not None and self.test_predictions is not None:
            # Create training predictions dataframe
            # Use only the records that correspond to actual predictions (after filtering)
            train_ages = [r.age for r in self.train_records][:len(self.y_train)]
            train_living = [r.living_setting for r in self.train_records][:len(self.y_train)]
            
            train_df = pd.DataFrame({
                'dataset': 'train',
                'actual': self.y_train,
                'predicted': self.train_predictions,
                'age': train_ages,
                'living_setting': train_living,
                'error': self.train_predictions - self.y_train
            })
        
            # Create test predictions dataframe (this one is fine)
            test_df = pd.DataFrame({
                'dataset': 'test',
                'actual': self.y_test,
                'predicted': self.test_predictions,
                'age': [r.age for r in self.test_records],
                'living_setting': [r.living_setting for r in self.test_records],
                'error': self.test_predictions - self.y_test
            })
            
            # Combine both datasets
            pred_df = pd.concat([train_df, test_df], ignore_index=True)
            pred_df.to_csv(self.output_dir / "predictions.csv", index=False)
            
            # Combine both datasets
            pred_df = pd.concat([train_df, test_df], ignore_index=True)
            pred_df.to_csv(self.output_dir / "predictions.csv", index=False)
            
        # Log what was saved
        self.logger.info("Results saved:")
        self.logger.info(f"  - Metrics JSON: {self.output_dir_relative / 'metrics.json'}")
        if self.train_predictions is not None and self.test_predictions is not None:
            self.logger.info(f"  - Predictions CSV: {self.output_dir_relative / 'predictions.csv'}")
            self.logger.info(f"    ({len(self.train_predictions):,} train + {len(self.test_predictions):,} test predictions)")
    
    def plot_diagnostics(self) -> None:
        """Generate diagnostic plots (2x3 grid)"""
        if self.test_predictions is None or self.y_test is None:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        
        # 1. Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(self.y_test / 1000, self.test_predictions / 1000, alpha=0.5, s=20)
        max_val = max(self.y_test.max(), self.test_predictions.max()) / 1000
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        ax.set_xlabel('Actual Cost ($1000s)')
        ax.set_ylabel('Predicted Cost ($1000s)')
        ax.set_title(f'Actual vs Predicted\nR^2 = {self.metrics.get("r2_test", 0):.4f}')
        ax.grid(True, alpha=0.3)
        
        # Additional plots omitted for space
        plt.suptitle(f'Model {self.model_id}: {self.model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'diagnostic_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Diagnostic plots saved to {self.output_dir_relative / 'diagnostic_plots.png'}")
        
def generate_coefficient_commands(self, model_number: int, output_path: Path) -> Dict[str, float]:
    """
    Generate LaTeX commands for model coefficients
    
    Args:
        model_number: Model identifier (1, 2, 3, etc.)
        output_path: Directory to save coefficient commands
        
    Returns:
        Dictionary mapping feature names to coefficient values
    """
    if not hasattr(self, 'model') or self.model is None:
        logger.warning(f"Model {model_number}: No fitted model found")
        return {}
    
    # Standard variable name mapping (all models should use this)
    STANDARD_VAR_MAPPING = {
        'LiveILSL': 'LiveILSL',
        'LiveRH1': 'LiveRHOne', 
        'LiveRH2': 'LiveRHTwo',
        'LiveRH3': 'LiveRHThree',
        'LiveRH4': 'LiveRHFour',
        'Age21_30': 'AgeTwentyOneToThirty',
        'Age31Plus': 'AgeThirtyOnePlus',
        'BSum': 'BehavioralSum',
        'FHFSum': 'FHFunctionalSum',
        'SLFSum': 'SLFunctionalSum',
        'SLBSum': 'SLBehavioralSum',
        'Q16': 'QSixteen',
        'Q18': 'QEighteen',
        'Q20': 'QTwenty',
        'Q21': 'QTwentyOne',
        'Q23': 'QTwentyThree',
        'Q28': 'QTwentyEight',
        'Q33': 'QThirtyThree',
        'Q34': 'QThirtyFour',
        'Q36': 'QThirtySix',
        'Q43': 'QFortyThree',
        'Age': 'AgeContinuous',
        # Add more as needed for other models
    }
    
    coef_dict = {}
    commands = []
    
    # Add intercept
    intercept = self.model.intercept_
    cmd_name = f"Model{model_number}Intercept"
    commands.append(f"\\newcommand{{\\{cmd_name}}}{{{intercept:.4f}}}")
    coef_dict['Intercept'] = intercept
    
    # Add each coefficient
    if hasattr(self, 'feature_names'):
        for feat_name, coef_val in zip(self.feature_names, self.model.coef_):
            # Map to standard name
            standard_name = STANDARD_VAR_MAPPING.get(feat_name, feat_name)
            cmd_name = f"Model{model_number}Coef{standard_name}"
            commands.append(f"\\newcommand{{\\{cmd_name}}}{{{coef_val:.4f}}}")
            coef_dict[standard_name] = coef_val
    
    # Write to file
    coef_file = output_path / f"model_{model_number}_coefficients.tex"
    with open(coef_file, 'w') as f:
        f.write(f"% Model {model_number} Coefficient Commands\n")
        f.write("% Generated automatically - do not edit\n\n")
        f.write('\n'.join(commands))
    
    logger.info(f"Model {model_number}: Generated {len(commands)} coefficient commands")
    return coef_dict


@staticmethod
def generate_coefficient_comparison_table(
    model_coefficients: Dict[int, Dict[str, float]],
    output_path: Path,
    include_2015: bool = True
) -> None:
    """
    Generate LaTeX table comparing coefficients across models
    
    Args:
        model_coefficients: Dict mapping model numbers to their coefficient dicts
        output_path: Where to save the table
        include_2015: Whether to include 2015 Model 5b baseline
    """
    
    # Collect all unique variables across all models
    all_vars = set()
    for coef_dict in model_coefficients.values():
        all_vars.update(coef_dict.keys())
    
    # Remove intercept - handle separately
    all_vars.discard('Intercept')
    
    # Define variable order and groupings
    VAR_ORDER = [
        # Living settings
        'LiveILSL', 'LiveRHOne', 'LiveRHTwo', 'LiveRHThree', 'LiveRHFour',
        # Age groups
        'AgeTwentyOneToThirty', 'AgeThirtyOnePlus', 'AgeContinuous',
        # Subscales
        'BehavioralSum', 'FHFunctionalSum', 'SLFunctionalSum', 'SLBehavioralSum',
        # Individual QSI items
        'QSixteen', 'QEighteen', 'QTwenty', 'QTwentyOne', 'QTwentyThree',
        'QTwentyEight', 'QThirtyThree', 'QThirtyFour', 'QThirtySix', 'QFortyThree',
    ]
    
    # Filter to only variables that exist
    ordered_vars = [v for v in VAR_ORDER if v in all_vars]
    # Add any remaining variables not in our order
    ordered_vars.extend(sorted(all_vars - set(ordered_vars)))
    
    # Variable display names
    VAR_LABELS = {
        'LiveILSL': 'Supported/Independent Living',
        'LiveRHOne': 'RH: Standard/Live-In',
        'LiveRHTwo': 'RH: Behavior Focus',
        'LiveRHThree': 'RH: Intensive Behavior',
        'LiveRHFour': 'RH: CTEP/SMHC',
        'AgeTwentyOneToThirty': 'Age 21--30',
        'AgeThirtyOnePlus': 'Age 31+',
        'AgeContinuous': 'Age (continuous)',
        'BehavioralSum': 'Behavioral Status Sum (Q25--30)',
        'FHFunctionalSum': 'FH: Functional Status Sum (Q14--24)',
        'SLFunctionalSum': 'SL: Functional Status Sum (Q14--24)',
        'SLBehavioralSum': 'SL: Behavioral Status Sum (Q25--30)',
        'QSixteen': 'QSI Q16',
        'QEighteen': 'QSI Q18',
        'QTwenty': 'QSI Q20',
        'QTwentyOne': 'QSI Q21',
        'QTwentyThree': 'QSI Q23',
        'QTwentyEight': 'QSI Q28',
        'QThirtyThree': 'QSI Q33',
        'QThirtyFour': 'QSI Q34',
        'QThirtySix': 'QSI Q36',
        'QFortyThree': 'QSI Q43',
    }
    
    # Build LaTeX table
    model_nums = sorted(model_coefficients.keys())
    
    # Column specification
    if include_2015:
        cols = 'l' + 'r' * (len(model_nums) + 1)  # +1 for 2015 column
        header_cols = ['Variable', '2015'] + [f'Model {n}' for n in model_nums]
    else:
        cols = 'l' + 'r' * len(model_nums)
        header_cols = ['Variable'] + [f'Model {n}' for n in model_nums]
    
    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append('\\caption{Coefficient Comparison Across Models}')
    lines.append('\\label{tab:coefficient-comparison}')
    lines.append('\\small')
    lines.append(f'\\begin{{tabular}}{{{cols}}}')
    lines.append('\\toprule')
    lines.append(' & '.join(header_cols) + ' \\\\')
    lines.append('\\midrule')
    
    # Intercept row
    if include_2015:
        row = ['Intercept (base)', '\\ModelFiveBBaseValue']
    else:
        row = ['Intercept (base)']
    
    for model_num in model_nums:
        coefs = model_coefficients[model_num]
        if 'Intercept' in coefs:
            row.append(f'\\Model{model_num}Intercept')
        else:
            row.append('---')
    lines.append(' & '.join(row) + ' \\\\')
    lines.append('\\midrule')
    
    # Variable rows grouped by category
    current_group = None
    for var_name in ordered_vars:
        # Determine group for section headers
        if var_name.startswith('Live'):
            group = 'living'
        elif var_name.startswith('Age'):
            group = 'age'
        elif 'Sum' in var_name:
            group = 'subscales'
        elif var_name.startswith('Q'):
            group = 'qsi'
        else:
            group = 'other'
        
        # Add section header if new group
        if group != current_group:
            if current_group is not None:
                lines.append('\\midrule')
            
            group_labels = {
                'living': '\\multicolumn{' + str(len(header_cols)) + '}{l}{\\textbf{Living Settings}} \\\\',
                'age': '\\multicolumn{' + str(len(header_cols)) + '}{l}{\\textbf{Age Groups}} \\\\',
                'subscales': '\\multicolumn{' + str(len(header_cols)) + '}{l}{\\textbf{QSI Subscale Sums}} \\\\',
                'qsi': '\\multicolumn{' + str(len(header_cols)) + '}{l}{\\textbf{Individual QSI Items}} \\\\',
                'other': '\\multicolumn{' + str(len(header_cols)) + '}{l}{\\textbf{Other Variables}} \\\\',
            }
            lines.append(group_labels.get(group, ''))
            current_group = group
        
        # Build row
        var_label = VAR_LABELS.get(var_name, var_name)
        
        if include_2015:
            # Check if 2015 had this variable
            if var_name == 'AgeContinuous':
                row = [var_label, '---']  # Age not in 2015
            else:
                row = [var_label, f'\\ModelFiveBCoef{var_name}']
        else:
            row = [var_label]
        
        for model_num in model_nums:
            coefs = model_coefficients[model_num]
            if var_name in coefs:
                row.append(f'\\Model{model_num}Coef{var_name}')
            else:
                row.append('---')
        
        lines.append(' & '.join(row) + ' \\\\')
    
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    
    # Write to file
    table_file = output_path / 'coefficient_comparison_table.tex'
    with open(table_file, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Coefficient comparison table written to {table_file}")        