"""
base_model.py
=============
Simplified Base Class for iBudget Models

FIXED: Type conversion issues for ConsumerRecord creation
- ConsumerID: int → str
- BLEVEL/FLEVEL/PLEVEL: str → float  
- QSI questions: None → 0.0
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
    - prepare_features(records) → (X, feature_names)
    - _fit_core(X, y) → fit model
    - _predict_core(X) → predict in fitted scale
    """
    
    def __init__(self,
                 model_id: int,
                 model_name: str,
                 use_outlier_removal: bool = False,
                 outlier_threshold: float = 1.645,
                 transformation: str = 'none',
                 random_seed: int = 42):
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
        self.subgroup_metrics: Dict[str, Dict[str, float]] = {}
        self.variance_metrics: Dict[str, float] = {}
        self.population_scenarios: Dict[str, Dict[str, float]] = {}
        
        # Output directory
        #self.output_dir = Path(f"report/models/model_{model_id}")
        #self.output_dir.mkdir(parents=True, exist_ok=True)
        # Make path relative to script location
        script_dir = Path(__file__).parent
        self.output_dir = script_dir / "../../report/models" / f"model_{model_id}"
        self.output_dir = self.output_dir.resolve()  # Convert to absolute path
        self.output_dir.mkdir(parents=True, exist_ok=True)        
        
        # Set up logging
        self._setup_logging()
        
        # Log initialization
        self.log_section(f"INITIALIZING MODEL {self.model_id}: {self.model_name.upper()}", "=")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Outlier removal: {use_outlier_removal}")
        if use_outlier_removal:
            self.logger.info(f"  - Outlier threshold: ±{outlier_threshold} (studentized residuals)")
        self.logger.info(f"  - Transformation: {transformation}")
        self.logger.info(f"  - Random seed: {random_seed}")
    
    def _setup_logging(self):
        """Set up model-specific logging"""
        log_dir = Path("report/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_filename = log_dir / f"model_{self.model_id}_log.txt"
        
        self.logger = logging.getLogger(f"MODEL_{self.model_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
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
            self.logger.info(f"Training R²: {self.metrics.get('r2_train', 0):.4f}")
            self.logger.info(f"Test R²: {self.metrics.get('r2_test', 0):.4f}")
            self.logger.info(f"RMSE: ${self.metrics.get('rmse_test', 0):,.2f}")
            self.logger.info(f"MAE: ${self.metrics.get('mae_test', 0):,.2f}")
            self.logger.info(f"MAPE: {self.metrics.get('mape_test', 0):.2f}%")
            
            if 'cv_mean' in self.metrics:
                self.logger.info(f"CV R² (mean ± std): {self.metrics['cv_mean']:.4f} ± "
                               f"{self.metrics.get('cv_std', 0):.4f}")
        
        self.logger.info("-" * 60)
    
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
    # TEMPLATE METHODS
    # ========================================================================
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model with preprocessing (template method)"""
        # Remove outliers if enabled
        if self.use_outlier_removal:
            X, y, self.outlier_diagnostics = self.remove_outliers_studentized(X, y)
        
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
            y_fold_pred_fitted = self._predict_core(X_fold_val)
            
            # Inverse transform
            y_fold_pred_original = self.inverse_transformation(y_fold_pred_fitted)
            y_fold_pred_original = np.maximum(0, y_fold_pred_original)
            
            # Evaluate on original scale
            score = r2_score(y_fold_val_original, y_fold_pred_original)
            cv_scores.append(score)
            
            self.logger.info(f"  Fold {fold}: R² = {score:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        self.logger.info(f"Mean R²: {cv_mean:.4f}")
        self.logger.info(f"Std R²: {cv_std:.4f}")
        
        return {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'scores': cv_scores
        }
    
    # ========================================================================
    # METRICS CALCULATION
    # ========================================================================
    
    def calculate_metrics(self) -> Dict[str, float]:
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
        
        # Cross-validation
        if perform_cv:
            cv_results = self.perform_cross_validation(n_splits=n_cv_folds)
            self.metrics['cv_mean'] = cv_results.get('cv_mean', 0)
            self.metrics['cv_std'] = cv_results.get('cv_std', 0)
        
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
        
        print(f"DEBUG: metrics keys = {list(self.metrics.keys())}")
        print(f"DEBUG: subgroup_metrics keys = {list(self.subgroup_metrics.keys())}")
        print(f"DEBUG: variance_metrics keys = {list(self.variance_metrics.keys())}")
        print(f"DEBUG: population_scenarios keys = {list(self.population_scenarios.keys())}")
        
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
        print(f"DEBUG: Starting LaTeX generation for model {self.model_id}") 
        
        model_word = self._number_to_word(self.model_id)
        newcommands_file = self.output_dir / f"model_{self.model_id}_newcommands.tex"
        renewcommands_file = self.output_dir / f"model_{self.model_id}_renewcommands.tex"
        
        print(f"DEBUG: Writing to {renewcommands_file}")
        
        # ========================================================================
        # NEWCOMMANDS (Placeholders)
        # ========================================================================
        with open(newcommands_file, 'w') as f:
            f.write(f"% Model {self.model_id} LaTeX Commands (Placeholders)\n")
            f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic performance metrics
            print("DEBUG: Writing basic metrics newcommands")
            for metric in ['RSquaredTrain', 'RSquaredTest', 'RMSETrain', 'RMSETest',
                        'MAETrain', 'MAETest', 'MAPETrain', 'MAPETest',
                        'CVMean', 'CVStd', 'CVCILower', 'CVCIUpper',
                        'TrainingSamples', 'TestSamples']:
                f.write(f"\\newcommand{{\\Model{model_word}{metric}}}{{\\WarningRunPipeline}}\n")
            
            # Accuracy bands
            print("DEBUG: Accuracy bands")
            for threshold in ['OneK', 'TwoK', 'FiveK', 'TenK', 'TwentyK']:
                f.write(f"\\newcommand{{\\Model{model_word}Within{threshold}}}{{\\WarningRunPipeline}}\n")
            
            # Subgroup: Living settings
            print("DEBUG: Subgroup: Living settings")
            for setting in ['FH', 'ILSL', 'RHOneFour']:
                for metric in ['N', 'RSquared', 'RMSE', 'Bias']:
                    f.write(f"\\newcommand{{\\Model{model_word}SubgroupLiving{setting}{metric}}}{{\\WarningRunPipeline}}\n")
            
            # Subgroup: Age groups
            print("DEBUG: Subgroup: Age groups")
            for age in ['AgeUnderTwentyOne', 'AgeTwentyOneToThirty', 'AgeThirtyOnePlus']:
                for metric in ['N', 'RSquared', 'RMSE', 'Bias']:
                    f.write(f"\\newcommand{{\\Model{model_word}SubgroupAge{age}{metric}}}{{\\WarningRunPipeline}}\n")
            
            # Subgroup: Cost quartiles
            print("DEBUG: Subgroup: Cost quartiles")
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
            else:
                # No outlier removal - set to N/A
                f.write(f"\n% Outlier Diagnostics (not used)\n")
                f.write(f"\\renewcommand{{\\Model{model_word}StudentizedResidualsMean}}{{N/A}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}StudentizedResidualsStd}}{{N/A}}\n")
                f.write(f"\\renewcommand{{\\Model{model_word}PctWithinThreshold}}{{N/A}}\n")
            
            # Number of features
            f.write(f"\n% Model Configuration\n")
            f.write(f"\\renewcommand{{\\Model{model_word}NumFeatures}}{{{len(self.feature_names)}}}\n")
        
                
        self.logger.info("LaTeX commands generated")
    
        
    def save_results(self) -> None:
        """Save results to files"""
        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        if self.test_predictions is not None:
            pred_df = pd.DataFrame({
                'actual': self.y_test,
                'predicted': self.test_predictions,
                'error': self.test_predictions - self.y_test
            })
            pred_df.to_csv(self.output_dir / "predictions.csv", index=False)
        
        self.logger.info("Results saved")
    
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
        ax.set_title(f'Actual vs Predicted\nR² = {self.metrics.get("r2_test", 0):.4f}')
        ax.grid(True, alpha=0.3)
        
        # Additional plots omitted for space
        plt.suptitle(f'Model {self.model_id}: {self.model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'diagnostic_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Diagnostic plots saved")