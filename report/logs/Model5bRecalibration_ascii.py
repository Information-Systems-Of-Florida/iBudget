"""
Model 5b Recalibration Script
Calibrates Model 5b parameters using claims data and saves coefficients to JSON
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
import pyodbc
from typing import Dict, Tuple, Optional
import os

# Configure logging with UTF-8 encoding
log_handlers = []

# File handler with UTF-8 encoding
file_handler = logging.FileHandler('model5b_calibration.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
log_handlers.append(file_handler)

# Console handler - check if console supports UTF-8
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
log_handlers.append(console_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

class Model5bCalibrator:
    """Calibrates Model 5b regression parameters"""
    
    def __init__(self, connection_string: str = None):
        """
        Initialize calibrator with database connection
        
        Args:
            connection_string: SQL Server connection string
        """
        # If no connection string provided, read from metadata.py pattern
        if connection_string is None:
            connection_string = self._get_connection_string()
        
        self.conn_string = connection_string
        self.data = None
        self.model = None
        self.coefficients = None
        self.metrics = None
        
    def _get_connection_string(self) -> str:
        """Get connection string using Windows authentication"""
        server = os.getenv('DB_SERVER', '.')  # Local server
        database = os.getenv('DB_NAME', 'APD')
        
        # Use Windows authentication
        return (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection=yes;"
        )
    
    def load_data(self, start_date: str = '2024-07-01', end_date: str = '2025-06-30') -> pd.DataFrame:
        """
        Load calibration data from database
        
        Args:
            start_date: Start date for claims period
            end_date: End date for claims period
            
        Returns:
            DataFrame with calibration data
        """
        logger.info(f"Loading data for period {start_date} to {end_date}")
        logger.info("Connecting to database...")
        
        query = f"""
        EXEC [dbo].[sp_Pull_Model5b_CalibrationData] 
            @StartDate = '{start_date}',
            @EndDate = '{end_date}',
            @Debug = 0
        """
        
        try:
            logger.info("Establishing database connection...")
            with pyodbc.connect(self.conn_string) as conn:
                logger.info("Connection established. Executing query...")
                logger.info("This may take several minutes for large datasets...")
                
                # Read data in chunks if needed
                self.data = pd.read_sql(query, conn)
            
            logger.info(f"[OK] Successfully loaded {len(self.data)} records")
            
            # Log data quality summary
            logger.info("Analyzing data quality...")
            
            # Log records with missing QSI data
            missing_qsi = self.data[self.data['HasMissingQSI'] == 1]
            if len(missing_qsi) > 0:
                logger.warning(f"[WARNING] Found {len(missing_qsi)} records with missing QSI data")
                # Write to separate log file
                missing_qsi[['CaseNo', 'AssessID']].to_csv(
                    'missing_qsi_records.log', 
                    index=False
                )
                logger.info(f"  Missing QSI records saved to missing_qsi_records.log")
            
            # Log records with missing demographics
            missing_demo = self.data[self.data['HasMissingDemographics'] == 1]
            if len(missing_demo) > 0:
                logger.warning(f"[WARNING] Found {len(missing_demo)} records with missing demographics")
            
            # Remove records with missing data
            original_count = len(self.data)
            self.data = self.data[
                (self.data['HasMissingQSI'] == 0) & 
                (self.data['HasMissingDemographics'] == 0)
            ]
            
            removed_count = original_count - len(self.data)
            if removed_count > 0:
                logger.info(f"  Removed {removed_count} incomplete records")
            
            logger.info(f"[OK] Retained {len(self.data)} complete records for calibration")
            
            # Log data distribution
            logger.info("\nData Distribution Summary:")
            logger.info(f"  Claims range: ${self.data['TotalClaims'].min():,.2f} - ${self.data['TotalClaims'].max():,.2f}")
            logger.info(f"  Mean claims: ${self.data['TotalClaims'].mean():,.2f}")
            logger.info(f"  Median claims: ${self.data['TotalClaims'].median():,.2f}")
            
            return self.data
            
        except pyodbc.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            logger.error("Please check:")
            logger.error("  1. SQL Server is running")
            logger.error("  2. Database 'APD' exists")
            logger.error("  3. You have necessary permissions")
            logger.error("  4. Stored procedure 'sp_Pull_Model5b_CalibrationData' exists")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            raise
    
    def validate_data(self) -> bool:
        """
        Validate the loaded data for completeness and quality
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        logger.info("Validating data quality...")
        
        is_valid = True
        
        # Check for required columns
        required_cols = ['CaseNo', 'TotalClaims', 'LivingSettingCategory', 'AgeGroup',
                        'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24',
                        'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30', 'Q33', 'Q34', 'Q36', 'Q43']
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            is_valid = False
        
        # Check for numeric values in QSI columns
        qsi_cols = ['Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24',
                   'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30', 'Q33', 'Q34', 'Q36', 'Q43']
        
        for col in qsi_cols:
            if col in self.data.columns:
                # Try to convert to numeric
                numeric_data = pd.to_numeric(self.data[col], errors='coerce')
                non_numeric_count = numeric_data.isnull().sum() - self.data[col].isnull().sum()
                
                if non_numeric_count > 0:
                    logger.warning(f"Column {col} has {non_numeric_count} non-numeric values")
                    # Show sample of non-numeric values
                    non_numeric_mask = numeric_data.isnull() & self.data[col].notnull()
                    if non_numeric_mask.any():
                        sample_values = self.data.loc[non_numeric_mask, col].head(5).tolist()
                        logger.warning(f"  Sample non-numeric values in {col}: {sample_values}")
        
        # Check claims data
        if 'TotalClaims' in self.data.columns:
            negative_claims = (self.data['TotalClaims'] < 0).sum()
            if negative_claims > 0:
                logger.warning(f"Found {negative_claims} records with negative claims")
            
            zero_claims = (self.data['TotalClaims'] == 0).sum()
            if zero_claims > 0:
                logger.info(f"Found {zero_claims} records with zero claims")
        
        # Check living setting distribution
        if 'LivingSettingCategory' in self.data.columns:
            living_dist = self.data['LivingSettingCategory'].value_counts()
            logger.info("Living Setting Distribution:")
            for setting, count in living_dist.items():
                logger.info(f"  {setting}: {count}")
            
            if 'OTHER' in living_dist and living_dist['OTHER'] > len(self.data) * 0.1:
                logger.warning(f"Large number of 'OTHER' living settings: {living_dist['OTHER']}")
        
        return is_valid
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target variable
        
        Returns:
            X: Feature matrix
            y: Target variable (square root of claims)
        """
        logger.info("Preparing features for Model 5b")
        
        df = self.data.copy()
        
        # Create target variable (square root of claims)
        df['sqrt_claims'] = np.sqrt(df['TotalClaims'].clip(lower=0))
        
        # Create age group indicators
        df['Age21_30'] = (df['AgeGroup'] == 'Age21-30').astype(int)
        df['Age31_plus'] = (df['AgeGroup'] == 'Age31+').astype(int)
        
        # Create living setting indicators
        df['LiveILSL'] = (df['LivingSettingCategory'] == 'ILSL').astype(int)
        df['LiveRH1'] = (df['LivingSettingCategory'] == 'RH1').astype(int)
        df['LiveRH2'] = (df['LivingSettingCategory'] == 'RH2').astype(int)
        df['LiveRH3'] = (df['LivingSettingCategory'] == 'RH3').astype(int)
        df['LiveRH4'] = (df['LivingSettingCategory'] == 'RH4').astype(int)
        
        # Convert QSI questions to numeric and handle missing values
        qsi_questions = ['Q16', 'Q18', 'Q20', 'Q21', 'Q23', 'Q28', 'Q33', 'Q34', 'Q36', 'Q43']
        
        for col in qsi_questions:
            if col in df.columns:
                # Convert to numeric, replacing non-numeric with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Replace NaN with 0 (assuming 0 means no support needed)
                df[col] = df[col].fillna(0)
            else:
                logger.warning(f"Column {col} not found in data, setting to 0")
                df[col] = 0
        
        # Calculate behavioral sum (Q25-Q30) - already done in SP but verify
        behavioral_questions = ['Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30']
        for col in behavioral_questions:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if 'BSum' not in df.columns or df['BSum'].isnull().any():
            df['BSum'] = df[behavioral_questions].sum(axis=1)
        else:
            df['BSum'] = pd.to_numeric(df['BSum'], errors='coerce').fillna(0)
        
        # Calculate functional sum (Q14-Q24) - already done in SP but verify
        functional_questions = ['Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 
                               'Q20', 'Q21', 'Q22', 'Q23', 'Q24']
        for col in functional_questions:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if 'FSum' not in df.columns or df['FSum'].isnull().any():
            df['FSum'] = df[functional_questions].sum(axis=1)
        else:
            df['FSum'] = pd.to_numeric(df['FSum'], errors='coerce').fillna(0)
        
        # Create interaction terms
        df['FHFSum'] = ((df['LivingSettingCategory'] == 'FH').astype(int) * df['FSum'])
        df['SLFSum'] = (df['LiveILSL'] * df['FSum'])
        df['SLBSum'] = (df['LiveILSL'] * df['BSum'])
        
        # Select features following Model 5b specification
        feature_columns = [
            'LiveILSL', 'LiveRH1', 'LiveRH2', 'LiveRH3', 'LiveRH4',
            'Age21_30', 'Age31_plus',
            'BSum', 'FHFSum', 'SLFSum', 'SLBSum',
            'Q16', 'Q18', 'Q20', 'Q21', 'Q23', 'Q28', 
            'Q33', 'Q34', 'Q36', 'Q43'
        ]
        
        X = df[feature_columns].copy()
        y = df['sqrt_claims']
        
        # Final check for NaN values
        if X.isnull().any().any():
            nan_counts = X.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0]
            logger.warning(f"Found NaN values in features: {nan_cols.to_dict()}")
            logger.info("Filling remaining NaN values with 0")
            X = X.fillna(0)
        
        # Verify no NaN values remain
        assert not X.isnull().any().any(), "Feature matrix still contains NaN values"
        assert not y.isnull().any(), "Target variable contains NaN values"
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target variable shape: {y.shape}")
        logger.info(f"Feature summary:")
        logger.info(f"  Living settings: ILSL={X['LiveILSL'].sum()}, RH1={X['LiveRH1'].sum()}, "
                   f"RH2={X['LiveRH2'].sum()}, RH3={X['LiveRH3'].sum()}, RH4={X['LiveRH4'].sum()}")
        logger.info(f"  Age groups: 21-30={X['Age21_30'].sum()}, 31+={X['Age31_plus'].sum()}")
        logger.info(f"  Mean BSum: {X['BSum'].mean():.2f}, Mean FSum: {df['FSum'].mean():.2f}")
        
        return X, y, df
    
    def remove_outliers(self, X: pd.DataFrame, y: pd.Series, 
                       outlier_percent: float = 9.40,
                       method: str = 'percentile') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove outliers from the data
        
        Args:
            X: Feature matrix
            y: Target variable
            outlier_percent: Percentage of outliers to remove
            method: Method for outlier detection ('percentile', 'iqr', 'zscore', 'cook')
            
        Returns:
            X_clean: Feature matrix without outliers
            y_clean: Target variable without outliers
        """
        logger.info(f"Removing outliers using {method} method ({outlier_percent}%)")
        
        # Ensure no NaN values before outlier detection
        if X.isnull().any().any() or y.isnull().any():
            logger.warning("Found NaN values before outlier removal, dropping rows with NaN")
            mask_no_nan = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask_no_nan]
            y = y[mask_no_nan]
            logger.info(f"Dropped {(~mask_no_nan).sum()} rows with NaN values")
        
        n_original = len(y)
        
        if method == 'percentile':
            # Remove top and bottom percentiles based on residuals from initial fit
            initial_model = LinearRegression()
            
            # Check for NaN again before fitting
            if X.isnull().any().any():
                logger.error("Feature matrix still contains NaN after cleaning")
                logger.info("NaN counts by column:")
                for col in X.columns:
                    nan_count = X[col].isnull().sum()
                    if nan_count > 0:
                        logger.info(f"  {col}: {nan_count} NaN values")
                raise ValueError("Cannot fit model with NaN values")
            
            initial_model.fit(X, y)
            residuals = y - initial_model.predict(X)
            
            lower_bound = np.percentile(residuals, outlier_percent/2)
            upper_bound = np.percentile(residuals, 100 - outlier_percent/2)
            
            mask = (residuals >= lower_bound) & (residuals <= upper_bound)
            
        elif method == 'iqr':
            # Interquartile range method
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (y >= lower_bound) & (y <= upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(y))
            threshold = stats.norm.ppf(1 - outlier_percent/200)  # Two-tailed
            mask = z_scores < threshold
            
        elif method == 'cook':
            # Cook's distance method
            initial_model = LinearRegression()
            initial_model.fit(X, y)
            
            # Calculate Cook's distance
            n = len(X)
            p = X.shape[1]
            
            predictions = initial_model.predict(X)
            residuals = y - predictions
            
            # Leverage scores
            H = X @ np.linalg.inv(X.T @ X) @ X.T
            leverage = np.diag(H)
            
            # Cook's distance
            mse = np.mean(residuals ** 2)
            cooks_d = (residuals ** 2) / (p * mse) * leverage / (1 - leverage) ** 2
            
            threshold = np.percentile(cooks_d, 100 - outlier_percent)
            mask = cooks_d < threshold
            
        else:
            raise ValueError(f"Unknown outlier removal method: {method}")
        
        X_clean = X[mask]
        y_clean = y[mask]
        
        n_removed = n_original - len(y_clean)
        percent_removed = (n_removed / n_original) * 100
        
        logger.info(f"Removed {n_removed} outliers ({percent_removed:.2f}%)")
        
        # Final check for NaN
        if X_clean.isnull().any().any() or y_clean.isnull().any():
            logger.error("NaN values remain after outlier removal")
            raise ValueError("Data contains NaN values after outlier removal")
        
        return X_clean, y_clean, mask
    
    def calibrate(self, outlier_percent: float = 9.40, 
                 outlier_method: str = 'percentile') -> Dict:
        """
        Calibrate the model
        
        Args:
            outlier_percent: Percentage of outliers to remove
            outlier_method: Method for outlier detection
            
        Returns:
            Dictionary with calibration results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("="*60)
        logger.info("Starting model calibration")
        logger.info("="*60)
        
        # Step 1: Prepare features
        logger.info("\nStep 1: Preparing features...")
        X, y, df = self.prepare_features()
        logger.info(f"[OK] Features prepared successfully")
        
        # Step 2: Remove outliers
        logger.info(f"\nStep 2: Removing outliers ({outlier_method} method, {outlier_percent}%)...")
        X_clean, y_clean, mask = self.remove_outliers(
            X, y, outlier_percent, outlier_method
        )
        logger.info(f"[OK] Outliers removed successfully")
        
        # Step 3: Fit the model
        logger.info("\nStep 3: Fitting regression model...")
        logger.info(f"  Training on {len(X_clean)} samples with {X_clean.shape[1]} features")
        self.model = LinearRegression()
        self.model.fit(X_clean, y_clean)
        logger.info(f"[OK] Model fitted successfully")
        
        # Step 4: Calculate metrics
        logger.info("\nStep 4: Calculating model metrics...")
        y_pred = self.model.predict(X_clean)
        
        # Calculate metrics
        r2 = r2_score(y_clean, y_pred)
        rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
        mae = mean_absolute_error(y_clean, y_pred)
        
        # Calculate residual standard error
        n = len(X_clean)
        p = X_clean.shape[1]
        residuals = y_clean - y_pred
        rse = np.sqrt(np.sum(residuals ** 2) / (n - p - 1))
        
        logger.info(f"[OK] Metrics calculated")
        logger.info(f"\nModel Performance:")
        logger.info(f"  R-squared: {r2:.4f}")
        logger.info(f"  Adjusted R-squared: {1 - (1 - r2) * (n - 1) / (n - p - 1):.4f}")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  Residual Std Error: {rse:.2f}")
        
        # Store coefficients
        feature_names = X.columns.tolist()
        self.coefficients = {
            'intercept': float(self.model.intercept_),
            'features': {
                name: float(coef) 
                for name, coef in zip(feature_names, self.model.coef_)
            }
        }
        
        # Store metrics
        self.metrics = {
            'r_squared': float(r2),
            'adjusted_r_squared': float(1 - (1 - r2) * (n - 1) / (n - p - 1)),
            'rmse': float(rmse),
            'mae': float(mae),
            'residual_standard_error': float(rse),
            'n_samples': int(n),
            'n_features': int(p),
            'outlier_percent': float(outlier_percent),
            'outlier_method': outlier_method
        }
        
        # Calculate SBC (Schwarz Bayesian Criterion) / BIC
        bic = n * np.log(np.sum(residuals ** 2) / n) + p * np.log(n)
        self.metrics['bic'] = float(bic)
        
        logger.info(f"\n[OK] Model calibration complete!")
        
        return {
            'coefficients': self.coefficients,
            'metrics': self.metrics,
            'calibration_date': datetime.now().isoformat()
        }
    
    def _bootstrap_confidence_intervals(self, X, y, n_iterations=1000, alpha=0.05):
        """Calculate bootstrap confidence intervals for coefficients"""
        logger.info(f"Calculating bootstrap confidence intervals ({n_iterations} iterations)")
        
        n = len(X)
        feature_names = ['intercept'] + X.columns.tolist()
        bootstrap_coefs = {name: [] for name in feature_names}
        
        for i in range(n_iterations):
            # Resample with replacement
            indices = np.random.choice(n, n, replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # Fit model
            model = LinearRegression()
            model.fit(X_boot, y_boot)
            
            # Store coefficients
            bootstrap_coefs['intercept'].append(model.intercept_)
            for j, name in enumerate(X.columns):
                bootstrap_coefs[name].append(model.coef_[j])
        
        # Calculate confidence intervals
        ci_lower = (alpha / 2) * 100
        ci_upper = (1 - alpha / 2) * 100
        
        self.coefficients['confidence_intervals'] = {}
        for name, values in bootstrap_coefs.items():
            self.coefficients['confidence_intervals'][name] = {
                'lower': float(np.percentile(values, ci_lower)),
                'upper': float(np.percentile(values, ci_upper))
            }
        
        logger.info("Bootstrap confidence intervals calculated")
    
    def save_coefficients(self, filename: str = 'Model5bRecal.json'):
        """
        Save calibrated coefficients to JSON file
        
        Args:
            filename: Output filename
        """
        if self.coefficients is None:
            raise ValueError("No coefficients to save. Run calibrate() first.")
        
        output = {
            'model_name': 'Model5b_Recalibrated',
            'coefficients': self.coefficients,
            'metrics': self.metrics,
            'calibration_date': datetime.now().isoformat(),
            'data_period': {
                'start_date': '2024-07-01',
                'end_date': '2025-06-30'
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Coefficients saved to {filename}")
    
    def compare_with_original(self):
        """Compare recalibrated coefficients with original Model 5b coefficients"""
        
        # Original Model 5b coefficients from the documentation
        original = {
            'intercept': 27.5720,
            'LiveILSL': 35.8220,
            'LiveRH1': 90.6294,
            'LiveRH2': 131.7576,
            'LiveRH3': 209.4558,
            'LiveRH4': 267.0995,
            'Age21_30': 47.8473,
            'Age31_plus': 48.9634,
            'BSum': 0.4954,
            'FHFSum': 0.6349,
            'SLFSum': 2.0529,
            'SLBSum': 1.4501,
            'Q16': 2.4984,
            'Q18': 5.8537,
            'Q20': 2.6772,
            'Q21': 2.7878,
            'Q23': 6.3555,
            'Q28': 2.2803,
            'Q33': 1.2233,
            'Q34': 2.1764,
            'Q36': 2.6734,
            'Q43': 1.9304
        }
        
        comparison = pd.DataFrame({
            'Original': original,
            'Recalibrated': [self.coefficients['intercept']] + 
                          [self.coefficients['features'].get(k, 0) 
                           for k in list(original.keys())[1:]]
        }, index=list(original.keys()))
        
        comparison['Difference'] = comparison['Recalibrated'] - comparison['Original']
        comparison['Pct_Change'] = (comparison['Difference'] / comparison['Original']) * 100
        
        logger.info("\nCoefficient Comparison:\n" + comparison.to_string())
        
        return comparison


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print(" Model 5b Recalibration Process")
    print("="*70)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Initialize calibrator
    logger.info("Initializing Model5b Calibrator...")
    
    try:
        calibrator = Model5bCalibrator()
        logger.info("[OK] Calibrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize calibrator: {str(e)}")
        return
    
    try:
        # Load data
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: DATA LOADING")
        logger.info("="*60)
        calibrator.load_data(start_date='2024-07-01', end_date='2025-06-30')
        
        # Validate data
        if not calibrator.validate_data():
            logger.warning("Data validation found issues - proceeding with caution")
        
        # Check if we have enough data
        if len(calibrator.data) < 100:
            logger.error(f"Insufficient data for calibration. Only {len(calibrator.data)} records found.")
            logger.error("Model calibration requires at least 100 complete records.")
            return
        
        # Calibrate model with different outlier methods
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: MODEL CALIBRATION")
        logger.info("="*60)
        
        methods = ['percentile', 'iqr']  # Can also try 'zscore', 'cook'
        
        best_r2 = 0
        best_method = None
        best_results = None
        
        for i, method in enumerate(methods, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing outlier method {i}/{len(methods)}: {method}")
            logger.info(f"{'='*50}")
            
            try:
                results = calibrator.calibrate(
                    outlier_percent=9.40,
                    outlier_method=method
                )
                
                if results['metrics']['r_squared'] > best_r2:
                    best_r2 = results['metrics']['r_squared']
                    best_method = method
                    best_results = results
                
                logger.info(f"[OK] R-squared for {method}: {results['metrics']['r_squared']:.4f}")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to calibrate with {method} method: {str(e)}")
                continue
        
        if best_results is None:
            logger.error("Failed to calibrate model with any method")
            return
        
        # Use best method for final calibration
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 3: FINALIZING RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Best method: {best_method} with R-squared = {best_r2:.4f}")
        
        calibrator.coefficients = best_results['coefficients']
        calibrator.metrics = best_results['metrics']
        
        # Save coefficients
        logger.info("\nSaving calibration results...")
        calibrator.save_coefficients('Model5bRecal.json')
        logger.info("[OK] Coefficients saved to Model5bRecal.json")
        
        # Compare with original
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: MODEL COMPARISON")
        logger.info("="*60)
        comparison = calibrator.compare_with_original()
        
        # Final summary
        print("\n" + "="*70)
        print(" CALIBRATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f" End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Model R-squared: {best_r2:.4f}")
        print(f" Samples Used: {calibrator.metrics['n_samples']:,}")
        print(f" Output File: Model5bRecal.json")
        print("="*70 + "\n")
        
        logger.info("[OK] Calibration completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("\n\nProcess interrupted by user")
        return
    except Exception as e:
        logger.error(f"\n[ERROR] Critical error during calibration: {str(e)}")
        logger.error("Please check the log file for details")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
