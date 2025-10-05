"""
Model 5b Prediction Script
Calculates predictions for individuals using recalibrated Model 5b and saves to database
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import pyodbc
from typing import Dict, Optional
import os

# Configure logging with UTF-8 encoding
log_handlers = []

# File handler with UTF-8 encoding
file_handler = logging.FileHandler('model5b_predictions.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
log_handlers.append(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
log_handlers.append(console_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)


class Model5bPredictor:
    """Generates predictions using recalibrated Model 5b coefficients"""
    
    def __init__(self, connection_string: str = None, coefficients_file: str = 'Model5bRecal.json'):
        """
        Initialize predictor with database connection and coefficients
        
        Args:
            connection_string: SQL Server connection string
            coefficients_file: JSON file with calibrated coefficients
        """
        if connection_string is None:
            connection_string = self._get_connection_string()
        
        self.conn_string = connection_string
        self.coefficients_file = coefficients_file
        self.coefficients = None
        self.model_version = None
        self.run_id = None
        
        # Load coefficients
        self.load_coefficients()
    
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
    
    def load_coefficients(self):
        """Load calibrated coefficients from JSON file"""
        
        # Check if coefficients file exists
        if not os.path.exists(self.coefficients_file):
            error_msg = (
                f"\n{'='*70}\n"
                f" ERROR: Coefficients file '{self.coefficients_file}' not found!\n"
                f"{'='*70}\n\n"
                f"The Model5b prediction program requires calibrated coefficients.\n\n"
                f"Please run the calibration program first:\n"
                f"  python calibrate_model5b.py\n\n"
                f"This will create the '{self.coefficients_file}' file needed for predictions.\n"
                f"{'='*70}\n"
            )
            logger.error(error_msg)
            raise FileNotFoundError(f"Coefficients file {self.coefficients_file} not found. Please run calibration first.")
        
        try:
            logger.info(f"Loading coefficients from {self.coefficients_file}")
            with open(self.coefficients_file, 'r') as f:
                data = json.load(f)
            
            self.coefficients = data['coefficients']
            self.model_version = f"Model5b_Recal_{data['calibration_date'][:10]}"
            
            logger.info(f"[OK] Loaded coefficients successfully")
            logger.info(f"  Model version: {self.model_version}")
            logger.info(f"  Number of features: {len(self.coefficients['features'])}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {str(e)}")
            logger.error("The coefficients file may be corrupted. Please re-run calibration.")
            raise
        except KeyError as e:
            logger.error(f"Missing required field in coefficients file: {str(e)}")
            logger.error("The coefficients file format may be incorrect. Please re-run calibration.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading coefficients: {str(e)}")
            raise
    
    def get_latest_qsi_data(self) -> pd.DataFrame:
        """
        Get latest QSI assessment data for all consumers
        
        Returns:
            DataFrame with QSI data
        """
        query = """
        WITH LatestQSI AS (
            SELECT 
                q.CaseNo,
                q.AssessID,
                q.ReviewDate,
                ROW_NUMBER() OVER (PARTITION BY q.CaseNo ORDER BY q.ReviewDate DESC, q.AssessID DESC) as RowNum
            FROM [dbo].[tbl_QSIAssessments] q
        )
        SELECT 
            q.CaseNo,
            q.AssessID,
            q.ReviewDate,
            c.DOB,
            DATEDIFF(YEAR, c.DOB, GETDATE()) AS CurrentAge,
            c.RESIDENCETYPE,
            -- QSI Questions needed for Model 5b
            q.Q14, q.Q15, q.Q16, q.Q17, q.Q18, q.Q19, q.Q20,
            q.Q21, q.Q22, q.Q23, q.Q24, q.Q25, q.Q26, q.Q27,
            q.Q28, q.Q29, q.Q30, q.Q33, q.Q34, q.Q36, q.Q43
        FROM LatestQSI lq
        INNER JOIN [dbo].[tbl_QSIAssessments] q ON lq.CaseNo = q.CaseNo AND lq.AssessID = q.AssessID
        INNER JOIN [dbo].[tbl_Consumers] c ON q.CaseNo = c.CASENO
        WHERE lq.RowNum = 1
        """
        
        logger.info("Loading QSI assessment data")
        
        try:
            with pyodbc.connect(self.conn_string) as conn:
                data = pd.read_sql(query, conn)
            
            logger.info(f"Loaded {len(data)} QSI assessments")
            return data
            
        except Exception as e:
            logger.error(f"Error loading QSI data: {str(e)}")
            raise
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for prediction
        
        Args:
            df: Raw data from database
            
        Returns:
            DataFrame with prepared features
        """
        logger.info("Preparing prediction data")
        
        # Create age groups
        df['AgeGroup'] = pd.cut(
            df['CurrentAge'],
            bins=[-np.inf, 20, 30, np.inf],
            labels=['Under21', 'Age21-30', 'Age31+']
        )
        df['Age21_30'] = (df['AgeGroup'] == 'Age21-30').astype(int)
        df['Age31_plus'] = (df['AgeGroup'] == 'Age31+').astype(int)
        
        # Map living settings
        df['LivingSettingCategory'] = df['RESIDENCETYPE'].apply(self._map_living_setting)
        
        # Create living setting indicators
        df['LiveILSL'] = (df['LivingSettingCategory'] == 'ILSL').astype(int)
        df['LiveRH1'] = (df['LivingSettingCategory'] == 'RH1').astype(int)
        df['LiveRH2'] = (df['LivingSettingCategory'] == 'RH2').astype(int)
        df['LiveRH3'] = (df['LivingSettingCategory'] == 'RH3').astype(int)
        df['LiveRH4'] = (df['LivingSettingCategory'] == 'RH4').astype(int)
        
        # Convert QSI questions to numeric
        qsi_cols = [f'Q{i}' for i in range(14, 44)]
        for col in qsi_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate sums
        df['BSum'] = df[['Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30']].sum(axis=1)
        df['FSum'] = df[['Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 
                         'Q20', 'Q21', 'Q22', 'Q23', 'Q24']].sum(axis=1)
        
        # Calculate interaction terms
        df['FHFSum'] = ((df['LivingSettingCategory'] == 'FH').astype(int) * df['FSum'])
        df['SLFSum'] = (df['LiveILSL'] * df['FSum'])
        df['SLBSum'] = (df['LiveILSL'] * df['BSum'])
        
        # Check for missing values in required fields
        required_cols = ['Q16', 'Q18', 'Q20', 'Q21', 'Q23', 'Q28', 'Q33', 'Q34', 'Q36', 'Q43']
        df['HasMissingData'] = df[required_cols].isnull().any(axis=1)
        
        # Log records with missing data
        missing_data = df[df['HasMissingData']]
        if len(missing_data) > 0:
            logger.warning(f"Found {len(missing_data)} records with missing QSI data")
            missing_data[['CaseNo', 'AssessID']].to_csv('prediction_missing_data.log', index=False)
        
        return df
    
    def _map_living_setting(self, residence_type: str) -> str:
        """Map residence type to living setting category"""
        if pd.isna(residence_type):
            return 'OTHER'
        
        residence_type = str(residence_type).upper()
        
        if 'FAMILY HOME' in residence_type:
            return 'FH'
        elif 'INDEPENDENT LIVING' in residence_type or 'SUPPORTED LIVING' in residence_type:
            return 'ILSL'
        elif ('SMALL GROUP HOME' in residence_type or 
              'FOSTER HOME' in residence_type or 
              'ADULT FAMILY CARE HOME' in residence_type):
            return 'RH1'
        elif 'LARGE GROUP HOME' in residence_type:
            return 'RH2'
        elif ('ICF/DD' in residence_type or 
              'ASSISTED LIVING' in residence_type or 
              'NURSING HOME' in residence_type):
            return 'RH3'
        elif ('DEVELOPMENTAL DISABILITIES CENTER' in residence_type or
              'RESIDENTIAL HABILITATION CENTER' in residence_type or
              'HOSPITAL' in residence_type or
              'DEFENDANT PROGRAM' in residence_type or
              'COMMITMENT FACILITY' in residence_type or
              'MENTAL HEALTH' in residence_type):
            return 'RH4'
        else:
            return 'OTHER'
    
    def calculate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Model 5b predictions
        
        Args:
            df: Prepared data
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Calculating predictions")
        
        # Filter out records with missing data
        valid_df = df[~df['HasMissingData']].copy()
        
        # Initialize prediction with intercept
        valid_df['SqrtPrediction'] = self.coefficients['intercept']
        
        # Add feature contributions
        feature_mapping = {
            'LiveILSL': 'LiveILSL',
            'LiveRH1': 'LiveRH1',
            'LiveRH2': 'LiveRH2',
            'LiveRH3': 'LiveRH3',
            'LiveRH4': 'LiveRH4',
            'Age21_30': 'Age21_30',
            'Age31_plus': 'Age31_plus',
            'BSum': 'BSum',
            'FHFSum': 'FHFSum',
            'SLFSum': 'SLFSum',
            'SLBSum': 'SLBSum',
            'Q16': 'Q16',
            'Q18': 'Q18',
            'Q20': 'Q20',
            'Q21': 'Q21',
            'Q23': 'Q23',
            'Q28': 'Q28',
            'Q33': 'Q33',
            'Q34': 'Q34',
            'Q36': 'Q36',
            'Q43': 'Q43'
        }
        
        for feature_name, col_name in feature_mapping.items():
            if feature_name in self.coefficients['features']:
                coef = self.coefficients['features'][feature_name]
                valid_df['SqrtPrediction'] += coef * valid_df[col_name]
        
        # Calculate final prediction (square of sqrt prediction)
        valid_df['Prediction'] = valid_df['SqrtPrediction'] ** 2
        valid_df['PredictedBudget'] = valid_df['Prediction'].round(2)
        
        logger.info(f"Calculated predictions for {len(valid_df)} records")
        
        # Add metadata
        valid_df['CalculationDate'] = datetime.now()
        valid_df['ModelVersion'] = self.model_version
        valid_df['ModelCoefficients'] = json.dumps(self.coefficients)
        
        return valid_df
    
    def create_run_id(self) -> int:
        """Create a new run ID for this batch of predictions"""
        query = """
        SELECT ISNULL(MAX(RunID), 0) + 1 as NextRunID 
        FROM [dbo].[tbl_Model5b_Predictions]
        """
        
        try:
            with pyodbc.connect(self.conn_string) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                self.run_id = cursor.fetchone()[0]
            
            logger.info(f"Created RunID: {self.run_id}")
            return self.run_id
            
        except Exception as e:
            logger.error(f"Error creating RunID: {str(e)}")
            self.run_id = 1
            return self.run_id
    
    def save_to_database(self, predictions: pd.DataFrame):
        """
        Save predictions to database
        
        Args:
            predictions: DataFrame with predictions
        """
        logger.info("Saving predictions to database")
        
        # Create run ID
        self.create_run_id()
        
        # Prepare data for insertion
        insert_data = predictions[[
            'CaseNo', 'AssessID', 'ReviewDate', 'LivingSettingCategory',
            'CurrentAge', 'AgeGroup', 'BSum', 'FSum', 
            'FHFSum', 'SLFSum', 'SLBSum',
            'SqrtPrediction', 'PredictedBudget', 'CalculationDate',
            'ModelVersion', 'ModelCoefficients'
        ]].copy()
        
        # Rename columns to match database
        insert_data = insert_data.rename(columns={
            'CurrentAge': 'Age',
            'LivingSettingCategory': 'LivingSetting'
        })
        
        # Add metadata
        insert_data['RunID'] = self.run_id
        insert_data['CreatedDate'] = datetime.now()
        insert_data['CreatedBy'] = 'Model5b_Recalibration_Script'
        insert_data['PSum'] = 0  # Placeholder for PSum if needed
        
        try:
            with pyodbc.connect(self.conn_string) as conn:
                cursor = conn.cursor()
                
                # Insert records
                for _, row in insert_data.iterrows():
                    insert_query = """
                    INSERT INTO [dbo].[tbl_Model5b_Predictions] (
                        CaseNo, AssessID, ReviewDate, LivingSetting, Age, AgeGroup,
                        BSum, FSum, PSum, FHFSum, SLFSum, SLBSum,
                        SqrtPrediction, PredictedBudget, Prediction,
                        CalculationDate, CreatedDate, CreatedBy, RunID,
                        ModelVersion, ModelCoefficients
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    cursor.execute(insert_query, (
                        row['CaseNo'], row['AssessID'], row['ReviewDate'],
                        row['LivingSetting'], row['Age'], row['AgeGroup'],
                        row['BSum'], row['FSum'], row['PSum'],
                        row['FHFSum'], row['SLFSum'], row['SLBSum'],
                        float(row['SqrtPrediction']), float(row['PredictedBudget']),
                        float(row['PredictedBudget']),  # Prediction = PredictedBudget
                        row['CalculationDate'], row['CreatedDate'],
                        row['CreatedBy'], row['RunID'],
                        row['ModelVersion'], row['ModelCoefficients']
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(insert_data)} predictions to database")
                
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            raise
    
    def update_claims_amounts(self, start_date: str = '2024-07-01', 
                            end_date: str = '2025-06-30'):
        """
        Update claims amounts for the prediction records
        
        Args:
            start_date: Start date for claims period
            end_date: End date for claims period
        """
        logger.info(f"Updating claims amounts for period {start_date} to {end_date}")
        
        query = f"""
        EXEC [dbo].[sp_Update_Model5b_Claims]
            @StartDate = '{start_date}',
            @EndDate = '{end_date}',
            @RunID = {self.run_id}
        """
        
        try:
            with pyodbc.connect(self.conn_string) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                result = cursor.fetchone()
                
                if result:
                    logger.info(f"Updated claims for {result[0]} records")
                    
        except Exception as e:
            logger.error(f"Error updating claims: {str(e)}")
            raise
    
    def generate_summary_report(self, predictions: pd.DataFrame):
        """Generate summary statistics report"""
        
        summary = {
            'RunID': self.run_id,
            'ModelVersion': self.model_version,
            'TotalPredictions': len(predictions),
            'PredictionDate': datetime.now().isoformat(),
            'Statistics': {
                'MeanPrediction': float(predictions['PredictedBudget'].mean()),
                'MedianPrediction': float(predictions['PredictedBudget'].median()),
                'StdPrediction': float(predictions['PredictedBudget'].std()),
                'MinPrediction': float(predictions['PredictedBudget'].min()),
                'MaxPrediction': float(predictions['PredictedBudget'].max())
            },
            'LivingSettingDistribution': predictions['LivingSettingCategory'].value_counts().to_dict(),
            'AgeGroupDistribution': predictions['AgeGroup'].value_counts().to_dict()
        }
        
        # Save summary to file
        with open(f'prediction_summary_run{self.run_id}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to prediction_summary_run{self.run_id}.json")
        
        return summary


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print(" Model 5b Prediction Generation")
    print("="*70)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    try:
        # Initialize predictor
        logger.info("Initializing Model5b Predictor...")
        predictor = Model5bPredictor(coefficients_file='Model5bRecal.json')
        logger.info("[OK] Predictor initialized successfully")
        
    except FileNotFoundError:
        # Provide helpful guidance if coefficients file is missing
        print("\nTo generate predictions, you must first calibrate the model.")
        print("Run the following commands in order:")
        print("  1. python calibrate_model5b.py")
        print("  2. python predict_model5b.py (this script)")
        return
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        return
    
    try:
        # Phase 1: Load QSI data
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: LOADING QSI DATA")
        logger.info("="*60)
        logger.info("Retrieving latest QSI assessments from database...")
        qsi_data = predictor.get_latest_qsi_data()
        logger.info(f"[OK] Loaded {len(qsi_data)} QSI assessments")
        
        if len(qsi_data) == 0:
            logger.error("No QSI assessment data found in database")
            return
        
        # Phase 2: Prepare data
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: PREPARING DATA")
        logger.info("="*60)
        logger.info("Processing QSI data and creating features...")
        prepared_data = predictor.prepare_prediction_data(qsi_data)
        
        # Report data quality
        valid_count = len(prepared_data[~prepared_data['HasMissingData']])
        missing_count = len(prepared_data[prepared_data['HasMissingData']])
        
        logger.info(f"[OK] Data preparation complete")
        logger.info(f"  Valid records: {valid_count:,}")
        logger.info(f"  Records with missing data: {missing_count:,}")
        
        if valid_count == 0:
            logger.error("No valid records found for prediction")
            return
        
        # Phase 3: Calculate predictions
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: CALCULATING PREDICTIONS")
        logger.info("="*60)
        logger.info("Applying Model 5b coefficients...")
        predictions = predictor.calculate_predictions(prepared_data)
        logger.info(f"[OK] Calculated {len(predictions)} predictions")
        
        # Phase 4: Save to database
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: SAVING TO DATABASE")
        logger.info("="*60)
        logger.info("Writing predictions to tbl_Model5b_Predictions...")
        predictor.save_to_database(predictions)
        logger.info(f"[OK] Saved predictions with RunID: {predictor.run_id}")
        
        # Phase 5: Update claims
        logger.info("\n" + "="*60)
        logger.info("PHASE 5: UPDATING CLAIMS AMOUNTS")
        logger.info("="*60)
        logger.info("Updating claims for validation period...")
        predictor.update_claims_amounts()
        logger.info("[OK] Claims amounts updated")
        
        # Phase 6: Generate report
        logger.info("\n" + "="*60)
        logger.info("PHASE 6: GENERATING SUMMARY REPORT")
        logger.info("="*60)
        summary = predictor.generate_summary_report(predictions)
        
        # Display summary statistics
        print("\n" + "="*70)
        print(" PREDICTION SUMMARY")
        print("="*70)
        print(f" RunID: {summary['RunID']}")
        print(f" Total Predictions: {summary['TotalPredictions']:,}")
        print(f" Mean Prediction: ${summary['Statistics']['MeanPrediction']:,.2f}")
        print(f" Median Prediction: ${summary['Statistics']['MedianPrediction']:,.2f}")
        print(f" Min Prediction: ${summary['Statistics']['MinPrediction']:,.2f}")
        print(f" Max Prediction: ${summary['Statistics']['MaxPrediction']:,.2f}")
        
        print("\n Living Setting Distribution:")
        for setting, count in summary['LivingSettingDistribution'].items():
            print(f"  {setting}: {count:,}")
        
        print("\n Age Group Distribution:")
        for age_group, count in summary['AgeGroupDistribution'].items():
            print(f"  {age_group}: {count:,}")
        
        print("\n" + "="*70)
        print(" PREDICTIONS COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f" End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Summary Report: prediction_summary_run{predictor.run_id}.json")
        print("="*70 + "\n")
        
        logger.info("[OK] All predictions completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("\n\nProcess interrupted by user")
        return
    except pyodbc.Error as e:
        logger.error(f"\n[ERROR] Database error: {str(e)}")
        logger.error("Please check:")
        logger.error("  1. SQL Server is running")
        logger.error("  2. Database 'APD' exists")
        logger.error("  3. Tables tbl_QSIAssessments, tbl_Consumers, tbl_Model5b_Predictions exist")
        logger.error("  4. You have necessary permissions")
        return
    except Exception as e:
        logger.error(f"\n[ERROR] Critical error during prediction: {str(e)}")
        logger.error("Please check the log file for details")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()