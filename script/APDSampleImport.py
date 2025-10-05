#!/usr/bin/env python3
"""
Import ADP Sample Excel data into SQL Server ADPSample table
Using the database approach from Metadata.py
"""

import pandas as pd
import pyodbc
from sqlalchemy import create_engine
import urllib
from datetime import datetime
import sys
import os
import numpy as np

class ADPSampleImporter:
    def __init__(self, server, database, username=None, password=None):
        """
        Initialize connection to SQL Server database.
        If username/password are not provided, uses Windows Authentication.
        
        Args:
            server: SQL Server instance (e.g., '.' for local, 'server\\instance' for named instance)
            database: Database name
            username: SQL username (optional)
            password: SQL password (optional)
        """
        # Build connection string
        if username and password:
            self.conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        else:
            # Windows Authentication
            self.conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes'
        
        # Create SQLAlchemy engine for pandas compatibility
        params = urllib.parse.quote_plus(self.conn_str)
        self.engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        
        self.database = database
        self.conn = None
        
        # Statistics for import
        self.total_rows = 0
        self.rows_inserted = 0
        self.errors = 0
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = pyodbc.connect(self.conn_str)
            print(f"✓ Successfully connected to {self.database}")
            return True
        except Exception as e:
            print(f"✗ Error connecting to database: {e}")
            return False
    
    def check_table_exists(self):
        """Check if ADPSample table exists"""
        try:
            cursor = self.conn.cursor()
            query = """
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'dbo' 
            AND TABLE_NAME = 'ADPSample'
            """
            cursor.execute(query)
            exists = cursor.fetchone()[0] > 0
            
            if not exists:
                print("✗ Table 'ADPSample' does not exist.")
                print("  Please run the DDL script first to create the table.")
                return False
            
            print("✓ Table 'ADPSample' exists")
            return True
            
        except Exception as e:
            print(f"✗ Error checking table: {e}")
            return False
    
    def parse_date(self, date_str):
        """Parse date strings in various formats."""
        if pd.isna(date_str) or date_str == '' or date_str == 'NULL':
            return None
        
        # Try different date formats
        date_formats = [
            '%d-%b-%y',   # 1-Jul-25
            '%d-%b-%Y',   # 1-Jul-2025
            '%m/%d/%Y',   # 07/01/2025
            '%Y-%m-%d',   # 2025-07-01
            '%d/%m/%Y',   # 01/07/2025
            '%Y/%m/%d',   # 2025/07/01
            '%d-%m-%Y',   # 01-07-2025
            '%d-%m-%y'    # 01-07-25
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        
        # If all formats fail, return None
        print(f"  Warning: Could not parse date '{date_str}'")
        return None
    
    def clean_numeric(self, value):
        """Clean numeric values."""
        if pd.isna(value) or value == '' or value == 'NULL':
            return None
        try:
            # Remove any non-numeric characters except . and -
            if isinstance(value, str):
                value = value.replace(',', '').strip()
            return float(value)
        except:
            return None
    
    def clean_integer(self, value):
        """Clean integer values."""
        if pd.isna(value) or value == '' or value == 'NULL':
            return None
        try:
            if isinstance(value, str):
                value = value.replace(',', '').strip()
            return int(float(value))
        except:
            return None
    
    def clean_string(self, value, max_length=None):
        """Clean string values."""
        if pd.isna(value) or value == 'NULL':
            return None
        
        value = str(value).strip()
        if value == '':
            return None
        
        # Truncate if necessary
        if max_length and len(value) > max_length:
            value = value[:max_length]
        
        return value
    
    def load_excel_data(self, file_path):
        """Load data from Excel file using pandas."""
        try:
            print(f"\nLoading Excel file: {file_path}")
            
            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=0)  # Read first sheet
            
            self.total_rows = len(df)
            print(f"✓ Loaded {self.total_rows:,} rows from Excel file")
            print(f"  Columns found: {len(df.columns)}")
            
            # Display column names
            print("\n  First 10 columns:")
            for i, col in enumerate(df.columns[:10]):
                print(f"    {i+1}. {col}")
            
            return df
            
        except FileNotFoundError:
            print(f"✗ File not found: {file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to load Excel file: {e}")
            sys.exit(1)
    
    def process_dataframe(self, df):
        """Process and clean the dataframe."""
        print("\nProcessing dataframe...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Date columns
        date_columns = ['BeginDate', 'ENDDATE', 'DOB', 'CompletedDate']
        for col in date_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].apply(self.parse_date)
        
        # Integer columns
        integer_columns = [
            'CaseNo', 'FiscalYear', 'VendorId', 'PlannedServiceId', 'PlanId',
            'CLIENTID', 'CurrentAge', 'FuncationalStatus', 'BehavioralStatus', 
            'PhysicalStatus', 'EstLevel', 'FunctSum', 'BehavSum', 'PhysSum',
            'RATERID', 'Days_Since_QSI'
        ]
        
        # Add Q columns (Q14-Q51A, Q13a-Q13c)
        q_columns = [f'Q{i}' for i in range(14, 52)]
        q_columns.extend(['Q13a', 'Q13b', 'Q13c', 'Q51A'])
        integer_columns.extend(q_columns)
        
        for col in integer_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].apply(self.clean_integer)
        
        # Numeric (decimal) columns
        numeric_columns = [
            'UnitsPer', 'TotalUnits', 'Rate', 'TotalAmount',
            'RH1', 'RH2', 'RH3', 'RH4', 'AGE50P', 'AGE60P', 'AGE70P',
            'ResHabFlag', 'FamilyHome', 'IndLiving_SuppLvg',
            'ResHabSrvcPlan_notResHabLivSetting_Flag', 'Intercept',
            'Live2ILSL', 'Live2RH1', 'Live2RH2', 'Live2RH3', 'Live2RH4',
            'Age21_30', 'Age30plus', 'BSum', 'FHFSum', 'SLFSum', 'SLBSum',
            'Q_16', 'Q_18', 'Q_20', 'Q_21', 'Q_23', 'Q_28', 'Q_33', 'Q_34',
            'Q_36', 'Q_43', 'CoefficientsSum', 'AlgorithmAmtModel5b',
            'InvalidAlgorithmLivingSetting'
        ]
        
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].apply(self.clean_numeric)
        
        # String columns with max lengths
        string_columns = {
            'IndexSubObjectCode': 100, 'ConsumerCounty': 100,
            'GeographicDifferential': 100, 'ProviderRateType': 100,
            'PROC': 25, 'Service': 200, 'UnitType': 100,
            'UnitsOfMeasure': 50, 'ProviderName': 200, 'ProviderMedcId': 20,
            'PIN': 20, 'SSN': 20, 'iConnect_Category': 100,
            'SBPG_C1': 50, 'Region2': 50, 'MedicaidId': 20,
            'LNAME': 50, 'FNAME': 50, 'MI': 10, 'SFX': 10,
            'RACE': 50, 'SEX': 10, 'MEDC_ID': 20,
            'CNTY_RECMD': 50, 'CNTY_RESID': 50,
            'REGION_NAME': 100, 'CNTY_RESID_NAME': 100,
            'PrimDisabilityDescription': 200,
            'SecondaryDisabilityDescription': 200,
            'OtherDisabilityDescription': 200,
            'Status_Description': 200, 'Category': 100,
            'GroupedProgCompDescript': 200, 'ProgCompDesc': 200,
            'Worker_Dist': 50, 'Worker_SD': 50, 'Worker_Unit': 50,
            'Worker_Code': 50, 'Worker_First_Name': 100,
            'Worker_Last_Name': 100, 'Worker_SSN': 20,
            'WSC_File': 50, 'WL_Priority': 100, 'DUPLICATE': 10,
            'CALCULATE': 50, 'ClientLivingSetting': 50
        }
        
        for col, max_len in string_columns.items():
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].apply(
                    lambda x: self.clean_string(x, max_len)
                )
        
        print(f"✓ Processed {len(processed_df)} rows")
        
        # Display data quality summary
        null_counts = processed_df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        
        if len(cols_with_nulls) > 0:
            print(f"\n  Columns with NULL values: {len(cols_with_nulls)}")
            print("  Top 5 columns with most NULLs:")
            for col, count in cols_with_nulls.nlargest(5).items():
                pct = (count / len(processed_df)) * 100
                print(f"    - {col}: {count:,} ({pct:.1f}%)")
        
        return processed_df
    
    def truncate_table(self):
        """Truncate the ADPSample table before inserting new data."""
        try:
            cursor = self.conn.cursor()
            
            # Get current row count
            cursor.execute("SELECT COUNT(*) FROM dbo.ADPSample")
            current_count = cursor.fetchone()[0]
            
            if current_count > 0:
                print(f"\n  Current table has {current_count:,} rows")
                response = input("  Do you want to truncate existing data? (y/n): ")
                
                if response.lower() != 'y':
                    print("  Keeping existing data. New data will be appended.")
                    return True
            
            cursor.execute("TRUNCATE TABLE dbo.ADPSample")
            self.conn.commit()
            print("✓ Table ADPSample truncated")
            return True
            
        except Exception as e:
            print(f"✗ Failed to truncate table: {e}")
            return False
    
    def insert_data_bulk(self, df):
        """Insert data using pandas to_sql for better performance."""
        try:
            print(f"\nInserting {len(df):,} rows using bulk insert...")
            
            # Remove ID column if it exists (it's an identity column)
            if 'ID' in df.columns:
                df = df.drop('ID', axis=1)
            
            # Use pandas to_sql for bulk insert
            df.to_sql('ADPSample', self.engine, schema='dbo', 
                     if_exists='append', index=False, method='multi',
                     chunksize=1000)
            
            self.rows_inserted = len(df)
            print(f"✓ Successfully inserted {self.rows_inserted:,} rows")
            return True
            
        except Exception as e:
            print(f"✗ Error during bulk insert: {e}")
            print("  Attempting row-by-row insert for debugging...")
            return self.insert_data_batch(df)
    
    def insert_data_batch(self, df):
        """Insert data in batches with detailed error handling."""
        cursor = self.conn.cursor()
        
        # Build the INSERT statement dynamically based on available columns
        available_columns = [col for col in df.columns if col != 'ID']
        
        # Build column list and placeholders
        columns_str = ', '.join([f'[{col}]' for col in available_columns])
        placeholders = ', '.join(['?' for _ in available_columns])
        
        insert_sql = f"""
            INSERT INTO dbo.ADPSample ({columns_str})
            VALUES ({placeholders})
        """
        
        # Insert data in batches
        batch_size = 100
        total_rows = len(df)
        
        print(f"\nInserting {total_rows:,} rows in batches...")
        
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:min(i+batch_size, total_rows)]
            
            for idx, row in batch.iterrows():
                try:
                    # Prepare values for insertion
                    values = []
                    for col in available_columns:
                        val = row[col]
                        if pd.isna(val):
                            values.append(None)
                        elif isinstance(val, np.integer):
                            values.append(int(val))
                        elif isinstance(val, np.floating):
                            values.append(float(val))
                        else:
                            values.append(val)
                    
                    cursor.execute(insert_sql, values)
                    self.rows_inserted += 1
                    
                    # Show progress
                    if self.rows_inserted % 100 == 0:
                        print(f"  Inserted {self.rows_inserted}/{total_rows} rows...", end='\r')
                        
                except Exception as e:
                    self.errors += 1
                    if self.errors <= 5:  # Show first 5 errors
                        print(f"\n  ✗ Error at row {i + idx + 1}: {e}")
                    
                    if self.errors > 50:
                        print("\n  Too many errors. Stopping insertion.")
                        self.conn.rollback()
                        return False
            
            # Commit after each batch
            self.conn.commit()
        
        print(f"\n✓ Inserted {self.rows_inserted:,} rows successfully")
        if self.errors > 0:
            print(f"  ⚠ {self.errors} rows failed to insert")
        
        return True
    
    def verify_import(self):
        """Verify the import by checking row count and sample data."""
        try:
            cursor = self.conn.cursor()
            
            # Get row count
            cursor.execute("SELECT COUNT(*) FROM dbo.ADPSample")
            count = cursor.fetchone()[0]
            print(f"\n✓ Verification: {count:,} rows in ADPSample table")
            
            # Get column count
            cursor.execute("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'ADPSample'
            """)
            col_count = cursor.fetchone()[0]
            print(f"  Total columns: {col_count}")
            
            # Show sample data
            print("\n  Sample data (first 5 rows):")
            cursor.execute("""
                SELECT TOP 5 
                    CaseNo, FiscalYear, ProviderName, 
                    Service, TotalAmount, CurrentAge
                FROM dbo.ADPSample
                ORDER BY ID
            """)
            
            rows = cursor.fetchall()
            for i, row in enumerate(rows, 1):
                provider = row[2][:30] + '...' if row[2] and len(row[2]) > 30 else row[2]
                service = row[3][:25] + '...' if row[3] and len(row[3]) > 25 else row[3]
                amount = f"${row[4]:,.2f}" if row[4] else "N/A"
                age = row[5] if row[5] else "N/A"
                
                print(f"    {i}. CaseNo: {row[0]}, FY: {row[1]}, Age: {age}")
                print(f"       Provider: {provider}")
                print(f"       Service: {service}, Amount: {amount}")
            
            # Show some statistics
            print("\n  Data statistics:")
            
            # Fiscal year distribution
            cursor.execute("""
                SELECT FiscalYear, COUNT(*) as cnt
                FROM dbo.ADPSample
                GROUP BY FiscalYear
                ORDER BY FiscalYear
            """)
            fy_dist = cursor.fetchall()
            print("    Fiscal Year distribution:")
            for fy, cnt in fy_dist:
                print(f"      FY {fy}: {cnt:,} rows")
            
            # Total amount summary
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT CaseNo) as unique_cases,
                    COUNT(DISTINCT ProviderName) as unique_providers,
                    SUM(TotalAmount) as total_amount,
                    AVG(TotalAmount) as avg_amount
                FROM dbo.ADPSample
                WHERE TotalAmount IS NOT NULL
            """)
            
            stats = cursor.fetchone()
            if stats:
                print(f"\n    Unique Cases: {stats[0]:,}")
                print(f"    Unique Providers: {stats[1]:,}")
                if stats[2]:
                    print(f"    Total Amount: ${stats[2]:,.2f}")
                    print(f"    Average Amount: ${stats[3]:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error during verification: {e}")
            return False
    
    def run_import(self, excel_file):
        """Main method to run the complete import process."""
        print("=" * 70)
        print(" ADP Sample Data Import Tool ")
        print(" Using Database Connection Method from Metadata.py ")
        print("=" * 70)
        
        # Check if file exists
        if not os.path.exists(excel_file):
            print(f"\n✗ Excel file not found: {excel_file}")
            print("  Please ensure the file exists at the specified location.")
            return False
        
        # Connect to database
        if not self.connect():
            return False
        
        try:
            # Check if table exists
            if not self.check_table_exists():
                return False
            
            # Load Excel data
            df = self.load_excel_data(excel_file)
            
            # Process and clean the data
            df_processed = self.process_dataframe(df)
            
            # Optional: Truncate table
            self.truncate_table()
            
            # Insert data
            success = self.insert_data_bulk(df_processed)
            
            if success:
                # Verify import
                self.verify_import()
                
                print("\n" + "=" * 70)
                print(" ✓ Import completed successfully!")
                print(f"   Total rows processed: {self.total_rows:,}")
                print(f"   Rows inserted: {self.rows_inserted:,}")
                if self.errors > 0:
                    print(f"   Rows with errors: {self.errors:,}")
                print("=" * 70)
            else:
                print("\n✗ Import failed!")
            
            return success
            
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            if self.conn:
                self.conn.rollback()
            return False
            
        finally:
            if self.conn:
                self.conn.close()
                print("\n✓ Database connection closed")

# Main execution
if __name__ == "__main__":
    # Configuration - UPDATE THESE VALUES
    SERVER = '.'  # e.g., 'localhost' or 'server\\instance'
    DATABASE = 'APD'
    
    # For Windows Authentication (leave username and password as None)
    USERNAME = None  
    PASSWORD = None
    
    # For SQL Server Authentication (uncomment and fill in)
    # USERNAME = 'your_username'
    # PASSWORD = 'your_password'
    
    # Excel file path
    EXCEL_FILE = '../data/ADP-sample.xlsx'
    
    # Note: Install required packages with:
    # pip install pandas pyodbc sqlalchemy openpyxl numpy
    
    # Create importer and run import
    importer = ADPSampleImporter(SERVER, DATABASE, USERNAME, PASSWORD)
    importer.run_import(EXCEL_FILE)