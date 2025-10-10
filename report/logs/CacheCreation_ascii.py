"""
iBudget Data Caching Utility
=============================
Extracts data from database year by year and caches locally for faster access.
Each year is saved as a separate file to enable incremental updates.
"""

import pyodbc
import pickle
import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class iBudgetDataExtractor:
    """Extracts and caches iBudget data from SQL Server"""
    
    def __init__(self, 
                 server: str = '.', 
                 database: str = 'APD',
                 cache_dir: str = 'data/cached'):
        """
        Initialize extractor
        
        Args:
            server: SQL Server instance
            database: Database name
            cache_dir: Directory for cached files
        """
        self.server = server
        self.database = database
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"Trusted_Connection=yes"
        )
        
        # Track extraction metadata
        self.metadata_file = self.cache_dir / 'extraction_metadata.json'
        self.metadata = self.load_metadata()
    
    def load_metadata(self) -> Dict:
        """Load existing metadata or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'extractions': {},
            'last_update': None,
            'total_records': 0
        }
    
    def save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def get_available_years(self) -> List[Tuple[int, int]]:
        """
        Query database to find available fiscal years
        Note: Fiscal years run Sept 1 to Aug 31
        FY 2020 = Sept 1, 2019 to Aug 31, 2020
        
        Returns:
            List of (fiscal_year, fiscal_year) tuples for stored procedure
        """
        query = """
        SELECT 
            MIN(YEAR(DATEADD(MONTH, 3, ServiceDate))) as MinFY,
            MAX(YEAR(DATEADD(MONTH, 3, ServiceDate))) as MaxFY
        FROM tbl_Claims_MMIS
        WHERE ServiceDate IS NOT NULL
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                row = cursor.fetchone()
                
                if row and row[0] and row[1]:
                    min_fy = int(row[0])
                    max_fy = int(row[1])
                    
                    # Create fiscal year pairs
                    # Note: We pass (year, year) to the stored procedure
                    # The stored procedure interprets this as FY year (Sept-Aug)
                    years = []
                    for year in range(min_fy, max_fy + 1):
                        years.append((year, year))  # Stored proc expects same year twice
                    
                    logger.info(f"Found fiscal years from FY{min_fy} to FY{max_fy}")
                    logger.info(f"(Sept 1, {min_fy-1} through Aug 31, {max_fy})")
                    return years
                else:
                    logger.warning("No data found in database")
                    return []
                    
        except Exception as e:
            logger.error(f"Error querying available years: {e}")
            return []
    
    def extract_year_data(self, start_year: int, end_year: int) -> Dict[str, Any]:
        """
        Extract data for a specific fiscal year
        Note: The stored procedure expects (year, year) for FY year
        Example: (2020, 2020) returns FY2020 data (Sept 1, 2019 - Aug 31, 2020)
        
        Args:
            start_year: Fiscal year (passed to stored proc)
            end_year: Same as start_year (stored proc convention)
            
        Returns:
            Dictionary with data and metadata
        """
        # Display the actual date range for clarity
        actual_start = f"Sept 1, {start_year-1}"
        actual_end = f"Aug 31, {start_year}"
        logger.info(f"Extracting FY{start_year} ({actual_start} to {actual_end})...")
        start_time = time.time()
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Execute stored procedure
                query = "EXEC sp_GetiBudgetModelData @FiscalYearStart = ?, @FiscalYearEnd = ?"
                cursor.execute(query, start_year, end_year)
                
                # Get column names
                columns = [column[0] for column in cursor.description]
                
                # Fetch all rows
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries for easier handling
                data = []
                for row in rows:
                    # Convert row to dictionary
                    row_dict = {}
                    for i, value in enumerate(row):
                        # Handle different data types
                        if value is None:
                            row_dict[columns[i]] = None
                        elif isinstance(value, (int, float, str, bool)):
                            row_dict[columns[i]] = value
                        elif hasattr(value, 'isoformat'):  # datetime
                            row_dict[columns[i]] = value.isoformat()
                        else:
                            row_dict[columns[i]] = str(value)
                    data.append(row_dict)
                
                elapsed_time = time.time() - start_time
                
                result = {
                    'fiscal_year': f"FY{start_year}",  # e.g., "FY2020"
                    'fiscal_year_range': f"{start_year-1}-{start_year}",  # e.g., "2019-2020"
                    'date_range': f"Sept 1, {start_year-1} to Aug 31, {start_year}",
                    'columns': columns,
                    'data': data,
                    'record_count': len(data),
                    'extraction_time': elapsed_time,
                    'extraction_date': datetime.now().isoformat()
                }
                
                logger.info(f"Extracted {len(data)} records in {elapsed_time:.1f} seconds")
                return result
                
        except Exception as e:
            logger.error(f"Error extracting FY {start_year}-{end_year}: {e}")
            return None
    
    def save_year_data(self, year_data: Dict[str, Any], format: str = 'pickle'):
        """
        Save year data to file
        
        Args:
            year_data: Data dictionary from extract_year_data
            format: File format ('pickle', 'csv', 'json')
        """
        if not year_data:
            return
        
        fiscal_year = year_data['fiscal_year']  # e.g., "FY2020"
        fy_number = fiscal_year.replace('FY', '')  # e.g., "2020"
        filename_base = f"fy{fy_number}"  # e.g., "fy2020"
        
        if format == 'pickle':
            # Most efficient for Python
            filepath = self.cache_dir / f"{filename_base}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(year_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved {filepath}")
            
        elif format == 'csv':
            # Human-readable, good for other tools
            filepath = self.cache_dir / f"{filename_base}.csv"
            if year_data['data']:
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=year_data['columns'])
                    writer.writeheader()
                    writer.writerows(year_data['data'])
                logger.info(f"Saved {filepath}")
                
        elif format == 'json':
            # Good for web APIs, but larger file size
            filepath = self.cache_dir / f"{filename_base}.json"
            with open(filepath, 'w') as f:
                json.dump(year_data, f, indent=2, default=str)
            logger.info(f"Saved {filepath}")
        
        # Update metadata
        self.metadata['extractions'][fiscal_year] = {
            'record_count': year_data['record_count'],
            'extraction_date': year_data['extraction_date'],
            'extraction_time': year_data['extraction_time'],
            'format': format,
            'filepath': str(filepath)
        }
        self.metadata['last_update'] = datetime.now().isoformat()
        self.metadata['total_records'] = sum(
            ext['record_count'] for ext in self.metadata['extractions'].values()
        )
        self.save_metadata()
    
    def load_year_data(self, fiscal_year: int, format: str = 'pickle') -> Dict[str, Any]:
        """
        Load cached year data
        
        Args:
            fiscal_year: Fiscal year to load (e.g., 2020 for FY2020)
            format: File format to load
            
        Returns:
            Data dictionary or None if not found
        """
        filename_base = f"fy{fiscal_year}"  # e.g., "fy2020"
        
        if format == 'pickle':
            filepath = self.cache_dir / f"{filename_base}.pkl"
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
                    
        elif format == 'csv':
            filepath = self.cache_dir / f"{filename_base}.csv"
            if filepath.exists():
                data = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    columns = reader.fieldnames
                    for row in reader:
                        data.append(row)
                return {
                    'fiscal_year': f"FY{fiscal_year}",
                    'fiscal_year_range': f"{fiscal_year-1}-{fiscal_year}",
                    'columns': columns,
                    'data': data,
                    'record_count': len(data)
                }
                
        elif format == 'json':
            filepath = self.cache_dir / f"{filename_base}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    return json.load(f)
        
        return None
    
    def extract_all_years(self, 
                         force_refresh: bool = False,
                         format: str = 'pickle',
                         year_range: Optional[Tuple[int, int]] = None):
        """
        Extract and cache all available years
        
        Args:
            force_refresh: Re-extract even if cached
            format: File format for saving
            year_range: Optional (min_year, max_year) to limit extraction
        """
        # Get available years
        available_years = self.get_available_years()
        
        if not available_years:
            logger.error("No years available for extraction")
            return
        
        # Apply year range filter if specified
        if year_range:
            min_year, max_year = year_range
            available_years = [
                (y1, y2) for y1, y2 in available_years 
                if min_year <= y1 <= max_year
            ]
        
        logger.info(f"Processing {len(available_years)} fiscal years")
        logger.info("Note: Each fiscal year runs Sept 1 to Aug 31")
        logger.info("="*60)
        
        total_records = 0
        total_time = 0
        successful = 0
        failed = 0
        
        for start_year, end_year in available_years:
            fiscal_year = f"FY{start_year}"  # e.g., "FY2020"
            
            # Check if already cached
            if not force_refresh and fiscal_year in self.metadata['extractions']:
                logger.info(f"{fiscal_year}: Already cached ({self.metadata['extractions'][fiscal_year]['record_count']} records)")
                total_records += self.metadata['extractions'][fiscal_year]['record_count']
                successful += 1
                continue
            
            # Extract data
            year_data = self.extract_year_data(start_year, end_year)
            
            if year_data:
                # Save to file
                self.save_year_data(year_data, format=format)
                total_records += year_data['record_count']
                total_time += year_data['extraction_time']
                successful += 1
            else:
                logger.error(f"Failed to extract {fiscal_year}")
                failed += 1
            
            # Small delay to avoid overwhelming the database
            time.sleep(1)
        
        # Print summary
        logger.info("="*60)
        logger.info("EXTRACTION COMPLETE")
        logger.info(f"Successful: {successful} years")
        logger.info(f"Failed: {failed} years")
        logger.info(f"Total records: {total_records:,}")
        logger.info(f"Total extraction time: {total_time/60:.1f} minutes")
        if successful > 0:
            logger.info(f"Average time per year: {total_time/successful:.1f} seconds")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info("="*60)
    
    def get_cached_years(self) -> List[str]:
        """Get list of cached fiscal years"""
        return list(self.metadata['extractions'].keys())
    
    def combine_all_cached_data(self, format: str = 'pickle') -> Dict[str, Any]:
        """
        Load and combine all cached years into single dataset
        
        Args:
            format: Format of cached files to load
            
        Returns:
            Combined dataset dictionary
        """
        all_data = []
        all_columns = set()
        
        for fiscal_year_key in sorted(self.metadata['extractions'].keys()):
            # Extract fiscal year number from key (e.g., "FY2020" -> 2020)
            fy_number = int(fiscal_year_key.replace('FY', ''))
            
            year_data = self.load_year_data(fy_number, format=format)
            
            if year_data:
                all_data.extend(year_data['data'])
                all_columns.update(year_data['columns'])
                logger.info(f"Loaded {year_data['record_count']} records from {fiscal_year_key}")
        
        return {
            'columns': sorted(list(all_columns)),
            'data': all_data,
            'record_count': len(all_data),
            'fiscal_years': sorted(self.metadata['extractions'].keys())
        }
    
    def print_cache_summary(self):
        """Print summary of cached data"""
        print("\n" + "="*60)
        print("CACHED DATA SUMMARY")
        print("="*60)
        
        if not self.metadata['extractions']:
            print("No cached data found")
            return
        
        print(f"Cache directory: {self.cache_dir}")
        print(f"Last update: {self.metadata['last_update']}")
        print(f"Total cached records: {self.metadata['total_records']:,}")
        print(f"\nCached fiscal years ({len(self.metadata['extractions'])}):")
        print("-"*40)
        
        for fy in sorted(self.metadata['extractions'].keys()):
            ext = self.metadata['extractions'][fy]
            # Extract year number and show date range
            fy_num = fy.replace('FY', '')
            if fy_num.isdigit():
                year = int(fy_num)
                date_range = f"(Sept {year-1} - Aug {year})"
            else:
                date_range = ""
            
            print(f"  {fy} {date_range}: {ext['record_count']:,} records "
                  f"({ext['extraction_time']:.1f}s) "
                  f"[{ext['extraction_date'][:10]}]")
        
        print("="*60)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cache iBudget data from database')
    parser.add_argument('--server', default='.', help='SQL Server instance')
    parser.add_argument('--database', default='APD', help='Database name')
    parser.add_argument('--cache-dir', default='data/cached', help='Cache directory')
    parser.add_argument('--format', choices=['pickle', 'csv', 'json'], default='pickle',
                       help='File format for caching')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Re-extract even if cached')
    parser.add_argument('--min-year', type=int, help='Minimum fiscal year')
    parser.add_argument('--max-year', type=int, help='Maximum fiscal year')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only show cache summary')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = iBudgetDataExtractor(
        server=args.server,
        database=args.database,
        cache_dir=args.cache_dir
    )
    
    if args.summary_only:
        # Just show summary
        extractor.print_cache_summary()
    else:
        # Extract data
        year_range = None
        if args.min_year or args.max_year:
            min_y = args.min_year or 2000
            max_y = args.max_year or 2030
            year_range = (min_y, max_y)
        
        print(f"\nStarting extraction process...")
        print(f"Format: {args.format}")
        print(f"Force refresh: {args.force_refresh}")
        if year_range:
            print(f"Year range: {year_range[0]}-{year_range[1]}")
        print()
        
        extractor.extract_all_years(
            force_refresh=args.force_refresh,
            format=args.format,
            year_range=year_range
        )
        
        # Show summary
        extractor.print_cache_summary()


if __name__ == "__main__":
    main()


"""
USAGE EXAMPLES:
==============

1. Extract all available years (first time):
   python cache_ibudget_data.py

2. Extract specific year range:
   python cache_ibudget_data.py --min-year 2019 --max-year 2025
   
   Note: This extracts FY2019 through FY2025 where:
   - FY2019 = Sept 1, 2018 to Aug 31, 2019
   - FY2020 = Sept 1, 2019 to Aug 31, 2020
   - FY2021 = Sept 1, 2020 to Aug 31, 2021
   - etc.

3. Force refresh of all data:
   python cache_ibudget_data.py --force-refresh

4. Save as CSV for external tools:
   python cache_ibudget_data.py --format csv

5. Just show what's cached:
   python cache_ibudget_data.py --summary-only

6. Custom cache directory:
   python cache_ibudget_data.py --cache-dir /path/to/cache

After caching, models can load data instantly:
=============================================

from cache_ibudget_data import iBudgetDataExtractor

# Load cached data
extractor = iBudgetDataExtractor()

# Load specific fiscal year
data_fy2020 = extractor.load_year_data(2020)  # FY2020: Sept 2019 - Aug 2020

# Load all cached years combined
all_data = extractor.combine_all_cached_data()

# This takes seconds instead of minutes!
"""
