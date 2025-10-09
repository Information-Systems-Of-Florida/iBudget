"""
iBudget Data Quality Testing Suite
===================================
Comprehensive testing after stored procedure updates and cache refresh.
Tests for LivingSetting fix and AvgAge column addition.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataQualityTester:
    def __init__(self, cache_dir='models/data/cached'):
        self.cache_dir = Path(cache_dir)
        self.results = {}
        self.issues = []
        
    def load_fiscal_year(self, fiscal_year):
        """Load a single fiscal year"""
        filepath = self.cache_dir / f"fy{fiscal_year}.pkl"
        if not filepath.exists():
            return None
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def test_column_existence(self, data, fiscal_year):
        """Test 1: Verify required columns exist"""
        print(f"\n{'='*70}")
        print(f"TEST 1: Column Existence Check - FY{fiscal_year}")
        print('='*70)
        
        required_columns = [
            'ConsumerID', 'LivingSetting', 'TotalCost', 'Usable',
            'FSum', 'BSum', 'PSum', 'Age', 'AvgAge',  # ← New column
            'Q14', 'Q15', 'Q16', 'AgeGroup'
        ]
        
        missing = []
        for col in required_columns:
            if col not in data['columns']:
                missing.append(col)
                self.issues.append(f"FY{fiscal_year}: Missing column '{col}'")
        
        if missing:
            print(f"❌ FAILED - Missing columns: {missing}")
            return False
        else:
            print(f"✓ PASSED - All {len(required_columns)} required columns present")
            print(f"  Total columns in dataset: {len(data['columns'])}")
            return True
    
    def test_living_setting_distribution(self, data, fiscal_year):
        """Test 2: Verify LivingSetting has RH1-RH4 categories"""
        print(f"\n{'='*70}")
        print(f"TEST 2: LivingSetting Distribution - FY{fiscal_year}")
        print('='*70)
        
        usable = [r for r in data['data'] if r.get('Usable') == 1]
        living_settings = Counter([r.get('LivingSetting') for r in usable])
        
        expected_categories = {'FH', 'ILSL', 'RH1', 'RH2', 'RH3', 'RH4'}
        found_categories = set(living_settings.keys())
        
        print(f"Expected categories: {expected_categories}")
        print(f"Found categories: {found_categories}")
        print(f"\nDistribution ({len(usable):,} usable records):")
        
        for cat in ['FH', 'ILSL', 'RH1', 'RH2', 'RH3', 'RH4']:
            count = living_settings.get(cat, 0)
            pct = (count / len(usable) * 100) if len(usable) > 0 else 0
            status = "✓" if count > 0 else "❌"
            print(f"  {status} {cat:6s}: {count:6,} ({pct:5.1f}%)")
            
            if count == 0 and cat.startswith('RH'):
                self.issues.append(f"FY{fiscal_year}: No records in category '{cat}'")
        
        missing_categories = expected_categories - found_categories
        if missing_categories:
            print(f"\n❌ WARNING - Missing categories: {missing_categories}")
            return False
        else:
            print(f"\n✓ PASSED - All 6 categories present")
            return True
    
    def test_avgage_column(self, data, fiscal_year):
        """Test 3: Verify AvgAge column quality"""
        print(f"\n{'='*70}")
        print(f"TEST 3: AvgAge Column Quality - FY{fiscal_year}")
        print('='*70)
        
        usable = [r for r in data['data'] if r.get('Usable') == 1]
        
        # Extract AvgAge values
        avg_ages = []
        ages = []
        null_count = 0
        zero_count = 0
        
        for r in usable:
            avg_age = r.get('AvgAge')
            age = r.get('Age')
            
            if avg_age is None:
                null_count += 1
            elif float(avg_age) == 0:
                zero_count += 1
            else:
                avg_ages.append(float(avg_age))
            
            if age is not None:
                ages.append(int(age))
        
        print(f"Records analyzed: {len(usable):,}")
        print(f"  NULL values: {null_count:,}")
        print(f"  Zero values: {zero_count:,}")
        print(f"  Valid values: {len(avg_ages):,}")
        
        if avg_ages:
            print(f"\nAvgAge Statistics:")
            print(f"  Min: {min(avg_ages):.1f} years")
            print(f"  Max: {max(avg_ages):.1f} years")
            print(f"  Mean: {np.mean(avg_ages):.1f} years")
            print(f"  Median: {np.median(avg_ages):.1f} years")
            print(f"  Std Dev: {np.std(avg_ages):.1f} years")
        
        if ages:
            print(f"\nConsumer Age Statistics (for comparison):")
            print(f"  Min: {min(ages)} years")
            print(f"  Max: {max(ages)} years")
            print(f"  Mean: {np.mean(ages):.1f} years")
        
        # Check for anomalies
        anomalies = []
        if null_count > len(usable) * 0.1:
            anomalies.append(f"High NULL rate: {null_count/len(usable)*100:.1f}%")
        if zero_count > len(usable) * 0.1:
            anomalies.append(f"High zero rate: {zero_count/len(usable)*100:.1f}%")
        
        if avg_ages:
            if min(avg_ages) < 3:
                anomalies.append(f"Suspiciously low min age: {min(avg_ages):.1f}")
            if max(avg_ages) > 100:
                anomalies.append(f"Suspiciously high max age: {max(avg_ages):.1f}")
        
        if anomalies:
            print(f"\n❌ WARNINGS:")
            for a in anomalies:
                print(f"  - {a}")
                self.issues.append(f"FY{fiscal_year}: AvgAge - {a}")
            return False
        else:
            print(f"\n✓ PASSED - AvgAge column looks good")
            return True
    
    def test_data_integrity(self, data, fiscal_year):
        """Test 4: General data integrity checks"""
        print(f"\n{'='*70}")
        print(f"TEST 4: Data Integrity - FY{fiscal_year}")
        print('='*70)
        
        usable = [r for r in data['data'] if r.get('Usable') == 1]
        
        checks = {
            'TotalCost > 0': sum(1 for r in usable if float(r.get('TotalCost', 0)) > 0),
            'TotalCost <= 0': sum(1 for r in usable if float(r.get('TotalCost', 0)) <= 0),
            'FSum valid (0-44)': sum(1 for r in usable if 0 <= int(r.get('FSum', -1)) <= 44),
            'BSum valid (0-24)': sum(1 for r in usable if 0 <= int(r.get('BSum', -1)) <= 24),
            'PSum valid (0-76)': sum(1 for r in usable if 0 <= int(r.get('PSum', -1)) <= 76),
            'Age > 0': sum(1 for r in usable if int(r.get('Age', 0)) > 0),
            'DaysInSystem >= 30': sum(1 for r in usable if int(r.get('DaysInSystem', 0)) >= 30),
        }
        
        print(f"Usable records: {len(usable):,}\n")
        
        all_passed = True
        for check, count in checks.items():
            pct = (count / len(usable) * 100) if len(usable) > 0 else 0
            status = "✓" if count == len(usable) else "⚠"
            
            if count < len(usable):
                all_passed = False
                diff = len(usable) - count
                self.issues.append(f"FY{fiscal_year}: {diff} records failed '{check}'")
            
            print(f"  {status} {check:25s}: {count:6,} / {len(usable):,} ({pct:5.1f}%)")
        
        return all_passed
    
    def test_model5b_features(self, data, fiscal_year):
        """Test 5: Model 5b specific feature availability"""
        print(f"\n{'='*70}")
        print(f"TEST 5: Model 5b Feature Availability - FY{fiscal_year}")
        print('='*70)
        
        usable = [r for r in data['data'] if r.get('Usable') == 1 and float(r.get('TotalCost', 0)) > 0]
        
        # Check for each living setting category
        living_settings = Counter([r.get('LivingSetting') for r in usable])
        age_groups = Counter([r.get('AgeGroup') for r in usable])
        
        print("Living Setting Categories for Dummy Variables:")
        for cat in ['FH', 'ILSL', 'RH1', 'RH2', 'RH3', 'RH4']:
            count = living_settings.get(cat, 0)
            status = "✓" if count > 10 else "❌"  # Need at least 10 for modeling
            print(f"  {status} {cat}: {count:,} records")
            
            if count < 10:
                self.issues.append(f"FY{fiscal_year}: Insufficient records for {cat} ({count})")
        
        print("\nAge Group Categories:")
        for cat in ['Age3_20', 'Age21_30', 'Age31Plus']:
            count = age_groups.get(cat, 0)
            pct = (count / len(usable) * 100) if len(usable) > 0 else 0
            status = "✓" if count > 10 else "❌"
            print(f"  {status} {cat}: {count:,} ({pct:.1f}%)")
        
        print("\nQSI Score Ranges:")
        fsums = [int(r.get('FSum', 0)) for r in usable]
        bsums = [int(r.get('BSum', 0)) for r in usable]
        psums = [int(r.get('PSum', 0)) for r in usable]
        
        print(f"  FSum: {min(fsums)} to {max(fsums)} (mean: {np.mean(fsums):.1f})")
        print(f"  BSum: {min(bsums)} to {max(bsums)} (mean: {np.mean(bsums):.1f})")
        print(f"  PSum: {min(psums)} to {max(psums)} (mean: {np.mean(psums):.1f})")
        
        return True
    
    def test_cross_year_consistency(self, years):
        """Test 6: Cross-year consistency"""
        print(f"\n{'='*70}")
        print(f"TEST 6: Cross-Year Consistency")
        print('='*70)
        
        year_data = {}
        for year in years:
            data = self.load_fiscal_year(year)
            if data:
                year_data[year] = data
        
        if len(year_data) < 2:
            print("⚠ Need at least 2 years for comparison")
            return True
        
        # Compare column sets
        print("\nColumn Consistency:")
        base_year = min(year_data.keys())
        base_columns = set(year_data[base_year]['columns'])
        
        for year in sorted(year_data.keys()):
            if year == base_year:
                continue
            year_columns = set(year_data[year]['columns'])
            
            missing = base_columns - year_columns
            extra = year_columns - base_columns
            
            if missing or extra:
                print(f"  ❌ FY{year} vs FY{base_year}:")
                if missing:
                    print(f"     Missing: {missing}")
                if extra:
                    print(f"     Extra: {extra}")
            else:
                print(f"  ✓ FY{year}: Consistent with FY{base_year}")
        
        # Compare LivingSetting distributions
        print("\nLivingSetting Distribution Comparison:")
        print(f"{'Year':<8} {'FH':>8} {'ILSL':>8} {'RH1':>8} {'RH2':>8} {'RH3':>8} {'RH4':>8} {'Total':>10}")
        print("-" * 70)
        
        for year in sorted(year_data.keys()):
            data = year_data[year]
            usable = [r for r in data['data'] if r.get('Usable') == 1]
            living_settings = Counter([r.get('LivingSetting') for r in usable])
            
            print(f"FY{year}  "
                  f"{living_settings.get('FH', 0):8,} "
                  f"{living_settings.get('ILSL', 0):8,} "
                  f"{living_settings.get('RH1', 0):8,} "
                  f"{living_settings.get('RH2', 0):8,} "
                  f"{living_settings.get('RH3', 0):8,} "
                  f"{living_settings.get('RH4', 0):8,} "
                  f"{len(usable):10,}")
        
        return True
    
    def run_all_tests(self, fiscal_years=None):
        """Run all quality tests"""
        if fiscal_years is None:
            fiscal_years = [2020, 2021, 2022, 2023, 2024]
        
        print("="*70)
        print("iBudget DATA QUALITY TESTING SUITE")
        print("="*70)
        print(f"Cache directory: {self.cache_dir}")
        print(f"Testing fiscal years: {fiscal_years}")
        
        # Test each year individually
        for year in fiscal_years:
            data = self.load_fiscal_year(year)
            
            if data is None:
                print(f"\n❌ FY{year}: File not found")
                self.issues.append(f"FY{year}: Cache file not found")
                continue
            
            # Run individual tests
            self.test_column_existence(data, year)
            self.test_living_setting_distribution(data, year)
            self.test_avgage_column(data, year)
            self.test_data_integrity(data, year)
            self.test_model5b_features(data, year)
        
        # Cross-year tests
        self.test_cross_year_consistency(fiscal_years)
        
        # Final summary
        self.print_summary()
    
    def print_summary(self):
        """Print final summary"""
        print(f"\n{'='*70}")
        print("TESTING SUMMARY")
        print('='*70)
        
        if not self.issues:
            print("✓ ALL TESTS PASSED - Data quality looks good!")
            print("\nReady for model training.")
        else:
            print(f"❌ FOUND {len(self.issues)} ISSUES:\n")
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
            
            print("\n⚠ RECOMMENDATION:")
            if any('Missing column' in issue for issue in self.issues):
                print("  - Re-run stored procedure and cache extraction")
            if any('No records in category' in issue for issue in self.issues):
                print("  - Check stored procedure LivingSetting CASE logic")
            if any('AvgAge' in issue for issue in self.issues):
                print("  - Verify tbl_Claims_MMIS.Age column exists and has data")
        
        print('='*70)


def main():
    """Run the test suite"""
    tester = DataQualityTester(cache_dir='models/data/cached')
    
    # Test all available years
    tester.run_all_tests(fiscal_years=[2020, 2021, 2022, 2023, 2024])
    
    # Optional: Test only specific years
    # tester.run_all_tests(fiscal_years=[2024])


if __name__ == "__main__":
    main()
