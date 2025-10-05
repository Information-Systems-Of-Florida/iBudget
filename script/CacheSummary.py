"""
inspect_cache.py
================
Utility to inspect cached pickle files and understand their structure
"""

import pickle
from pathlib import Path
import sys
from typing import Dict, Any


def inspect_pickle_file(filepath: Path) -> None:
    """Inspect a pickle file and print its structure"""
    print(f"\nInspecting: {filepath}")
    print("="*60)
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Check top-level structure
        if isinstance(data, dict):
            print(f"Top-level type: Dictionary")
            print(f"Keys: {list(data.keys())}")
            
            # Check for 'data' key
            if 'data' in data:
                records = data['data']
                print(f"\nNumber of records in 'data': {len(records)}")
                
                if records and len(records) > 0:
                    # Inspect first record
                    first_record = records[0]
                    print(f"\nFirst record type: {type(first_record)}")
                    
                    if isinstance(first_record, dict):
                        print(f"First record keys ({len(first_record.keys())} total):")
                        # Show first 30 keys
                        for i, key in enumerate(sorted(first_record.keys())[:30]):
                            value = first_record[key]
                            value_type = type(value).__name__
                            if value is not None:
                                print(f"  {key:30s} : {value_type:10s} = {str(value)[:50]}")
                            else:
                                print(f"  {key:30s} : None")
                        
                        if len(first_record.keys()) > 30:
                            print(f"  ... and {len(first_record.keys()) - 30} more keys")
                        
                        # Check for Usable flag specifically
                        print(f"\nUsable flag check:")
                        print(f"  'Usable': {first_record.get('Usable', 'NOT FOUND')}")
                        print(f"  'usable': {first_record.get('usable', 'NOT FOUND')}")
                        print(f"  'USABLE': {first_record.get('USABLE', 'NOT FOUND')}")
                        
                        # Count usable records
                        usable_count = sum(1 for r in records if r.get('Usable', 0) == 1)
                        print(f"\nUsable records (Usable=1): {usable_count} / {len(records)}")
                        
                        # Check for important cost fields
                        print(f"\nCost fields check:")
                        print(f"  'TotalCost': {first_record.get('TotalCost', 'NOT FOUND')}")
                        print(f"  'TotalPaidClaims': {first_record.get('TotalPaidClaims', 'NOT FOUND')}")
                        print(f"  'PositiveCost': {first_record.get('PositiveCost', 'NOT FOUND')}")
                        
                        # Check TotalCost distribution
                        # Try different field names for cost
                        cost_field = None
                        for field in ['TotalCost', 'TotalPaidClaims', 'PositiveCost']:
                            if field in first_record:
                                cost_field = field
                                break
                        
                        if cost_field:
                            costs = []
                            for r in records[:100]:
                                cost_val = r.get(cost_field, 0)
                                try:
                                    costs.append(float(cost_val) if cost_val else 0)
                                except:
                                    costs.append(0)
                            
                            positive_costs = [c for c in costs if c > 0]
                            print(f"\n{cost_field} distribution (first 100 records):")
                            print(f"  Records with positive cost: {len(positive_costs)}")
                            if positive_costs:
                                print(f"  Min cost: ${min(positive_costs):,.0f}")
                                print(f"  Max cost: ${max(positive_costs):,.0f}")
                                print(f"  Avg cost: ${sum(positive_costs)/len(positive_costs):,.0f}")
                    
            # Check for other important keys
            for key in ['columns', 'fiscal_year', 'record_count', 'extraction_date']:
                if key in data:
                    print(f"\n{key}: {data[key]}")
                    
        else:
            print(f"Top-level type: {type(data)}")
            
    except Exception as e:
        print(f"Error reading file: {e}")


def main():
    """Main function to inspect cache files"""
    # Look for cache directory
    cache_dirs = [Path("models/data/cached")]
    
    cache_dir = None
    for dir_path in cache_dirs:
        if dir_path.exists():
            cache_dir = dir_path
            break
    
    if cache_dir is None:
        print("Cache directory not found!")
        print("Looked in:", cache_dirs)
        sys.exit(1)
    
    print(f"Found cache directory: {cache_dir.absolute()}")
    
    # List all pickle files
    pickle_files = sorted(cache_dir.glob("*.pkl"))
    
    if not pickle_files:
        print("No pickle files found in cache directory!")
        sys.exit(1)
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Inspect each file
    for pkl_file in pickle_files:
        inspect_pickle_file(pkl_file)
    
    print("\n" + "="*60)
    print("Inspection complete!")
    print("="*60)
    
    # Summary
    print("\nSummary:")
    print("--------")
    print("If you're seeing 'Usable records: 0', check that:")
    print("1. The stored procedure is setting Usable=1 correctly")
    print("2. The data extraction included the Usable flag")
    print("3. The data quality criteria in the stored procedure are not too strict")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect cached iBudget data')
    parser.add_argument('--file', help='Specific file to inspect (e.g., fy2020.pkl)')
    args = parser.parse_args()
    
    if args.file:
        filepath = Path(args.file)
        if filepath.exists():
            inspect_pickle_file(filepath)
        else:
            print(f"File not found: {filepath}")
    else:
        main()