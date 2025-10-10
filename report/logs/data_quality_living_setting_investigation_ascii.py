import pickle
from collections import Counter
import sys

# Load FY2024 data
print("Loading FY2024 data...")
with open('models/data/cached/fy2024.pkl', 'rb') as f:
    data = pickle.load(f)

records = data['data']
print(f"Total records in FY2024: {len(records):,}\n")

# Filter to usable only with positive cost
usable = [r for r in records if r.get('Usable') == 1 and float(r.get('TotalCost', 0)) > 0]
print(f"Usable records (Usable=1 and TotalCost>0): {len(usable):,}\n")

# ============================================================
# CRITICAL INVESTIGATION: LivingSetting Distribution
# ============================================================
print("="*70)
print("LIVINGSETTING DISTRIBUTION (Aggregated by Stored Procedure)")
print("="*70)

living_settings = Counter([r.get('LivingSetting') for r in usable])
print(f"\nUnique LivingSetting values found: {len(living_settings)}")
print(f"\nDistribution:")
for setting, count in sorted(living_settings.items(), key=lambda x: x[1], reverse=True):
    pct = (count / len(usable)) * 100
    print(f"  {setting:15s}: {count:5,} ({pct:5.1f}%)")

# Check for RH categories
print("\n" + "="*70)
print("RH CATEGORY ANALYSIS")
print("="*70)
rh_variants = [s for s in living_settings.keys() if 'RH' in str(s)]
print(f"RH variants found in LivingSetting: {rh_variants}")
print(f"Number of RH variants: {len(rh_variants)}")

# ============================================================
# INVESTIGATE ORIGINAL RESIDENCETYPE FIELD
# ============================================================
print("\n" + "="*70)
print("ORIGINAL RESIDENCETYPE FIELD (Before Aggregation)")
print("="*70)

residence_types = Counter([r.get('RESIDENCETYPE') for r in usable])
print(f"\nUnique RESIDENCETYPE values found: {len(residence_types)}")
print(f"\nTop 20 most common RESIDENCETYPE values:")
for res_type, count in residence_types.most_common(20):
    pct = (count / len(usable)) * 100
    print(f"  {str(res_type)[:50]:50s}: {count:5,} ({pct:5.1f}%)")

# Check for RH patterns in original field
print("\n" + "="*70)
print("RH PATTERNS IN RESIDENCETYPE")
print("="*70)
rh_residence = [s for s in residence_types.keys() if 'RH' in str(s).upper() or 'RESIDENTIAL' in str(s).upper()]
print(f"RESIDENCETYPE values containing 'RH' or 'RESIDENTIAL': {len(rh_residence)}")
for res_type in sorted(rh_residence):
    count = residence_types[res_type]
    pct = (count / len(usable)) * 100
    print(f"  {str(res_type)[:50]:50s}: {count:5,} ({pct:5.1f}%)")

# ============================================================
# CROSS-REFERENCE: What RESIDENCETYPE maps to each LivingSetting?
# ============================================================
print("\n" + "="*70)
print("MAPPING: RESIDENCETYPE -> LivingSetting")
print("="*70)

mapping = {}
for r in usable:
    res_type = r.get('RESIDENCETYPE')
    living_setting = r.get('LivingSetting')
    if living_setting not in mapping:
        mapping[living_setting] = Counter()
    mapping[living_setting][res_type] += 1

for living_setting in sorted(mapping.keys()):
    print(f"\n{living_setting} is created from:")
    for res_type, count in mapping[living_setting].most_common(10):
        pct = (count / sum(mapping[living_setting].values())) * 100
        print(f"  {str(res_type)[:50]:50s}: {count:5,} ({pct:5.1f}%)")

# ============================================================
# DIAGNOSTIC: Sample records for each LivingSetting
# ============================================================
print("\n" + "="*70)
print("SAMPLE RECORDS")
print("="*70)

for living_setting in sorted(living_settings.keys()):
    sample = next((r for r in usable if r.get('LivingSetting') == living_setting), None)
    if sample:
        print(f"\n{living_setting} example:")
        print(f"  RESIDENCETYPE: {sample.get('RESIDENCETYPE')}")
        print(f"  TotalCost: ${float(sample.get('TotalCost', 0)):,.2f}")
        print(f"  FSum: {sample.get('FSum')}, BSum: {sample.get('BSum')}, PSum: {sample.get('PSum')}")

print("\n" + "="*70)
print("INVESTIGATION COMPLETE")
print("="*70)
