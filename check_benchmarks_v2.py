#!/usr/bin/env python3
"""Check benchmark instance availability - Fixed version"""

import csv
import os
import glob
import sys

# Manifest'te bulunan instance'ları al
print("=" * 70)
print("BENCHMARK VALIDATION REPORT")
print("=" * 70)

try:
    manifest_file = 'configs/experiment_manifest_paper_main3_real_ready.csv'
    with open(manifest_file) as f:
        reader = csv.DictReader(f)
        manifest_instances = sorted(set(row['instance_id'].lower() for row in reader))
    
    print(f"\n✓ Manifest file loaded: {manifest_file}")
    print(f"  Total unique instances: {len(manifest_instances)}")
    print(f"  Instances: {manifest_instances}")
except Exception as e:
    print(f"❌ Error reading manifest: {e}")
    sys.exit(1)

# Benchmark dosyalarında bulunan instance'ları al
print("\nScanning benchmark files...")
benchmark_files = glob.glob('data/benchmarks/**/*.txt', recursive=True)
found_instances = set()

for filepath in benchmark_files:
    basename = os.path.basename(filepath)
    # Remove extension and convert to lowercase
    instance = basename.lower().replace('.txt', '')
    found_instances.add(instance)

print(f"✓ Total benchmark files found: {len(benchmark_files)}")
print(f"  Unique instances: {len(found_instances)}")

# Sample of found instances
print(f"\n  Sample benchmark instances found:")
for inst in sorted(list(found_instances))[:20]:
    print(f"    - {inst}")

# Missing instances kontrol et
missing = set(manifest_instances) - found_instances
if missing:
    print(f"\n❌ MISSING instances ({len(missing)} total):")
    for inst in sorted(missing):
        print(f"   - {inst}")
else:
    print(f"\n✓ All {len(manifest_instances)} manifest instances found in benchmarks")

# Extra instances kontrol et  
extra = found_instances - set(manifest_instances)
if extra:
    print(f"\n⚠ Extra benchmark instances ({len(extra)} total, not in manifest):")
    for inst in sorted(list(extra)[:20]):
        print(f"   - {inst}")
    if len(extra) > 20:
        print(f"   ... and {len(extra) - 20} more")

# Stale instances check
print(f"\n📋 Checking for potentially stale instances...")
stale_patterns = ['c109', 'r109']  # lowercase
found_stale = [inst for inst in manifest_instances if any(p in inst for p in stale_patterns)]
if found_stale:
    print(f"   Found potentially old instances: {found_stale}")
    print(f"   ⚠ NOTE: These may be valid extended instances")
    print(f"   Solomon standard set includes:")
    print(f"      - C101, C102, ... C109 (9 instances)")
    print(f"      - R101, R102, ... R112 (12 instances)")
    print(f"      - RC101, RC102, ... RC108 (8 instances)")
else:
    print(f"   No classic 'stale' instances found")

print("\n" + "=" * 70)
print("SUMMARY:")
print(f"  Manifest instances: {len(manifest_instances)}")
print(f"  Benchmark instances: {len(found_instances)}")
print(f"  Missing: {len(missing)}")
print(f"  Extra: {len(extra)}")
print("=" * 70)
