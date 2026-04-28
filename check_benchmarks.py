#!/usr/bin/env python3
"""Check benchmark instance availability"""

import csv
import os
import glob
import sys

# Manifest'te bulunan instance'ları al
print("=" * 60)
print("BENCHMARK VALIDATION REPORT")
print("=" * 60)

try:
    manifest_file = 'configs/experiment_manifest_paper_main3_real_ready.csv'
    with open(manifest_file) as f:
        reader = csv.DictReader(f)
        manifest_instances = sorted(set(row['instance_id'] for row in reader))
    
    print(f"\n✓ Manifest file loaded: {manifest_file}")
    print(f"  Total unique instances: {len(manifest_instances)}")
    print(f"  Instances: {manifest_instances}")
except Exception as e:
    print(f"❌ Error reading manifest: {e}")
    sys.exit(1)

# Benchmark dosyalarında bulunan instance'ları al
print("\nScanning benchmark files...")
benchmark_files = glob.glob('data/benchmarks/**/*.txt', recursive=True)
found_instances = {}

for filepath in benchmark_files:
    basename = os.path.basename(filepath).replace('.txt', '')
    if basename not in found_instances:
        found_instances[basename] = []
    found_instances[basename].append(filepath)

print(f"✓ Total benchmark files found: {len(benchmark_files)}")
print(f"  Unique instances: {len(found_instances)}")

# Missing instances kontrol et
missing = set(manifest_instances) - set(found_instances.keys())
if missing:
    print(f"\n❌ MISSING instances (in manifest but not in benchmarks):")
    for inst in sorted(missing):
        print(f"   - {inst}")
else:
    print(f"\n✓ All manifest instances found in benchmarks")

# Extra instances kontrol et  
extra = set(found_instances.keys()) - set(manifest_instances)
if extra:
    print(f"\n⚠ Extra benchmark instances (in benchmarks but not in manifest):")
    for inst in sorted(list(extra)[:20]):
        print(f"   - {inst}")
    if len(extra) > 20:
        print(f"   ... and {len(extra) - 20} more")

# Stale instances check
print(f"\n📋 Checking for potentially stale instances (RC109, etc)...")
stale_patterns = ['C109', 'R109', 'RC109']
found_stale = [inst for inst in manifest_instances if any(p in inst for p in stale_patterns)]
if found_stale:
    print(f"   Found potentially old instances: {found_stale}")
    print(f"   ⚠ Note: These may be valid extended instances (C1_xx, R1_xx)")

print("\n" + "=" * 60)
print("END OF REPORT")
print("=" * 60)
