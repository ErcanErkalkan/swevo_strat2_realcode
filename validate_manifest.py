#!/usr/bin/env python3
"""Validate manifest against benchmarks and registries"""

import csv
import yaml
import os

def main():
    print("=" * 80)
    print("MANIFEST VALIDATION REPORT")
    print("=" * 80)

    # Load manifest
    with open('configs/experiment_manifest_full.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    manifest_instances = set(row['instance_id'] for row in rows)
    manifest_scenarios = set(row['scenario_id'] for row in rows)
    manifest_methods = set(row['method_id'] for row in rows)
    manifest_seeds = set(row['seed'] for row in rows)

    print(f"Manifest contains:")
    print(f"  - {len(manifest_instances)} instances")
    print(f"  - {len(manifest_scenarios)} scenarios")
    print(f"  - {len(manifest_methods)} methods")
    print(f"  - {len(manifest_seeds)} seeds")
    print(f"  - {len(rows)} total runs")

    # Load method registry
    with open('configs/method_registry.yaml') as f:
        registry = yaml.safe_load(f)

    registry_methods = set(registry['methods'].keys())
    print(f"\nMethod registry contains: {len(registry_methods)} methods")

    # Check method consistency
    missing_methods = manifest_methods - registry_methods
    extra_methods = registry_methods - manifest_methods

    if missing_methods:
        print(f"❌ Methods in manifest but not in registry: {missing_methods}")
    else:
        print("✓ All manifest methods found in registry")

    if extra_methods:
        print(f"⚠ Methods in registry but not used in manifest: {extra_methods}")

    # Load scenario registry
    with open('configs/scenario_registry.csv') as f:
        reader = csv.DictReader(f)
        registry_scenarios = set(row['scenario_id'] for row in reader)

    print(f"\nScenario registry contains: {len(registry_scenarios)} scenarios")

    # Check scenario consistency
    missing_scenarios = manifest_scenarios - registry_scenarios
    extra_scenarios = registry_scenarios - manifest_scenarios

    if missing_scenarios:
        print(f"❌ Scenarios in manifest but not in registry: {missing_scenarios}")
    else:
        print("✓ All manifest scenarios found in registry")

    if extra_scenarios:
        print(f"⚠ Scenarios in registry but not used in manifest: {extra_scenarios}")

    # Check benchmark files
    benchmark_files = []
    for root, dirs, files in os.walk('data/benchmarks'):
        for file in files:
            if file.endswith('.txt'):
                benchmark_files.append(file.replace('.txt', ''))

    benchmark_instances = set(benchmark_files)
    print(f"\nBenchmark files contain: {len(benchmark_instances)} instances")

    # Check instance consistency
    missing_instances = manifest_instances - benchmark_instances
    extra_instances = benchmark_instances - manifest_instances

    if missing_instances:
        print(f"❌ Instances in manifest but no benchmark file: {missing_instances}")
    else:
        print("✓ All manifest instances have benchmark files")

    if extra_instances:
        print(f"⚠ Benchmark instances not used in manifest: {len(extra_instances)} total")
        print(f"   Sample: {sorted(list(extra_instances))[:10]}...")

    # Check seed range
    seeds_int = sorted([int(s) for s in manifest_seeds])
    expected_seeds = list(range(1, 21))  # 1-20

    if seeds_int == expected_seeds:
        print("✓ Seeds are correctly 1-20")
    else:
        print(f"❌ Seed mismatch. Expected: {expected_seeds}, Got: {seeds_int}")

    # Check total calculation
    expected_total = len(manifest_instances) * len(manifest_scenarios) * len(manifest_methods) * len(manifest_seeds)
    if len(rows) == expected_total:
        print(f"✓ Total runs match: {len(rows)}")
    else:
        print(f"❌ Total mismatch. Expected: {expected_total}, Got: {len(rows)}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()