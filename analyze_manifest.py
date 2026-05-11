#!/usr/bin/env python3
"""Analyze experiment manifest"""

import csv

def main():
    with open('configs/experiment_manifest_full.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f'Total rows: {len(rows)}')

        # Unique counts
        instances = set(row['instance_id'] for row in rows)
        scenarios = set(row['scenario_id'] for row in rows)
        methods = set(row['method_id'] for row in rows)
        seeds = set(row['seed'] for row in rows)

        print(f'Unique instances: {len(instances)}')
        print(f'Unique scenarios: {len(scenarios)}')
        print(f'Unique methods: {len(methods)}')
        print(f'Unique seeds: {len(seeds)}')

        expected = len(instances) * len(scenarios) * len(methods) * len(seeds)
        print(f'Expected total: {len(instances)} × {len(scenarios)} × {len(methods)} × {len(seeds)} = {expected}')
        print(f'Actual total: {len(rows)}')
        print(f'Match: {len(rows) == expected}')

        print(f'\nInstances: {sorted(instances)}')
        print(f'Scenarios: {sorted(scenarios)}')
        print(f'Methods: {sorted(methods)}')
        print(f'Seeds: {sorted(seeds)}')

if __name__ == '__main__':
    main()