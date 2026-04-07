#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import csv, sys
from pathlib import Path

REQUIRED = [
    'run_id','accepted_final','strict_duty_final','v_cap_final','v_tw_final','v_shift_final',
    'overtime_sum_final','j_scaled_init','j_scaled_final','compute_wh','imp_per_wh'
]

def main(path_str: str) -> int:
    path = Path(path_str)
    errors = []
    accepted_rows = 0
    with path.open() as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing columns: {missing}")
        for i, row in enumerate(reader, start=2):
            acc = int(float(row['accepted_final']))
            strict = int(float(row['strict_duty_final']))
            vcap = float(row['v_cap_final'])
            vtw = float(row['v_tw_final'])
            vshift = float(row['v_shift_final'])
            overtime = float(row['overtime_sum_final'])
            j_init = float(row['j_scaled_init'])
            j_final = float(row['j_scaled_final'])
            compute_wh = float(row['compute_wh'])
            imp = float(row['imp_per_wh'])
            if acc == 1:
                accepted_rows += 1
                if any(abs(v) > 1e-8 for v in (vcap, vtw, vshift)):
                    errors.append(f'line {i}: accepted row has nonzero violation(s)')
            if strict == 1 and not (acc == 1 and abs(overtime) <= 1e-8):
                errors.append(f'line {i}: strict_duty_final inconsistent')
            if j_final > j_init + 1e-6:
                errors.append(f'line {i}: j_scaled_final exceeds j_scaled_init')
            if compute_wh <= 0:
                errors.append(f'line {i}: compute_wh must be positive')
            if imp < 0:
                errors.append(f'line {i}: imp_per_wh must be nonnegative')
    if errors:
        print('VALIDATION FAILED')
        for e in errors[:200]:
            print('-', e)
        return 1
    print(f'VALIDATION PASSED: {accepted_rows} accepted rows checked')
    return 0

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "generated/master_runs.csv"
    raise SystemExit(main(path))