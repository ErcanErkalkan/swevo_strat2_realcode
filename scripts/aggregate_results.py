#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from swevo_suite.paths import GENERATED, ensure_generated_dirs
from swevo_suite.stats import load_runs, summary_by_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('src', nargs='?', default=str(GENERATED / 'master_runs.csv'))
    parser.add_argument('dst', nargs='?', default=str(GENERATED / 'summary_by_method.csv'))
    args = parser.parse_args()
    ensure_generated_dirs()
    df = load_runs(Path(args.src))
    out = summary_by_method(df)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"Wrote {dst}")


if __name__ == '__main__':
    main()
