from pathlib import Path
import csv
import subprocess
import sys

def test_template_has_header():
    path = Path(__file__).resolve().parents[1] / "templates" / "master_runs_template.csv"
    header = path.read_text().strip().split(",")
    assert "accepted_final" in header
    assert "strict_duty_final" in header
