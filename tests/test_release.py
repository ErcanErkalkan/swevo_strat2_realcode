from pathlib import Path
import sys
S=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(S))
from release_contract import SUPPLEMENT_FILES,PUBLIC_FILES,RAW_INPUTS,TESTS
def test_reviewer_allowlist_has_no_operational_or_portal_assets():
    assert len(SUPPLEMENT_FILES)==len(set(SUPPLEMENT_FILES))
    assert not any('manuscript/' in s or '/logs/' in s or s.endswith('.log') or 'SOURCE_LOCK_PREFLIGHT' in s for s in SUPPLEMENT_FILES)
    assert 'data/primary/runs.csv' in SUPPLEMENT_FILES
def test_public_tree_excludes_raw_and_private_assets():
    assert not RAW_INPUTS.intersection(PUBLIC_FILES)
    assert not any(s.startswith(('manuscript/','data/ablation/','results/ablation/','archive/')) or s.endswith(('.pdf','.zip','.gdoc')) for s in PUBLIC_FILES)
def test_required_core_reference_and_templates_are_packaged():
    for s in ['data/reference/protocol_sensitivity.csv','templates/claim_evidence_map_template.csv','templates/master_runs_template.csv','requirements.lock.txt','scripts/core_contract.py']:assert s in SUPPLEMENT_FILES
def test_audit_regressions_are_shipped():
    assert 'test_ablation_audit.py' in TESTS and 'test_provenance.py' in TESTS
