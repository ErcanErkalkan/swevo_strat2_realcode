import json,sys
from pathlib import Path
import pandas as pd
import pytest
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
from core_contract import input_paths,OUTPUTS
from compare_outputs import compare_csv

def test_reference_and_templates_are_inputs():
    paths=input_paths(SCRIPTS.parent);roles={p.relative_to(SCRIPTS.parent).as_posix():role for p,role in paths}
    assert roles['data/reference/protocol_sensitivity.csv']=='frozen_reference'
    assert roles['templates/claim_evidence_map_template.csv']=='template'
    assert roles['templates/master_runs_template.csv']=='template'
    assert 'master_runs.csv' not in OUTPUTS and 'protocol_sensitivity.csv' not in OUTPUTS
    assert set(p for p,_ in paths).isdisjoint(SCRIPTS.parent/'results/primary'/name for name in OUTPUTS)

def test_line_endings_and_last_bit_are_not_byte_integrity(tmp_path):
    a=tmp_path/'a.csv';b=tmp_path/'b.csv'
    a.write_bytes(b'id,score,p_value\r\na,0.75,1e-20\r\n')
    b.write_bytes(b'id,score,p_value\na,0.7500000000000001,1.0000000000000001e-20\n')
    r=compare_csv(a,b);assert not r['byte_identical'] and r['numerically_equivalent']
def test_changed_effect_or_tiny_p_value_fails(tmp_path):
    a=tmp_path/'a.csv';b=tmp_path/'b.csv';a.write_text('score,p_value\n0.75,1e-20\n')
    b.write_text('score,p_value\n0.751,1e-20\n');assert not compare_csv(a,b)['numerically_equivalent']
    b.write_text('score,p_value\n0.75,2e-20\n');assert not compare_csv(a,b)['numerically_equivalent']
def test_python_minimum_matches_locked_runtime():
    import tomllib
    p=tomllib.loads((SCRIPTS.parent/'pyproject.toml').read_text())
    assert p['project']['requires-python']=='>=3.11'
def test_final_inference_refuses_partial(tmp_path):
    from analyze_ablation import require_final
    # Exact test grid, intentionally no scientific outcomes.
    rows=[]
    for i in range(9):
        for j in range(3):
            for seed in range(1,11):
                for m in ['EDE','A1_NoSeed','A2_NoJDE','A3_NoLNS']:
                    rows.append(dict(phase='test',benchmark_family='test',customer_count=100,instance_id=f'TEST{i}',structure_class='test',scenario_id=f'S{j}',method_id=m,seed=seed,tier='small',eval_budget=25050,walltime_cap_s=1800))
    pd.DataFrame(rows).to_csv(tmp_path/'confirmatory_ablation_manifest_1080_cli.csv',index=False)
    with pytest.raises(ValueError,match='Final ablation inference blocked'):require_final(tmp_path,tmp_path/'out')
