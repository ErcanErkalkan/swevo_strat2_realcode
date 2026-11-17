from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_submission_gates.py"


def _load_script_module():
    spec = spec_from_file_location("check_submission_gates", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gate_helpers_return_expected_status_labels() -> None:
    module = _load_script_module()
    assert module._pass(1, "x", "y")["status"] == "pass"
    assert module._fail(1, "x", "y")["status"] == "fail"
    assert module._manual(1, "x", "y")["status"] == "manual_review_needed"


def test_write_markdown_renders_status_table(tmp_path: Path) -> None:
    module = _load_script_module()
    rows = [
        {"item_id": 1, "title": "Example gate", "status": "pass", "notes": "ok"},
        {"item_id": 2, "title": "Another gate", "status": "manual_review_needed", "notes": "manual"},
    ]
    out = tmp_path / "report.md"
    module.write_markdown(out, rows)
    text = out.read_text(encoding="utf-8")
    assert "Submission Gating Report" in text
    assert "| 1. Example gate | pass | ok |" in text


def test_scientific_tex_formatter_matches_manuscript_style() -> None:
    module = _load_script_module()
    assert module._sci_tex(5.615298190733405e-69) == r"5.62\times10^{-69}"
    assert module._sci_tex(1.8686821759772635e-19) == r"1.87\times10^{-19}"


@pytest.mark.skipif(not (ROOT / "manuscript/elsarticle-template-num.tex").is_file(), reason="Manuscript assets are supplied in the C15 supplement")
def test_quantitative_manuscript_fragments_match_regenerated_evidence() -> None:
    """C15 consumes generated tables rather than C14 literal table sentences."""
    generated = ROOT / "generated_submission"
    manuscript = (ROOT / "manuscript" / "elsarticle-template-num.tex").read_text(encoding="utf-8")
    tables=['table_c15_tier_comparison.tex','table_c15_instance_sensitivity.tex','table_c15_cluster_ci.tex']
    for name in tables:
        assert r'\input{../generated_submission/'+name+'}' in manuscript
        assert (generated/name).stat().st_size>100
    summary=pd.read_csv(generated/'summary_by_method.csv')
    tier=(generated/'table_c15_tier_comparison.tex').read_text()
    for size in ['small','medium','large']:
        scores=summary[summary.tier==size].set_index('method_id').median_j
        best=scores.drop('EDE').idxmin()
        assert f'{scores.EDE:.3f}' in tier and f'{scores[best]:.3f}' in tier
        assert f'{(1-scores.EDE/scores[best])*100:.1f}' in tier
    stats=pd.read_csv(generated/'c15_inference.csv')
    instance=(generated/'table_c15_instance_sensitivity.tex').read_text()
    for row in stats[(stats.scope=='36_instances')&(stats.test=='Wilcoxon')].itertuples():
        assert f'{row.median_diff:.5f}' in instance and f'{row.p_holm:.3g}' in instance
    assert '108 independent' not in manuscript


@pytest.mark.skipif(not (ROOT / "manuscript/c15_build_provenance.json").is_file(), reason="Manuscript assets are supplied in the C15 supplement")
def test_manuscript_pdf_build_provenance_matches_sources() -> None:
    import hashlib,json
    record=json.loads((ROOT/'manuscript/c15_build_provenance.json').read_text())
    assert record['schema']=='c15-build-v1' and 'bibtex' in record['commands']
    for rel,value in record['files'].items():
        if rel=='manuscript/COVER_LETTER.txt':continue # separate portal-only check
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==value,rel



def test_manuscript_build_provenance_requires_bst_and_bibtex_enabled_build(tmp_path: Path) -> None:
    import hashlib
    import json

    module = _load_script_module()
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    files = {
        "elsarticle-template-num.tex": b"\\documentclass{article}\n",
        "cas-refs.bib": b"@article{x,title={X}}\n",
        "elsarticle-num.bst": b"ENTRY{}{}{}\n",
        "manuscript.pdf": b"%PDF-1.4\n",
    }
    for name, data in files.items():
        (manuscript / name).write_bytes(data)
    h = lambda name: hashlib.sha256(files[name]).hexdigest()
    provenance = {
        "schema": "swevo-manuscript-build-provenance-v3",
        "tex_file": "elsarticle-template-num.tex",
        "tex_sha256": h("elsarticle-template-num.tex"),
        "bib_file": "cas-refs.bib",
        "bib_sha256": h("cas-refs.bib"),
        "bst_file": "elsarticle-num.bst",
        "bst_sha256": h("elsarticle-num.bst"),
        "pdf_file": "manuscript.pdf",
        "pdf_sha256": h("manuscript.pdf"),
        "pages": 1,
        "build_command": "latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error elsarticle-template-num.tex",
        "source_date_epoch": 1788436800,
        "force_source_date": True,
    }
    (manuscript / "build_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    ok, notes = module._manuscript_build_provenance_ok(manuscript)
    assert ok, notes

    (manuscript / "elsarticle-num.bst").write_bytes(b"changed")
    ok, notes = module._manuscript_build_provenance_ok(manuscript)
    assert not ok and "bst_sha256" in notes

    (manuscript / "elsarticle-num.bst").write_bytes(files["elsarticle-num.bst"])
    provenance["build_command"] = provenance["build_command"].replace("-bibtex ", "-bibtex- ")
    (manuscript / "build_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    ok, notes = module._manuscript_build_provenance_ok(manuscript)
    assert not ok and "BibTeX-enabled" in notes

def test_runtime_cap_gate_rejects_invalid_runtime_values() -> None:
    module = _load_script_module()
    manifest = [{"run_id": "A", "seed": "1", "walltime_cap_s": "7200"}]
    for value in [float("nan"), float("inf"), 0.0, -1.0]:
        bad, integrity_issues = module._runtime_cap_violations(
            pd.DataFrame([{"run_id": "A", "runtime_sec": value}]),
            manifest,
            10.0,
        )
        assert bad.empty
        assert integrity_issues == ["A"]


def test_runtime_cap_gate_enforces_tolerance_boundary() -> None:
    module = _load_script_module()
    manifest = [{"run_id": "A", "seed": "1", "walltime_cap_s": "7200"}]
    at_boundary, issues = module._runtime_cap_violations(
        pd.DataFrame([{"run_id": "A", "runtime_sec": 7210.0}]),
        manifest,
        10.0,
    )
    assert at_boundary.empty
    assert not issues

    over_boundary, issues = module._runtime_cap_violations(
        pd.DataFrame([{"run_id": "A", "runtime_sec": 7210.001}]),
        manifest,
        10.0,
    )
    assert len(over_boundary) == 1
    assert not issues


def test_portal_mutable_numeric_guard_rejects_rerun_sensitive_literals() -> None:
    module = _load_script_module()
    issues = module._portal_mutable_numeric_issues(
        "Reduction is 18.0%.",
        "Friedman W=0.751.",
        "p=0.001.",
    )
    assert "percentage/effect-size literal" in issues
    assert "Kendall/Friedman W literal" in issues
    assert "p-value literal" in issues


def test_portal_mutable_numeric_guard_accepts_stable_design_language() -> None:
    module = _load_script_module()
    issues = module._portal_mutable_numeric_issues(
        "10,800 runs across 36 instances and five methods.",
        "108 block medians with Friedman omnibus and Wilcoxon-Holm post hoc.",
        "All five methods finish hard-feasible and strict-duty compliant.",
    )
    assert issues == []


def test_highlights_txt_docx_normalization_contract(tmp_path: Path) -> None:
    import zipfile

    module = _load_script_module()
    docx = tmp_path / "Highlights.docx"
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>Highlights</w:t></w:r></w:p>
<w:p><w:r><w:t>First stable highlight.</w:t></w:r></w:p>
<w:p><w:r><w:t>Second stable highlight.</w:t></w:r></w:p>
</w:body></w:document>'''
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", xml)
    txt = "Highlights\n\n• First stable highlight.\n• Second stable highlight.\n"
    docx_lines = module._read_docx_paragraphs(docx)
    assert module._normalized_highlight_lines(txt) == module._normalized_highlight_lines("\n".join(docx_lines))


def test_graphical_abstract_provenance_matches_source_pdf_png(tmp_path: Path) -> None:
    import hashlib
    import json

    module = _load_script_module()
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    files = {
        "graphical_abstract_source.tex": b"source",
        "graphical_abstract.pdf": b"pdf",
        "graphical_abstract.png": b"png",
    }
    for name, data in files.items():
        (manuscript / name).write_bytes(data)
    h = lambda name: hashlib.sha256(files[name]).hexdigest()
    provenance = {
        "schema": "swevo-graphical-abstract-provenance-v1",
        "source_file": "graphical_abstract_source.tex",
        "source_sha256": h("graphical_abstract_source.tex"),
        "pdf_file": "graphical_abstract.pdf",
        "pdf_sha256": h("graphical_abstract.pdf"),
        "png_file": "graphical_abstract.png",
        "png_sha256": h("graphical_abstract.png"),
        "png_width_px": 1140,
        "png_height_px": 412,
        "raster_dpi": 220,
        "compile_command": "latexmk -pdf -interaction=nonstopmode -halt-on-error graphical_abstract_source.tex",
        "raster_command": "pdftoppm -png -r 220 -singlefile graphical_abstract.pdf graphical_abstract",
        "source_date_epoch": 1788436800,
        "force_source_date": True,
    }
    (manuscript / "graphical_abstract_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    ok, notes = module._graphical_abstract_provenance_ok(manuscript)
    assert ok, notes
    (manuscript / "graphical_abstract.png").write_bytes(b"changed")
    ok, notes = module._graphical_abstract_provenance_ok(manuscript)
    assert not ok and "png_sha256" in notes


def _make_runtime_exception_fixture(tmp_path: Path, module):
    import hashlib
    import json

    generated = tmp_path / "generated_submission"
    generated.mkdir()
    module.GENERATED = generated

    ids = sorted(module.LOCKED_RUNTIME_EXCEPTION_IDS)
    runtimes = {
        "R1_4_5__S1_balanced__EDE__seed20": 10663.3076682,
        "R1_4_5__S1_balanced__StdDE__seed20": 10655.5799634,
        "R1_4_5__S1_balanced__ALNS_MS__seed20": 10617.8735869,
        "R1_4_5__S1_balanced__HGS_MS__seed20": 10656.4598120,
        "R1_4_5__S1_balanced__ILS_MS__seed20": 10658.9589841,
    }
    runtime_bad = pd.DataFrame([
        {"run_id": rid, "runtime_sec": runtimes[rid], "walltime_cap_s": 7200.0}
        for rid in ids
    ])
    master = pd.DataFrame([
        {"run_id": rid, "runtime_sec": runtimes[rid]}
        for rid in ids
    ])
    master.to_csv(generated / "master_runs.csv", index=False)

    deviations = runtime_bad.copy()
    deviations["instance_id"] = "R1_4_5"
    deviations["scenario_id"] = "S1_balanced"
    deviations["method_id"] = deviations["run_id"].str.extract(r"S1_balanced__(.*?)__seed20", expand=False)
    deviations["seed"] = 20
    deviations["tier"] = "large"
    deviations[["run_id", "instance_id", "scenario_id", "method_id", "seed", "tier", "runtime_sec", "walltime_cap_s"]].to_csv(
        generated / "runtime_cap_deviations.csv", index=False
    )

    rows = []
    for scope in ("all_10800", "cap_compliant_10795"):
        rows.append({
            "scope": scope, "section": "omnibus", "metric": "j_scaled_final",
            "control_method": None, "compare_method": None, "blocks": 108, "k_methods": 5,
            "statistic": 324.49629629629635, "p_value": 5.615298190733405e-69,
            "kendalls_w": 0.7511488340192045, "median_diff_ede_minus_compare": None, "p_holm": None,
        })
        pair = {
            "StdDE": (0.0, 1.8686821759772635e-19, -0.24838799275957846, 7.474728703909054e-19),
            "ALNS_MS": (0.0, 1.8686821759772635e-19, -0.14312574580128457, 7.474728703909054e-19),
            "HGS_MS": (56.0, 8.825514475370894e-19, -0.126836776244031, 8.825514475370894e-19),
            "ILS_MS": (14.0, 2.7623087555720136e-19, -0.1370404336694998, 7.474728703909054e-19),
        }
        for comp, (stat, p, diff, holm) in pair.items():
            rows.append({
                "scope": scope, "section": "pairwise", "metric": "j_scaled_final",
                "control_method": "EDE", "compare_method": comp, "blocks": 108, "k_methods": 5,
                "statistic": stat, "p_value": p, "kendalls_w": None,
                "median_diff_ede_minus_compare": diff, "p_holm": holm,
            })
    pd.DataFrame(rows).to_csv(generated / "protocol_sensitivity.csv", index=False)

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "schema": "ede-submission-strengthening-evidence-v1",
        "primary_source": {"rows": len(master), "sha256": sha(generated / "master_runs.csv")},
        "outputs": {
            "runtime_cap_deviations.csv": sha(generated / "runtime_cap_deviations.csv"),
            "protocol_sensitivity.csv": sha(generated / "protocol_sensitivity.csv"),
        },
    }
    (generated / "strengthening_evidence_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    manuscript = (
        "Documented synchronized runtime-cap deviation. "
        "The raw rows are preserved unchanged; the sensitivity set contains 10,795 cap-compliant records "
        "and does not rewrite the historical campaign."
    )
    return runtime_bad, master, manuscript, generated


def test_runtime_protocol_exception_accepts_only_locked_five_with_invariant_sensitivity(tmp_path: Path) -> None:
    module = _load_script_module()
    runtime_bad, master, manuscript, _ = _make_runtime_exception_fixture(tmp_path, module)
    ok, notes = module._runtime_protocol_exception_ok(runtime_bad, master, manuscript)
    assert ok, notes


def test_runtime_protocol_exception_rejects_unexpected_offender(tmp_path: Path) -> None:
    module = _load_script_module()
    runtime_bad, master, manuscript, _ = _make_runtime_exception_fixture(tmp_path, module)
    runtime_bad = pd.concat([runtime_bad, pd.DataFrame([{"run_id": "unexpected", "runtime_sec": 8000.0, "walltime_cap_s": 7200.0}])], ignore_index=True)
    ok, notes = module._runtime_protocol_exception_ok(runtime_bad, master, manuscript)
    assert not ok and "locked synchronized five" in notes


def test_runtime_protocol_exception_rejects_missing_artifact(tmp_path: Path) -> None:
    module = _load_script_module()
    runtime_bad, master, manuscript, generated = _make_runtime_exception_fixture(tmp_path, module)
    (generated / "protocol_sensitivity.csv").unlink()
    ok, notes = module._runtime_protocol_exception_ok(runtime_bad, master, manuscript)
    assert not ok and "missing runtime-sensitivity artifacts" in notes


def test_runtime_protocol_exception_rejects_changed_sensitivity(tmp_path: Path) -> None:
    module = _load_script_module()
    runtime_bad, master, manuscript, generated = _make_runtime_exception_fixture(tmp_path, module)
    sensitivity = pd.read_csv(generated / "protocol_sensitivity.csv")
    mask = (sensitivity["scope"] == "cap_compliant_10795") & (sensitivity["section"] == "omnibus")
    sensitivity.loc[mask, "statistic"] = 999.0
    sensitivity.to_csv(generated / "protocol_sensitivity.csv", index=False)
    ok, notes = module._runtime_protocol_exception_ok(runtime_bad, master, manuscript)
    assert not ok and ("omnibus sensitivity changed" in notes or "output hash mismatch" in notes)
