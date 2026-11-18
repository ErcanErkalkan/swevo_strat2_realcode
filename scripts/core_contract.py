"""Explicit evidence inputs/outputs. A frozen reference is never a generated result."""
CONFIG_INPUTS=['experiment_manifest_submission_primary.csv', 'benchmark_inventory_submission_primary.csv', 'budget_policy.yaml', 'method_registry.yaml', 'scenario_registry.csv', 'metrics_schema.yaml', 'stats_plan.yaml']
SECONDARY_INPUTS=['operational_shift_stress_locked.csv', 'solomon_equal_3shift_feasibility_audit.csv', 'parameter_sensitivity_locked.csv', 'anytime_profile_locked.csv']
TEMPLATE_INPUTS=['claim_evidence_map_template.csv','master_runs_template.csv']
OUTPUTS=[
 'summary_by_method.csv','summary_by_method_accepted_only.csv','master_stats.csv','master_stats_accepted_only.csv',
 'table_feasibility_summary.tex','table_feasibility_summary_accepted_only.tex','table_friedman_omnibus.tex','table_friedman_omnibus_accepted_only.tex','table_posthoc_holm.tex','table_posthoc_holm_accepted_only.tex',
 'robustness_by_scenario.csv','robustness_by_structure.csv','tier_native_endpoint_reductions.csv','secondary_endpoint_stats.csv','claim_macros.tex','claim_evidence_map.csv',
 'operational_shift_stress_summary.csv','table_operational_shift_stress.tex','operational_shift_evidence_provenance.json',
 'parameter_sensitivity_summary.csv','parameter_sensitivity_stats.csv','table_parameter_sensitivity.tex','parameter_sensitivity_evidence_provenance.json',
 'anytime_profile_summary.csv','anytime_profile_auc.csv','anytime_profile_overall.csv','table_anytime_profile.tex','anytime_profile_evidence_provenance.json',
 'cell_medians.csv','instance_medians.csv','inference.csv','cluster_bootstrap.csv','primary_runtime_deviations.csv','method_tier_summary.csv','runtime_by_method_tier.csv','primary_audit.json','table_instance_sensitivity.tex','table_cluster_ci.tex','table_tier_comparison.tex','macros.tex','tier_effects.json']
CORE_SCRIPTS=['validate_master_runs.py','aggregate_results.py','run_stats.py','build_latex_tables.py','build_robustness_evidence.py','build_claim_macros.py','build_claim_evidence_map.py','build_operational_shift_evidence.py','build_parameter_sensitivity_evidence.py','build_anytime_profile_evidence.py','analyze_primary.py','core_contract.py','finalize.py']
def input_paths(root):
    return [(root/'data/primary/runs.csv','raw_primary'),(root/'data/reference/protocol_sensitivity.csv','frozen_reference'),(root/'requirements.lock.txt','environment_lock'),(root/'pyproject.toml','environment_declaration'),*[(root/'configs'/name,'locked_config') for name in CONFIG_INPUTS],*[(root/'data/secondary'/name,'locked_secondary') for name in SECONDARY_INPUTS],*[(root/'templates'/name,'template') for name in TEMPLATE_INPUTS],*[(root/'scripts'/name,'generator_code') for name in CORE_SCRIPTS],*[(p,'source_code') for p in sorted((root/'src').rglob('*.py'))]]
