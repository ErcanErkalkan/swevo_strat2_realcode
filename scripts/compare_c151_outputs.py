"""Separate byte integrity from numerical equivalence; never rewrite evidence."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from c151_core_contract import OUTPUTS
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def compare_csv(reference,actual):
    a=pd.read_csv(reference,float_precision='round_trip');b=pd.read_csv(actual,float_precision='round_trip')
    report=dict(reference_sha256=sha(reference),actual_sha256=sha(actual),byte_identical=sha(reference)==sha(actual),numerically_equivalent=True,max_abs_difference=0.,issues=[])
    if list(a.columns)!=list(b.columns) or a.shape!=b.shape:
        report.update(numerically_equivalent=False,issues=['schema_or_shape']);return report
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            x=a[col].to_numpy(float);y=b[col].to_numpy(float)
            if not np.array_equal(np.isnan(x),np.isnan(y)):report['issues'].append(col+':missingness')
            mask=np.isfinite(x)&np.isfinite(y)
            absolute=1e-300 if col.startswith('p_') or 'pvalue' in col else 1e-12
            if not np.allclose(x,y,rtol=1e-10,atol=absolute,equal_nan=True):report['issues'].append(col+':numeric')
            if mask.any():report['max_abs_difference']=max(report['max_abs_difference'],float(np.max(np.abs(x[mask]-y[mask]))))
        elif not a[col].fillna('<NA>').astype(str).equals(b[col].fillna('<NA>').astype(str)):report['issues'].append(col+':text')
    report['numerically_equivalent']=not report['issues'];return report
def compare(reference,actual):
    result={}
    for name in OUTPUTS:
        a=reference/name;b=actual/name
        if name.endswith('.csv'):result[name]=compare_csv(a,b)
        else:result[name]=dict(reference_sha256=sha(a),actual_sha256=sha(b),byte_identical=sha(a)==sha(b),newline_normalized_identical=a.read_text(encoding='utf-8').replace('\r\n','\n')==b.read_text(encoding='utf-8').replace('\r\n','\n'))
    return dict(schema='c15.1-equivalence-v1',byte_identical_files=sum(v['byte_identical'] for v in result.values()),compared_files=len(result),all_numeric_csv_equivalent=all(v.get('numerically_equivalent',True) for v in result.values()),numeric_tolerance=dict(relative=1e-10,absolute=1e-12,p_value_absolute=1e-300),files=result)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--actual',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();report=compare(a.reference,a.actual);a.out.write_text(json.dumps(report,indent=2)+'\n');assert report['all_numeric_csv_equivalent'];print('Numerical equivalence PASS; byte-identical',report['byte_identical_files'],'/',report['compared_files'])
