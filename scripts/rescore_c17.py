"""Exploratory rescoring of saved FULL-model routes; never a new search."""
from pathlib import Path
import argparse,json,csv
from itertools import product


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--records',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();rows=[]
    for path in sorted(a.records.glob('*.json')):
        record=json.loads(path.read_text());r=record.get('result',record)
        if r.get('cohort')!='C17_INSTRUMENTED':continue
        if not r['independent_validation']['valid']:continue
        refs=r['references'];final=r['final'];ot=r['independent_validation']['recomputed']['overtime_ratio_sum']
        for weights,penalty in product([(1,1,1),(3,1,1),(1,3,1),(1,1,3)],[0,.25,1]):
            value=sum(w*final[k]/ref for w,k,ref in zip(weights,['cost','energy','co2'],refs))/sum(weights)+penalty*ot
            rows.append({'run_id':r['run_id'],'source':str(path),'label':'EXPLORATORY_RESCORING_ONLY','development_only':record.get('label')=='DEVELOPMENT_CONTRACT_CHECK','weights':str(weights),'penalty':penalty,'rescored':value})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with a.output.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print('rescored rows',len(rows),'new searches=0')
if __name__=='__main__':main()
