from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_c17_ablation import exact_signed_rank


def test_exact_small_sample_signed_rank_and_null():
    assert exact_signed_rank(np.zeros(9))==1
    assert exact_signed_rank(np.arange(1,10))==2/512
    assert exact_signed_rank(-np.arange(1,10))==2/512
    assert exact_signed_rank([1,-1])==1
