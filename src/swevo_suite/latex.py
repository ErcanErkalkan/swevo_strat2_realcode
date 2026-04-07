from __future__ import annotations
from pathlib import Path
import pandas as pd

def dataframe_to_latex_table(df: pd.DataFrame, caption: str, label: str, floatfmt: str = ".3f") -> str:
    latex = df.to_latex(index=False, escape=False, float_format=lambda x: format(x, floatfmt))
    return "\n".join([
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        latex,
        "\\end{table}",
    ])
