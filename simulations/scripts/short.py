#!/usr/bin/env python3
import os
import pandas as pd

script_dir = os.path.dirname(__file__)
analysis_csv = os.path.abspath(
    os.path.join(script_dir, "..", "data", "analysis.csv")
)
df = pd.read_csv(analysis_csv)
cols = [0, 1, 2, 74, 75, 146, 217, 618, 619]
df_selected = df.iloc[:, cols]
short_csv = os.path.abspath(
    os.path.join(script_dir, "..", "data", "short.csv")
)
df_selected.to_csv(short_csv, index=False)
