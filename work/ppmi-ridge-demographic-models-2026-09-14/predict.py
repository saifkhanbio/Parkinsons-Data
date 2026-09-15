"""Apply a trusted local Step 2 model to aligned metadata and optional counts."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from demographic_models import predict_artifact

p=argparse.ArgumentParser()
p.add_argument("--model",required=True)
p.add_argument("--metadata",required=True,help="TSV: PATNO, age_collection_years, sex")
p.add_argument("--counts",help="TSV: Geneid rows, participant columns; full raw annotation")
p.add_argument("--output",required=True)
a=p.parse_args()
out=Path(a.output)
if out.exists():raise SystemExit("Output exists; refusing overwrite")
state=joblib.load(a.model)
meta=pd.read_csv(a.metadata,sep="\t",dtype={"PATNO":str})
assert meta.PATNO.is_unique
raw=None
if state["family"]!="demographic":
    if not a.counts:raise SystemExit("RNA predictors require --counts")
    counts=pd.read_csv(a.counts,sep="\t").set_index("Geneid")
    assert counts.index.is_unique and counts.columns.is_unique
    assert set(counts.index)==set(state["feature_universe"])
    assert set(counts.columns)==set(meta.PATNO)
    raw=counts.loc[state["feature_universe"],meta.PATNO].to_numpy().T
prob=predict_artifact(state,raw,meta)
pd.DataFrame({"PATNO":meta.PATNO,"probability_PD":prob,"threshold":state["threshold"],
              "predicted_PD":prob>=state["threshold"]}).to_csv(out,sep="\t",index=False)
