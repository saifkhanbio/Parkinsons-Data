"""Process completed model artifacts while the DESeq2 runner continues."""
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
MODELS = ["primary", "medication_timing", "phase", "usable_bases", "all_579"]
deadline = time.monotonic() + 24 * 3600
for model in MODELS:
    while not (OUT / model / "summary.json").exists():
        log = (OUT / "run.log").read_text()
        if "Execution halted" in log or time.monotonic() > deadline:
            raise RuntimeError("DESeq2 failed or exceeded the monitoring deadline; inspect run.log")
        time.sleep(5)
    for script in ["diagnose_influence.R", "run_enrichment.R"]:
        print(f"Running {script}: {model}", flush=True)
        with (OUT / "postprocess.log").open("a") as log:
            subprocess.run(["Rscript", str(OUT / script), model], stdout=log, stderr=subprocess.STDOUT, check=True)
while not (OUT / "model_summaries.json").exists():
    if time.monotonic() > deadline:
        raise RuntimeError("Missing final DESeq2 summary")
    time.sleep(2)
subprocess.run([sys.executable, str(OUT / "summarize_results.py")], check=True)
print("All model diagnostics, enrichment, validation, and reports complete.", flush=True)
