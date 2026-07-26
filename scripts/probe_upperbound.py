"""One-seed upperbound probe (baseline only)."""
from calibral.config import load_config
from calibral.experiment import run_comparison
from calibral.reporting import build_report

config = load_config("configs/thesis_upperbound.yaml")
print(
    "probe",
    config["model"]["name"],
    "max_length",
    config["model"]["max_length"],
    "batch",
    config["train"]["batch_size"],
)
comparison = run_comparison(config, methods=("baseline",), seeds=[42])
run = comparison["runs"][0]
print(
    "DONE",
    "macro_f1=",
    run.get("macro_f1"),
    "accuracy=",
    run.get("accuracy"),
    "status=",
    run.get("status"),
)
print(build_report(config).get("experiment_summary_path"))
