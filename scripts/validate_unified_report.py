import os
import json
from bs4 import BeautifulSoup
from pathlib import Path

REPORT_PATH = Path("reports/unified_report.html")
PROVENANCE_DIR = Path("artifacts/provenance")

# Helper to extract text from a table cell
def get_cell_text(row, col_idx):
    cells = row.find_all("td")
    return cells[col_idx].get_text(strip=True) if len(cells) > col_idx else None

def parse_report():
    with open(REPORT_PATH) as f:
        soup = BeautifulSoup(f, "html.parser")
    runs = {}
    for run_section in soup.find_all("div", class_="run-section"):
        run_id = run_section.find("h2").get_text().replace("Run: ", "").strip()
        run = {"id": run_id}
        # Data Information
        data_table = run_section.find("h3", string="Data Information").find_next("table")
        data_rows = data_table.find_all("tr")
        run["data"] = {
            "dataset": get_cell_text(data_rows[0], 1),
            "train_samples": int(get_cell_text(data_rows[1], 1)),
            "test_samples": int(get_cell_text(data_rows[2], 1)),
            "train_mean": float(get_cell_text(data_rows[3], 1)),
            "train_std": float(get_cell_text(data_rows[4], 1)),
            "test_mean": float(get_cell_text(data_rows[5], 1)),
            "test_std": float(get_cell_text(data_rows[6], 1)),
        }
        # Training Results
        train_table = run_section.find("h3", string="Training Results").find_next("table")
        train_rows = train_table.find_all("tr")
        run["training"] = {
            "final_train_acc": float(get_cell_text(train_rows[0], 1).replace("%", "")) / 100,
            "final_val_acc": float(get_cell_text(train_rows[1], 1).replace("%", "")) / 100,
            "training_time": float(get_cell_text(train_rows[2], 1).replace(" seconds", "")),
            "epochs": int(get_cell_text(train_rows[3], 1)),
        }
        # Privacy Guarantees
        priv_table = run_section.find("h3", string="Privacy Guarantees").find_next("table")
        priv_rows = priv_table.find_all("tr")
        run["privacy"] = {
            "target_epsilon": float(get_cell_text(priv_rows[0], 1)),
            "final_epsilon": float(get_cell_text(priv_rows[1], 1)),
            "delta": float(get_cell_text(priv_rows[2], 1)),
        }
        # Exclude Merkle tree and hash structure data
        merkle_section = run_section.find("h3", string="Merkle Tree")
        if merkle_section:
            merkle_section.decompose()
        hash_section = run_section.find("h3", string="Hash Structure")
        if hash_section:
            hash_section.decompose()
        runs[run_id] = run
    return runs

def parse_json(run_id):
    provenance_path = PROVENANCE_DIR / run_id / "provenance_report.json"
    verification_path = PROVENANCE_DIR / run_id / "verification.json"
    with open(provenance_path) as f:
        provenance = json.load(f)
    with open(verification_path) as f:
        verification = json.load(f)
    # Extract comparable fields
    last_epoch = provenance.get("training_history", [{}])[-1] if provenance.get("training_history") else {}
    privacy = provenance.get("privacy_metrics", {})
    return {
        "data": {
            "dataset": provenance.get("dataset", "N/A"),
            "train_samples": provenance.get("train_samples", 0),
            "test_samples": provenance.get("test_samples", 0),
            "train_mean": provenance.get("train_mean", 0.0),
            "train_std": provenance.get("train_std", 0.0),
            "test_mean": provenance.get("test_mean", 0.0),
            "test_std": provenance.get("test_std", 0.0),
        },
        "training": {
            "final_train_acc": last_epoch.get("train_acc", 0.0),
            "final_val_acc": last_epoch.get("val_acc", 0.0),
            "training_time": provenance.get("training_time", 0.0),
            "epochs": len(provenance.get("training_history", [])),
        },
        "privacy": {
            "target_epsilon": privacy.get("target_epsilon", 0.0),
            "final_epsilon": privacy.get("final_epsilon", 0.0),
            "delta": privacy.get("delta", 0.0),
        }
    }

def compare_and_report():
    report_runs = parse_report()
    mismatches = []
    for run_id, report_data in report_runs.items():
        json_data = parse_json(run_id)
        for section in ["data", "training", "privacy"]:
            for key, val in report_data[section].items():
                json_val = json_data[section][key]
                if abs(val - json_val) > 1e-4 if isinstance(val, float) else val != json_val:
                    mismatches.append(f"Run {run_id}: {section}.{key} - Report: {val}, JSON: {json_val}")
    if mismatches:
        print("Mismatches found:")
        for m in mismatches:
            print(m)
    else:
        print("All values in the unified report match the JSON files.")
    print(f"Checked {len(report_runs)} runs.")

if __name__ == "__main__":
    compare_and_report() 