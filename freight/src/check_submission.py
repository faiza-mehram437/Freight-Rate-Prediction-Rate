"""Check validation_predictions.csv against the template before submitting."""
import sys

import pandas as pd

from pipeline import DATA, ROOT

pred = pd.read_csv(ROOT / "validation_predictions.csv")
template = pd.read_csv(DATA / "validation_predictions_template.csv")

checks = {
    "columns are exactly load_id,predicted_rate": list(pred.columns) == ["load_id", "predicted_rate"],
    f"row count is {len(template)}": len(pred) == len(template),
    "load_id values and order match the template": pred["load_id"].tolist() == template["load_id"].tolist(),
    "no missing predictions": pred["predicted_rate"].notna().all(),
    "all predictions are positive": (pred["predicted_rate"] > 0).all(),
}
for name, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + name)
print(pred["predicted_rate"].describe().round(1).to_string())
sys.exit(0 if all(checks.values()) else 1)
