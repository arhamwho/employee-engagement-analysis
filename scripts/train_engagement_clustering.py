"""Fit the leakage-free engagement clustering model and write reports/plots."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clustering import fit_and_save  # noqa: E402
from src.data_loader import load_raw_dataset  # noqa: E402


def main():
    raw = load_raw_dataset()
    bundle = fit_and_save(raw, save_artifacts=True)
    m = bundle["metadata"]
    print("K", m["k"])
    print("silhouette", round(m["silhouette"], 4))
    print("davies_bouldin", round(m["davies_bouldin"], 4))
    print("calinski_harabasz", round(m["calinski_harabasz"], 1))
    print("reason:", m["k_selection_reason"])
    print("names:", m["names"])
    for p in bundle["profiles"]:
        print(
            f"  cluster {p['cluster']} n={p['n_employees']} ({p['percent']}%) "
            f"sat={p.get('mean_satisfaction_items'):.3f} ot={p.get('overtime_percent'):.1f}% "
            f"pay={p.get('mean_MonthlyIncome'):.0f} tenure={p.get('mean_YearsAtCompany'):.2f}"
        )


if __name__ == "__main__":
    main()
