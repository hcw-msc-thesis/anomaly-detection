"""
    python export_grafana.py --metrics metrics/combined/metrics.json
    cd grafana && docker compose up      
    
    The site should be found at http://localhost:3000 (ensure you have docker running!)

Tables:
  metrics  category, au_pro, au_roc
  roc      category, point, fpr, tpr  
  info     key, value                  
"""

import argparse
import datetime
import json
import os
import sqlite3

CATEGORIES = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam",
                "peach", "potato", "rope", "tire"]
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grafana", "data", "results.db")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("--metrics", required=True, help="the official evaluation's metrics.json")
    p.add_argument("--model", default="combined", help="what the maps were, shown on the dashboard")
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.metrics) as f:
        metrics = json.load(f)
    categories = [category for category in CATEGORIES if category in metrics]
    if not categories:
        raise SystemExit(f"No categories in {args.metrics}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    db = sqlite3.connect(args.out)
    db.executescript("""
        CREATE TABLE metrics (category TEXT PRIMARY KEY, au_pro REAL, au_roc REAL);
        CREATE TABLE roc (category TEXT, point INTEGER, fpr REAL, tpr REAL);
        CREATE TABLE info (key TEXT PRIMARY KEY, value TEXT);
    """)
    for category in categories:
        result = metrics[category]
        db.execute("INSERT INTO metrics VALUES (?, ?, ?)", (category, result["au_pro"], result["au_roc"]))
        db.executemany("INSERT INTO roc VALUES (?, ?, ?, ?)",
                       [(category, i, fpr, tpr) for i, (fpr, tpr)
                        in enumerate(zip(result["roc_curve_fpr"], result["roc_curve_tpr"]))])
    db.executemany("INSERT INTO info VALUES (?, ?)", [
        ("model", args.model),
        ("source", os.path.relpath(args.metrics).replace(os.sep, "/"))
        ("exported", datetime.datetime.now().isoformat(timespec="seconds")),
    ])
    db.commit()
    db.close()
    print(f"Wrote {len(categories)} categories ({', '.join(categories)}) to {args.out}")


if __name__ == "__main__":
    main()
