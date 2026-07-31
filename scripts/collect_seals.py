"""Collect seal-species validation accuracy through time from Comet."""
import comet_ml
import json
import os
import io
import datetime
import pandas as pd

API_KEY = "ypQZhYfs3nSyKzOfz13iuJpj2"
SCRATCH = "/tmp/claude-4736/-blue-ewhite-b-weinstein-src-BOEM/1656d312-e1dd-44bb-9dc2-a708a1542ce5/scratchpad"
SPECIES = ["Halichoerus grypus", "Phoca vitulina"]

api = comet_ml.API(api_key=API_KEY)
keys = json.load(open(os.path.join(SCRATCH, "seal_exp_keys.json")))

rows = []
for i, k in enumerate(keys):
    e = api.get_experiment("bw4sz", "BOEM", k)
    try:
        tags = e.get_tags()
    except Exception:
        tags = []
    ts = e.start_server_timestamp
    date = datetime.datetime.utcfromtimestamp(ts / 1000).date().isoformat()

    # per-epoch series
    series = {}
    for name in ["val_loss"] + ["Class Accuracy_" + s for s in SPECIES]:
        try:
            d = e.get_metrics(name)
        except Exception:
            d = []
        series[name] = {int(x["epoch"]): float(x["metricValue"])
                        for x in d if x.get("epoch") is not None}

    vl = series["val_loss"]
    best_epoch = min(vl, key=vl.get) if vl else None
    last_epoch = max(vl) if vl else None

    # validation sample counts per species from the logged val annotations table
    counts = {}
    n_val = None
    for fname in ["val_annotations.csv", "validation_annotations.csv"]:
        try:
            assets = [a for a in e.get_asset_list() if a["fileName"] == fname]
        except Exception:
            assets = []
        if not assets:
            continue
        try:
            raw = e.get_asset(assets[0]["assetId"], return_type="binary")
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as ex:
            print(f"  asset read fail {k}: {ex}")
            continue
        col = "label" if "label" in df.columns else (
            "cropmodel_label" if "cropmodel_label" in df.columns else None)
        if col is None:
            continue
        n_val = len(df)
        vc = df[col].value_counts()
        for s in SPECIES:
            counts[s] = int(vc.get(s, 0))
        break

    for s in SPECIES:
        acc_series = series["Class Accuracy_" + s]
        if not acc_series:
            continue
        rows.append({
            "experiment": k,
            "date": date,
            "timestamp_ms": ts,
            "tags": ";".join(tags),
            "species": s,
            "acc_best_val_loss_epoch": acc_series.get(best_epoch),
            "acc_final_epoch": acc_series.get(last_epoch),
            "acc_max": max(acc_series.values()),
            "n_epochs": len(acc_series),
            "best_epoch": best_epoch,
            "val_loss_best": vl.get(best_epoch) if vl else None,
            "n_val_species": counts.get(s),
            "n_val_total": n_val,
        })
    print(f"[{i+1}/{len(keys)}] {k} {date} epochs={len(vl)} counts={counts}")

out = pd.DataFrame(rows).sort_values(["timestamp_ms", "species"])
out.to_csv(os.path.join(SCRATCH, "seal_accuracy_history.csv"), index=False)
print(out.to_string())
