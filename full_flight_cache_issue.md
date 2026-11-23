# `upload_full_flight` drops model predictions when its cache is stale

Found 2026-11-23 while promoting H-CAST 52121226. Edge case: only `Pipeline.upload_full_flight`
(the full-flight Label Studio upload / flythrough video path) is affected. The main pipeline run
re-runs classification and H-CAST every time and is unaffected.

## Symptom

A flight whose `<image_dir>/.full_flight_predictions.csv` was written under a different flat
classifier checkpoint or temperature logs

```
Ignoring .../.full_flight_predictions.csv: written under a different classifier or temperature (...); re-predicting.
```

but no prediction runs. The upload then carries human annotations only, with no model boxes,
and no error is raised.

## Cause

`src/pipeline.py`, introduced in beb7001 when the cache stamp was added:

```python
if os.path.isfile(full_flight_cache) and cached_stamp == current_stamp:
    pool_predictions = pd.read_csv(full_flight_cache)   # valid cache
elif os.path.isfile(full_flight_cache):
    print("Ignoring ...; re-predicting.")                # <- only prints
else:
    ...load models, generate_pool_predictions(...)       # only reached with NO cache file
```

`pool_predictions` is initialised to `None` and stays `None` on the stale-cache branch.

A second gap: the stamp (`<image_dir>/.full_flight_predictions.csv.classifier`) records only
`classification_model.checkpoint|temperature`. Promoting a new H-CAST checkpoint does not
invalidate it, so a valid-looking cache is reused with the old model's `hcast_*` columns.

## Who was affected

Any full-flight upload made after a flat classifier or temperature promotion (most recently
0139af8c / T 4.4675 on 2026-11-20) on a flight that already had a cache from an earlier model.
The ledger note at that promotion ("every flight re-predicts on its next run") was wrong for
this path. Check any full-flight Label Studio project or flythrough video built since a
classifier change for missing model boxes.

## Fix (branch `full-flight-cache-repredict`)

- The stale-cache branch now falls through to prediction instead of being a dead end.
- The stamp now also includes `hierarchical.checkpoint`.

Consequence: every existing `.full_flight_predictions.csv` re-predicts once on its next
full-flight upload (one full GPU pass per flight).
