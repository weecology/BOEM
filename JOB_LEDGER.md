# BOEM Job Ledger

Append-only history of SLURM jobs submitted from this repo. Scoped to this
project only (the shared `~/logs/job_ledger.md` mixes in MillionTrees and
other repos, so it's not a reliable place to reconstruct BOEM-only history).

Format:

```
## <JOBID> — <YYYY-MM-DD HH:MM> — <script> — <STATE>
Why: <goal behind the run>
Result: <what happened — completed cleanly / failed with X / timed out at Y>
Next: <what to do next time / what unblocks this>
```

Log files live in `~/logs/<name><jobid>.out|.err` (check accounting with
`sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Start,End,Elapsed`).

---

## Backfilled: BRI (BRI_GOM_23_24_geotiff/2023-08-18) prediction pipeline

Status as of 2026-07-13: **blocked, not run since 2026-05-22.** Three
consecutive attempts to run the full active-learning/prediction pipeline
(`submit.sh` -> `main.py` -> `src/pipeline.py Pipeline.run()`) over the
64,805-image BRI pool each failed at a different stage. The last failure's
likely fix (retrained classification checkpoint) is now staged in
`boem_conf/classification_model/finetune.yaml` and `boem_conf/boem_config.yaml`
(both currently uncommitted on `check_augs`), but the fix has **not yet been
verified by rerunning against BRI**.

### 32303031 — 2026-05-13 16:15 — submit.sh — FAILED
Why: First attempt at inference-only active-learning pass over BRI
(no existing annotations for this flight yet).
Result: `IndexError: list index out of range` early in the pipeline.
Next: Debug and fix the indexing bug (see full traceback in
`~/logs/BOEM32303031.err`).

## 32722727 — 2026-05-19 14:58 — submit.sh (job-name BOEM_BRI) — FAILED (55s)
Why: Retry after the 05-13 IndexError fix; confirmed pool built (64,805
images after excluding existing) before failing.
Result: `ValueError: classification_model.use_metadata=True but no image
metadata matched flight 2023-08-18` (src/pipeline.py:98,
`_metadata_lookup_for_pool`) — BRI has no matching rows in
`report.metadata_dir` (`/blue/ewhite/b.weinstein/BOEM/metadata_aflight_csvs`).
Next: Either add BRI flight metadata, or disable metadata use for this
flight. -> led to adding `classification_model.use_metadata=False` override
in submit.sh.

## 33084407 — 2026-05-22 23:52 — submit.sh (+ use_metadata=False) — FAILED (~30s)
Why: Retry with `classification_model.use_metadata=False` added to work
around the missing-metadata error above.
Result: `RuntimeError: Error(s) in loading state_dict for CropModel` —
architecture mismatch loading
`.../UBFAI Images with Detection Data/classification/checkpoints/buffer_30/5a5d6698f2e74f1b8c21cdb0e187b080.ckpt`.
Checkpoint keys are `backbone.*` / `metadata_encoder.*` / `classifier.*`
(older CropModel class shape), but the installed deepforest's CropModel
expects `model.*` / `fc.*` — this is a stale checkpoint vs. currently
installed deepforest version, unrelated to the use_metadata flag.
Next: Retrain/point to a classification checkpoint saved with the current
CropModel class.

### Since 2026-05-22 (no BRI job submitted, but relevant changes landed)
- 2026-07-07/08, job 36539340 (`classification_BOEM`) — COMPLETED cleanly.
  Produced `training/classification/checkpoints/buffer_30/d8995ca8690046ce9dd775fb42f55cb1.ckpt`,
  now pointed to by `boem_conf/classification_model/finetune.yaml` (uncommitted
  change on `check_augs`). This checkpoint should match the current CropModel
  class and is the likely fix for 33084407's RuntimeError, but this has not
  been tested against BRI yet.
- 2026-07-07/08, job 36523583 (`BOEM_det_balanced`) — TIMEOUT at 24h wall,
  epoch 16/20. Per-epoch checkpoint `training/checkpoints/a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt`
  is usable and now pointed to by `boem_conf/boem_config.yaml` (uncommitted).

**Next step to actually resolve BRI status**: resubmit `submit.sh` against
`BRI_GOM_23_24_geotiff/2023-08-18` now that both checkpoint configs point at
current-architecture checkpoints, and watch for a 4th failure mode before
assuming it's fixed.

## 37110906 — 2026-07-14 12:04 — submit.sh — FAILED (6m11s)
Why: Resubmit of the BRI pipeline now that `finetune.yaml`'s classification
checkpoint (job 36539340) and `boem_config.yaml`'s detection checkpoint (job
36523583) point at current-architecture checkpoints — attempting to clear
the 33084407 CropModel state_dict mismatch.
Result: Never got that far. `IndexError: list index out of range` in
`src/label_studio.py:432 download_completed_tasks()` —
`labeled_task['annotations'][0]["result"]` assumes every task returned by the
Label Studio "review" queue has >=1 annotation; some task in the
Bureau-of-Ocean-Energy-Management-Review project has zero. Crashes during
`check_annotations()`, i.e. before the pipeline ever reaches prediction or
loads either checkpoint — so this is the SAME bug family as 32303031
(2026-05-13), just never actually fixed; it simply didn't reproduce on
05-19/05-22 because those runs hit different, earlier errors first.
Next: Add a guard in `download_completed_tasks` (src/label_studio.py:432) to
skip/log tasks with an empty `annotations` list instead of indexing [0]
blindly. Once fixed, resubmit and watch for whether it then reaches the
checkpoint-loading stage cleanly (the seals array job, 37136785/37136972,
already confirms both updated checkpoints co-load fine via smoke_ckpt_37019549,
so that part should be clear).

## 37147247, 37147051, 37147052, 37147053 — 2026-07-14 — submit_upload_full_flight.sh — SUBMITTED
Why: Finish full-flight prediction + human review for the 2026 Gulf of Mexico
set. Of the five Gulf flights (Feb 1-2 2026, 329,097 images), only
JPG_20260202_141900 had ever had a full-flight pass (job 29183554, 2026-04-08);
its 379 uploaded images are all reviewed and returned to
`annotations/train/JPG_20260202_141900/`. The other four had no full-flight
predictions at all — only a random 500-image active-learning sample in
`.prediction_cache/pool_predictions.csv`. Jobs (one per flight, all with
`--skip-annotated` so already-reviewed images are not re-uploaded):
  37147247 JPG_20260201_134000 (96,063 imgs, ~7h expected)
  37147051 JPG_20260201_093500 (79,379 imgs, ~6h)
  37147052 JPG_20260202_094800 (60,693 imgs, ~4.6h)
  37147053 JPG_20260202_122400 (22,749 imgs, ~1.7h)
Runtime estimates scaled from 29183554 (70,176 imgs in 5h17m on hpg-b200).
Two code fixes were required first, both uncommitted on `check_augs`:
  1. `src/pipeline.py upload_full_flight` silently fell back to the
     active-learning pool cache when `.full_flight_predictions.csv` was absent
     — which is the exact state all four flights were in. It would have
     uploaded ~128 images from a stale 500-image random sample and printed
     "Full flight upload complete", looking done at <1% coverage. Fallback
     removed; only the full-flight cache is accepted.
  2. `src/label_studio.py:432 download_completed_tasks` — added the guard that
     37110906's entry (above) prescribed, since `upload_full_flight` calls
     `check_annotations()` on startup and would have hit the same IndexError.
     Tasks with an empty `annotations` list are now skipped, and `images` is
     no longer appended before the guard (which would have desynced the lists).
Note: 37147050 was the original submission for JPG_20260201_134000; it was
cancelled 34s in because it started before fix (2) landed and may have imported
the old module. 37147247 is its resubmission.
Result: pending as of writing.
Next: On completion, confirm each flight wrote `.full_flight_predictions.csv`
and that a "BOEM - Full Flight - <flight>" project appeared in Label Studio
with a plausible task count (141900 yielded 342 detection images from 70k, a
~0.5% hit rate — anything near the old 500-sample numbers means the pool cache
leaked back in). Then the 2026 Gulf set is fully queued for human review; the
detector used is a09c6933/epoch16, vs a1c5649 for 141900 — per Ben, both are
fine and cross-flight detector consistency is not a concern here.

## 37325655 — 2026-07-16 14:10 — submit_verify_metadata.sh — COMPLETED (13m36s)
Why: neaq and BRI have no rows in `report.metadata_dir`, so both hit the same
`use_metadata=True but no image metadata matched flight` ValueError that killed
BRI job 32722727 and seals task 37254762_12. `submit_seals_array.sh` warns that
the classifier "was trained with metadata (metadata_dim=32) and needs it at
inference", which — if true — would mean neither dataset can be classified at
all. Tested that claim against the current checkpoint (d8995ca8) directly.
Result: **The warning is wrong for this checkpoint; use_metadata=False is safe.**
  - `CropModel.load_from_checkpoint` restores `use_metadata: True` from the ckpt's
    own hparams, so the BOEM yaml flag does NOT change the architecture. Loaded as
    backbone + metadata_encoder + `classifier: Linear(2080 -> 69)` (2048 image +
    32 metadata), `model is None`. No state_dict error.
  - `forward(x, metadata=None)` returns valid logits — deepforest model.py:522-526
    zero-fills the metadata half rather than raising.
  - So `classification_model.use_metadata=False` only skips the *lookup*
    (pipeline.py:84-85), which is exactly what the missing-metadata datasets need.
  - Job 33084407's `state_dict` RuntimeError — the origin of the warning — was the
    stale `5a5d6698` checkpoint, not the flag, as that entry already suspected.
  - Bonus: the no-metadata path is also FASTER. detection.py:324-355 forces
    `workers=0` and runs image-by-image when a metadata_lookup is present ("leaves
    the GPU idle"); metadata_lookup=None takes the batched path at line 356.
Caveat (RESOLVED by 37327824 — see below): real vs zeroed metadata gave
*byte-identical* predictions here (mean AND max prob delta 0.0000), which looked
like the metadata branch might be dead. That was an artifact of this test, not a
finding — see the correction below. Do not cite the 0.0000 number.
Next: use `classification_model.use_metadata=False` for neaq/BRI.

## 37327824 — 2026-07-16 14:35 — submit_probe_encoder.sh — COMPLETED (16m)
Why: 37325655 measured zero drift between real and zeroed metadata, which would
mean the SpatialTemporalEncoder contributes nothing and the seals runs pay the
slow per-image path for a dead feature. Probed the encoder directly.
Result: **The encoder is alive; 37325655's zero-drift number was a bad test.**
  - Encoder output for neaq coords (41.4, -70.9, doy 82): 9/32 dims nonzero,
    out sum 4.11. Gulf (28.5, -89.0, doy 230): 10/32, sum 3.60. Non-trivial.
  - Linear(6,32) weights are trained, not collapsed: mean |w| 0.198, max 0.442.
  - Why 37325655 saw 0.0000: it fed `torch.randn` noise images. The model
    saturates on noise, so the small metadata logit shift disappears under
    softmax at 4dp. Any future drift test must use real crops.
  - Metadata is nonetheless a *minor* contributor: classifier meta-half mean |w|
    0.0146 vs image-half 0.1305 (ratio 0.112), and only ~9/32 dims fire at neaq.
  - Aside: (lat=0, lon=0, doy=1) gives 0/32 nonzero — "null island" is
    numerically identical to the zero-fill path. Never impute missing coords as 0.
Implication: zero-filling neaq/BRI loses a real but modest signal. That is
acceptable because those datasets have no metadata at all — the alternative is
not running. It does NOT follow that seals should drop metadata: seals has real
metadata, it contributes, so keep use_metadata=True there.
Next: quantify the actual accuracy cost on real crops before trusting neaq
species-diversity numbers as comparable to metadata-having flights.

## Correction: BRI imagery is raw undemosaiced Bayer, not "corrupted"
Investigated 2026-07-16 while scoping a BRI run. The 64,805 tifs in
`BRI/BRI_GOM_23_24_geotiff/2023-08-18` (479 GB, one date, ~37 camera-session
prefixes of ~1,300-1,800 frames) are **structurally fine but were never
demosaiced** — the raw Bayer mosaic is still present in every band.
Evidence (3184x2160, 3-band uint8, EPSG:32619):
  - 2x2 quadrant means are a textbook RGGB signature: (0,0)=42.7 R, (0,1)=128.7 G,
    (1,0)=129.2 G, (1,1)=180.7 B — the two greens match to within 0.5.
  - Even rows mean 85.7 vs odd rows 155.0; band-1 row-mean profile alternates
    81.1/148.4/82.0/149.3/... with period 2. This ~70-level line alternation is
    the "large striations" reported by eye.
  - No directional banding (row/col jitter ratio 0.93-1.01 over 8 random frames)
    and zero dropped scan lines, so this is not sensor/transfer damage.
Implication: the detector has only ever seen mosaic texture on BRI, so **every
BRI result to date is suspect**, and running the pipeline on these tifs as-is
would produce garbage regardless of any Label Studio upload cap. This supersedes
the "resubmit submit.sh and watch for a 4th failure mode" advice above — the
blocker is the imagery, not the pipeline.
Next: demosaic (e.g. `cv2.cvtColor(..., COLOR_BayerRG2RGB)`, pattern to be
confirmed) into a derived image dir, eyeball a contact sheet, and only then run
the pipeline. Do not chase the pipeline failures until the imagery is fixed.

## 37329130 — 2026-07-16 14:26 — submit_neaq.sh --array=19 — FAILED, OUT_OF_MEMORY (17m26s)
Why: first end-to-end pilot of the pipeline on neaq (`neaq_20220819_whale`, 192
images) — the dataset had never been run at all, and three blockers had just been
patched (flight_name collision, metadata gate, non-recursive image collection).
Result: **host-RAM OOM. MaxRSS 83.7GB against `--mem=90GB`**; DataLoader worker
killed, `Detected 1 oom_kill event`. Not a code bug — a resource request inherited
from a differently-shaped dataset.
Chain: `use_metadata=False` -> no metadata_lookup -> detection.py:356 takes
predict_tile's *batched* path instead of the per-image one (detection.py:324-355,
which forces workers=0). deepforest documents `dataloader_strategy="batch"` as
loading whole images with CPU parallelism, so each of the 5 workers holds a full
frame + its patch views. neaq frames are 9504x6336 whale / 8688x5792 belly (~60MP
/ ~50MP) vs UBFAI's ~6464x4852 (~31MP) — roughly 2x — so the 90GB that fits the
seals array does not fit neaq. Seals never hit this: it *has* metadata, so it
takes the slow per-image path, on half-size images.
Next: 37330544 retries the same pilot at `--mem=350GB` (b200 nodes have 2TB,
FreeMem 1.9TB — headroom is cheap). Only memory changed; workers=5 and
batch_size=64 left alone so the retry isolates one variable. If it OOMs again at
350GB, the cost is accumulating across the 192-image list rather than per-worker,
and the fix is chunking `image_paths` in `predict_boxes`, not more RAM.

## 37329936 — 2026-07-16 14:41 — submit_neaq.sh --array=0-18,20-39%4 — CANCELLED (5 tasks)
Why: released the remaining 39 neaq tasks on user instruction while pilot 37329130
was still loading models and had produced no prediction output.
Result: cancelled ~3m in once the pilot OOM'd, since all 40 would have hit the same
wall. No task reached an upload, so Label Studio was untouched. The `%4` concurrency
cap is what held the blast radius to 5 tasks.
Next: do not release the array again until a pilot has completed an upload
end-to-end. "Still loading models after 12 minutes" is not evidence of health —
checkpoint load alone takes ~10-13m on this cluster (see 37325655, 37327824).

## 37330544 — 2026-07-16 14:52 — submit_neaq.sh --array=19 @ --mem=350GB — FAILED, CUDA OOM
Why: retry of the 37329130 host-RAM OOM with 350GB instead of 90GB, memory the only
variable changed.
Result: cleared the host-RAM wall, then died on the GPU:
`torch.OutOfMemoryError: Tried to allocate 170.90 GiB. GPU 0 has a total capacity of
178.35 GiB` — inside `predict_step` -> RetinaNet `backbone` -> the first `conv2d`.
**`predict.batch_size` silently means different things on the two predict_tile paths**
(deepforest main.py:610-684):
  - `dataloader_strategy="single"` (taken when a metadata_lookup exists — seals):
    one image at a time, batch_size counts **patches**. boem_config's comment
    "64 fits all patches of an image in one pass" is written for this path.
  - `dataloader_strategy="batch"` (taken when metadata_lookup is None — neaq/BRI,
    detection.py:356): the `MultiImage` dataset, where each item is a whole image and
    `create_overlapping_views` returns `[N*num_windows, C, size, size]`. batch_size
    counts **images**. 64 images x ~70 patches (9504x6336 @ patch_size=1000) = 4,480
    patches in one forward pass -> 170.90 GiB.
So `classification_model.use_metadata=False` — nominally a flag about *metadata* —
silently changes the meaning of batch_size by a factor of ~70. Both neaq OOMs
(37329130 host, 37330544 GPU) are downstream of that one path switch.
Next: retry the pilot with `predict.batch_size=4` (~280 patches ~= 11GB, vs the
~85GB already allocated when the fatal conv ran); now set in submit_neaq.sh.
Keep --mem=350GB: the host-RAM pressure from workers=5 holding full 60MP frames is
real and independent of the GPU issue.
Landmine for the belly tasks: `MultiImage` docstring says "Images are expected to be
the same size" and it sizes the dataset from `paths[0]` only. neaq is uniform *within*
a camera dir (whale 9504x6336, belly 8688x5792) but a mixed dir would silently
mis-window. Verify per-dir uniformity before releasing the array.

## 38309076 — 2026-07-29 14:30 — submit_prepare_annotations.sh — SUBMITTED
Why: Rebuild detection/UBFAI crops + train.csv/test.csv after pulling 375 new
Label Studio tasks (2,504 annotation rows: 1,921 train / 175 validation / 408
review) via the new `scripts/download_annotations.py`. Tasks were NOT deleted
from the LS server, so a rerun would re-download them.
Note: this is the first prepare run since fixing `src/annotators.py:136`, which
had been writing validation and review downloads into `annotations/train/`.
Validation and review annotations now land in their own dirs; prepare_USGS
stage 0 reads all three subdirs per flight, so the merge is unaffected.
Watch: neaq flights ("Belly camera edited", "Whale camera edited", 242 of the
review rows) resolve imagery by bare flight_name under IMAGERY_BASE /
SCREENED_IMAGES_BASE — expect "imagery dir not found" skips for those.
Result: COMPLETED, 44m09s, MaxRSS 21.5GB of 64GB. Refreshed 50 flights (1 already
up to date); 118 flights skipped for missing imagery, including all three neaq dirs
("Belly camera edited", "Whale Camera Edited", "Whale camera edited") exactly as
predicted — those 242 review annotations did NOT reach training. Saved
train (582,531 rows), test (132,634 rows), zero_shot (22,996 rows) to
/blue/ewhite/b.weinstein/BOEM/training/crops. Hard-negative conversion:
27,010 nuisance rows -> 16,972 markers (train), 1,109 -> 797 (test). Post-relabel
dedup dropped 130,321 train / 5,848 test duplicate rows. Clean stderr.
Next: FIXED 2026-07-29 (uncommitted, scripts/prepare_USGS.py). Stage 0 assumed one
root_dir per flight, which neaq cannot satisfy: the LS flight_name is the *camera dir*
("Belly camera edited"), and that dir repeats under every date in
/blue/ewhite/b.weinstein/BOEM/neaq/<YYYY.MM.DD>/, so one flight_name spans 20 roots.
Added NEAQ_BASE + _neaq_image_index() (basename -> abs path, basenames carry the
capture date so they are globally unique: 74,404 files, 0 collisions), and rekeyed the
loop onto (containing dir, basename) so a flight can span many roots. Dry run: all
2,356 rows / 311 images across the 3 neaq flights resolve, 0 unresolved. Verified
end-to-end on "Whale Camera Edited" (24 patches, label Eubalaena glacialis) and
regression-checked on 2023_June20b_JPG (screened_images path unchanged).
The 114 other skipped flights are genuinely missing imagery — unrelated to neaq.
Rerun prepare_USGS to pick these up, but NOT while 38309080 is training (it would
rewrite the train.csv/test.csv underneath it).

## 38309080 — 2026-07-29 14:30 — submit_USGS_detection_augs.sh — SUBMITTED (afterok:38309076)
Why: New detection model on the refreshed annotation set. Defaults kept
(POSITIVE_BATCH_FRACTION=0.90, batch 64, lr 0.001, 10 epochs).
Result: RUNNING and healthy as of 2026-07-29 21:20 (5h50m elapsed). Comet
d1248ed881a847128752f86d1370add3. 4 epoch checkpoints written to
/blue/ewhite/b.weinstein/BOEM/training/checkpoints/d1248ed881a847128752f86d1370add3/
at ~72 min/epoch: val_cls 0.0207 -> 0.0170 -> 0.0164 -> 0.0163 (epoch00-03).
10 epochs projects to ~12h total, finishing ~03:30, well inside the 24h wall.
Note: stdout has not been written since 16:50 — the rich progress bar does not
flush to file, so log mtime is NOT a liveness signal here. Use the checkpoint dir
mtimes instead. Non-fatal: ~13 "Failed to plot ... Input must be valid geometry
objects" during "Logging training/validation dataset samples", so some Comet
sample visualizations are missing. sstat MaxRSS 244GB against --mem=200GB —
worth watching, though the job is alive and progressing.
Next: epoch03 already matches 36523583's best (epoch16 val_cls 0.0163) at
epoch 3 of this run — compare final metrics before swapping the checkpoint into
boem_conf/boem_config.yaml.

## 38309081 — 2026-07-29 14:30 — submit_USGS_classification.sh — SUBMITTED (afterok:38309076)
Why: New classification model on the refreshed crops, with
`classification_model.use_metadata=False` (added to the submit script) per user
request — no metadata this round.
Result: COMPLETED, 3h03m, all 45 epochs (Epoch 44/44). Checkpoint
/blue/ewhite/b.weinstein/BOEM/training/classification/checkpoints/buffer_30/dce4406478df4191b18e21362156eaf8.ckpt.
Comet logged use_metadata: False, confirming the override took. 69 classes kept /
6 dropped, 190,149 crops pooled, split by parent image (31,112 train / 1,410 test),
class balance downsampled 170,878 -> 51,416. Final: Micro-Average Accuracy 0.745,
Macro-Average Precision 0.632, val_loss 4.45. Weak classes at 0.0 accuracy include
Megaptera novaeangliae, Mergus merganser, Ardea herodias, Puffinus puffinus,
Sternula antillarum; Tursiops truncatus only 0.125.
**Architecture change — the "Next" note above was wrong.** Training with
use_metadata=False built a plain CropModel: this checkpoint's state_dict has only
`model.*` (model.fc.weight 69x2048), no metadata_encoder. The previous checkpoint
d8995ca8 has `backbone.*` / `metadata_encoder.*` / `classifier.*`. The two are NOT
interchangeable.
Next: pointing boem_conf/classification_model/finetune.yaml at dce44064 while it
still says `use_metadata: true` risks the same class of state_dict failure as
33084407. Set use_metadata: false there in the same commit as the checkpoint swap.

## (no job) — 2026-07-29 23:41 — config swap: metadata-free classifier goes live
Why: user asked for a classification model that does not depend on metadata. 38309081
had already produced one, so no new training run was needed — this is the wiring step
the previous entry's "Next" note called for.
Change (uncommitted, boem_conf/classification_model/finetune.yaml):
  checkpoint: d8995ca8690046ce9dd775fb42f55cb1 -> dce4406478df4191b18e21362156eaf8
  use_metadata: true -> false        (both in the same edit, per the 38309081 note)
Verified before swapping: `CropModel.load_from_checkpoint(dce44064)` succeeds; state_dict
top-level prefixes are `model.*` only with no metadata_encoder; model.fc.weight is
69x2048; the checkpoint's own config carries use_metadata False; label_dict has all 69
species. Note hyper_parameters['config'] and ['config_args'] are both None in this ckpt,
but load_from_checkpoint fills the config anyway — not a problem.
Verified after swapping: hydra compose of boem_config resolves
classification_model.checkpoint to the dce44064 path (file exists) and use_metadata to
python bool False, so Pipeline._use_classification_metadata() returns False and
_metadata_lookup_for_pool() short-circuits to None without needing report.metadata_dir.
Note: 38309081 trained on crops that EXCLUDE the 3 neaq flights, since prepare_USGS.py's
neaq fix is still uncommitted and unrun. Rerunning prepare_USGS (safe once detection
38309080 finishes) then retraining metadata-free would fold those in.
Next: commit the finetune.yaml swap; not yet committed. Weak classes carry over from
38309081 — Megaptera novaeangliae, Mergus merganser, Ardea herodias, Puffinus puffinus,
Sternula antillarum at 0.0 accuracy, Tursiops truncatus 0.125 — so any pipeline run on
this checkpoint will be unreliable for those.

## 38374816 — 2026-07-30 12:09 — submit_prepare_USGS.sh — SUBMITTED
Why: fold the 3 neaq flights into training. The neaq fix in scripts/prepare_USGS.py
(NEAQ_BASE + _neaq_image_index, rekeying the loop onto (containing dir, basename) so one
flight_name can span 20 date roots) is still uncommitted but present in the working tree,
and this is the first run that exercises it for real. Safe to run now: detection 38309080
COMPLETED 2026-07-30T04:05 (12h34m, MaxRSS 244GB), so nothing is reading
/blue/ewhite/b.weinstein/BOEM/training/crops/{train,test}.csv while they are rewritten.
Expect the 3 neaq camera dirs ("Belly camera edited", "Whale Camera Edited",
"Whale camera edited") to stop being skipped and contribute ~2,356 rows / 311 images.
The other 114 skipped flights are genuinely missing imagery and will still be skipped.
Result: **COMPLETED, 21m27s. The neaq fix works — all 3 flights landed.** Log lines
"using neaq fallback": Belly camera edited (72,942 images indexed across date dirs) ->
206 images refreshed; Whale camera edited (1,303 indexed) -> 101 images; Whale Camera
Edited (159 indexed) -> 4 images. Verified in
/blue/ewhite/b.weinstein/BOEM/training/crops/train.csv: 2,255 rows carry a neaq
flight_name (1,991 Whale camera edited / 258 Belly camera edited / 6 Whale Camera
Edited), plus 93 rows in test.csv. Their original_label distribution is Anatidae 795,
Delphinidae 369, Bird 291, Morus bassanus 232, Larus 155, ..., Eubalaena glacialis 41.
Totals: train 608,980 rows (was 582,531), test 134,154 (was 132,634), zero_shot 13,768.
Hard-negative conversion 26,224 -> 16,904 markers (train), 1,996 -> 862 (test); dedup
dropped 158,475 train / 6,219 test. The other 114 flights still skipped for missing
imagery, as expected.
Next: nothing for prepare. But note the detection model 38309080 finished BEFORE this
ran, so **the current detection checkpoint has never seen neaq** — a detection retrain
on these refreshed crops is the outstanding piece.

## 38309080 (result addendum) — detection trained on the PRE-neaq crops
COMPLETED 2026-07-30T04:05, 12h34m, all 10 epochs. Comet d1248ed881a847128752f86d1370add3.
Best val_classification 0.01436 (vs 0.0163 for the old a09c6933/epoch16), val_loss 0.0311,
epoch09 is the best checkpoint (epoch09-val_cls0.0144.ckpt). Zero-shot post-training:
box precision 0.863, box recall 0.033, empty-frame accuracy 0.998.
The low box_recall (0.056-0.062 val, 0.033 zero-shot) is NOT a regression — it is the
metric bug written up in deepforest_box_recall_issue.md: `RecallPrecision` increments
`num_images` on every frame but early-returns on empty frames before adding to the
numerator, so box_recall is ceilinged at num_nonempty/num_total. With a 90% empty-frame
diet the ceiling is ~0.10. Do not read it as detector quality; use
evaluate.evaluate_geometry, which strips empty rows.
Next: (1) boem_conf/boem_config.yaml still points at the OLD a09c6933/epoch16 checkpoint
— swap to d1248ed8/epoch09 or to a neaq-inclusive retrain; (2) retrain detection on the
38374816 crops so neaq is actually in the detector.

## 38374817 — 2026-07-30 12:09 — submit_USGS_classification.sh — SUBMITTED (afterok:38374816)
Why: retrain the metadata-free classifier on the neaq-inclusive crops. Supersedes
38309081/dce44064, which never saw neaq data. Submit script already carries
`classification_model.use_metadata=False`, so this stays a plain CropModel with no
metadata_encoder — same architecture as dce44064, more data.
Result: **COMPLETED, 3h04m, all 45 epochs.** Checkpoint
/blue/ewhite/b.weinstein/BOEM/training/classification/checkpoints/buffer_30/e79ca03e4e1b45f6820c009df6c190b5.ckpt.
Comet https://www.comet.com/bw4sz/boem/e79ca03e4e1b45f6820c009df6c190b5.
Pooled 72,596 per-image CSVs (was 72,285), 34,168 parent images -> 31,159 train /
1,391 test, **70 classes kept** (was 69) / 6 dropped, class balance downsampled
171,034 -> 48,997.
Headline vs 38309081/dce44064: Micro-Average Accuracy 0.745 -> **0.764**,
Macro-Average Precision 0.632 -> **0.638**, val_loss 4.45 -> **2.77**. Better on all three.
neaq's effect on the marine mammals is mixed:
  - Megaptera novaeangliae 0.00 -> 0.60, Tursiops truncatus 0.125 -> 0.54,
    Balaenoptera acutorostrata 0.20 -> 0.80, Halichoerus grypus 0.20 -> 0.67,
    Phoca vitulina 0.39 -> 0.78, Pelecanus occidentalis 0.41 -> 0.96,
    Sternula antillarum 0.00 -> 0.05, Calidris alba 0.00 -> 0.08.
  - **Eubalaena glacialis entered as the new 70th class at 0.0 accuracy.** It did not
    exist in the 69-class pre-neaq run, so this is not a regression — it is a new class
    the model has not learned. The pooled annotation set has 1,827 Eubalaena rows in
    train / 70 in test, and neaq contributed only 41 of the train rows, so the class was
    mostly already present in the data and only crossed the split-by-parent-image
    keep threshold once neaq was added.
  - Regressions to watch: Cepphus grylle 0.40 -> 0.83 (up), but Chlidonias niger
    0.95 -> 0.70, Anas rubripes 0.98 -> 0.90, Oceanodroma leucorhoa 0.33 -> 0.10,
    Gavia stellata 0.75 -> 0.94 (up). Still at 0.0: Ardea herodias, Mergus merganser,
    Puffinus puffinus, Phalaropus lobatus, Pluvialis squatarola, Podiceps auritus,
    Pterodroma hasitata, Eubalaena glacialis.
Next: boem_conf/classification_model/finetune.yaml still points at dce44064 (pre-neaq).
Repoint it at e79ca03e — same plain-CropModel architecture, use_metadata already false,
so the swap is a one-line checkpoint change. Then decide whether Eubalaena at 0.0 is
acceptable for the incoming data or whether right whale needs targeted crops.

## (no job) — 2026-07-31 11:40 — config swap: both pointers moved to current checkpoints
Why: both configs were stale — a pipeline run would have used neither of the models
trained on 07-29/07-30. User asked to update the pointers before the incoming data.
Changes (uncommitted):
  boem_conf/boem_config.yaml detection_model.checkpoint
    a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt  (job 36523583, TIMEOUT)
    -> d1248ed881a847128752f86d1370add3/epoch09-val_cls0.0144.ckpt  (job 38309080, 10/10 epochs)
  boem_conf/classification_model/finetune.yaml checkpoint
    dce4406478df4191b18e21362156eaf8.ckpt  (job 38309081, 69 classes, pre-neaq)
    -> e79ca03e4e1b45f6820c009df6c190b5.ckpt  (job 38374817, 70 classes, neaq-inclusive)
  use_metadata stays false; both old and new classification checkpoints are plain
  CropModels (model.* only), so this swap carries no state_dict risk.
Verified: both files exist on disk; hydra compose of boem_config resolves both paths to
existing files and use_metadata to python bool False. The classification checkpoint's
state_dict has top-level prefix `model` only (no metadata_encoder), model.fc.weight is
70x2048, and the ckpt's top-level `label_dict` has 70 entries including Eubalaena
glacialis. Note hyper_parameters {model, config, config_args} are all None — same as
dce44064, and load_from_checkpoint fills them in, so not a problem.
Next: commit. The whole working tree is still uncommitted on `augmentation-hard-negatives`
(neaq prepare fix, annotator dir fix, LS empty-annotation guard, both config swaps).

## 38458860 — 2026-07-31 11:40 — submit_USGS_detection_augs.sh — SUBMITTED
Why: the outstanding half of neaq inclusion. Detection 38309080 finished 07-30T04:05,
before the neaq-inclusive prepare 38374816 ran at 12:10 the same day, so the current
detector has never seen a neaq frame while the classifier now has. No code change
needed — the crops on disk are already right (train.csv/test.csv mtime 07-30 12:31,
608,980 / 134,154 rows, 2,255 neaq rows in train).
Defaults unchanged from 38309080 (POSITIVE_BATCH_FRACTION=0.90, batch 64, workers 32,
lr 0.001, 10 epochs) so this is a clean data-only comparison against
d1248ed881a847128752f86d1370add3.
One resource change: --mem 200GB -> 300GB. 38309080 reported sstat MaxRSS 244GB against
the 200GB request and survived, and this crop set is ~4.5% larger; a 12h job is not worth
losing to an OOM.
Result: PENDING (Priority) as of 11:40. Expect ~12.5h on hpg-b200 (~72 min/epoch).
Next: compare final val_classification against 38309080's 0.0144 — a data-only delta.
Do NOT compare box_recall between runs without accounting for empty-frame ratio; see
deepforest_box_recall_issue.md. If it beats 0.0144, repoint
boem_conf/boem_config.yaml detection_model.checkpoint at the new hash. Watch that MaxRSS
now fits under 300GB. Checkpoints land in
/blue/ewhite/b.weinstein/BOEM/training/checkpoints/<comet_hash>/ — use those mtimes as
the liveness signal, not the stdout log (the rich progress bar does not flush to file).
