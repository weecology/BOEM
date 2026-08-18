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

## 38458860 (result addendum) — detection augs, neaq-inclusive crops
COMPLETED 2026-08-01T01:59, 12h53m, all 10 epochs. Comet hash
6fd54c2201e14ce99de8634e4a7fe5cb. Best epoch08-val_cls0.0143, essentially tied with
38309080's d1248ed8/epoch09 0.0144 — folding neaq into the detector cost nothing and
gained nothing measurable on val_classification. Final-epoch val_classification 0.0252
is epoch09 noise, not the selected checkpoint.
Next: boem_conf/boem_config.yaml still points at d1248ed8/epoch09. Left alone
deliberately — the 2026-08-06 retrain (38834235) supersedes both, so the pointer swap
should happen once against that result rather than twice.

## 38834217 — 2026-08-06 14:05 — submit_prepare_annotations.sh (--mem=128GB) — SUBMITTED
Why: fold the 2026-08-06 Label Studio pull into detection/crops and the UBFAI training
CSVs. scripts/download_annotations.py wrote 62 new CSVs (train 3,012 rows / 130 images /
15 flights; validation 232 / 78 / 42; review 898 / 525 / 5). Most are re-downloads of
tasks left on the server by the 2026-07-29 pull — genuinely new imagery is 74 train, 1
validation, 283 review images. All 62 mirrored into annotations_backup/ and committed
(8b12eec); backup --check reports 0 new / 0 changed / 1,137 total.
Used submit_prepare_annotations.sh rather than submit_prepare_USGS.sh so the zero-shot
holdout is PINNED to JPG_20260202_141900 + JPG_20260201_134000. The 07-30 run
(38374816) used the unpinned script and drew JPG_20260202_094800 + JPG_2023_Dec14 at
random, so its train/test split is not comparable to anything before or after it.
Pinning from here on makes successive runs a data-only comparison.
--mem bumped 64GB -> 128GB on the command line (not in the file): 38309076 fit in 64GB
at 582k train rows, but the crop set has grown since and 38374816 asked for 128GB.
Stage 0 is incremental — only images whose annotation CSV is newer get re-cropped — so
expect roughly the 733 touched images to refresh, not the full tree.
Result: **COMPLETED, 28m35s.** Zero-shot holdout is the pinned pair
{JPG_20260201_134000, JPG_20260202_141900}, as intended. AWS source filter kept 419,012
human rows (dropped 4,095,159 machine); 398,887 rows / 63,644 images after the image
filter. Hard-negative conversion 26,589 -> 17,023 markers (train), 1,666 -> 811 (test);
post-relabel dedup dropped 165,593 train / 6,392 test rows.
Totals: **train 621,844 rows** (was 608,980), **test 132,578** (was 134,154),
**zero_shot 22,996** (was 13,768).
test shrinking and zero_shot nearly doubling is NOT data loss — it is the pinned holdout
swapping which flights sit in which split. 38374816 held out JPG_20260202_094800 +
JPG_2023_Dec14 at random; those flights are back in train/test now and the two pinned
flights left. Only train-vs-train comparisons against 38374816 are meaningful, and only
loosely; from 38834217 onward the split is stable and successive runs are clean
data-only deltas.
Next: none. 38834235 and 38834236 released from Dependency to Priority at 14:35.

## 38834235 — 2026-08-06 14:05 — submit_USGS_detection_augs.sh — SUBMITTED (afterok:38834217)
Why: retrain detection on the 08-06 crops. Defaults unchanged from 38458860
(POSITIVE_BATCH_FRACTION=0.90, batch 64, workers 32, lr 0.001, 10 epochs, 300GB), so
this is again a data-only delta.
Result: PENDING. Expect ~13h on hpg-b200.
Next: compare best val_classification against 0.0143 (38458860) and 0.0144 (38309080).
Do NOT compare box_recall across runs — see deepforest_box_recall_issue.md. If it wins,
repoint boem_conf/boem_config.yaml detection_model.checkpoint at the new hash; that
pointer is currently two retrains stale.

## 38834236 — 2026-08-06 14:05 — submit_USGS_classification.sh — SUBMITTED (afterok:38834217)
Why: retrain the metadata-free classifier on the same 08-06 crops. Runs concurrently
with 38834235; they read the crops read-only and use separate environments (.venv for
detection, .venv-classification for classification) so neither job's uv sync can mutate
the other's.
Result: PENDING. Expect ~3h.
Next: compare Micro-Average Accuracy against 38374817/e79ca03e's 0.764 and watch whether
Eubalaena glacialis moves off 0.0 accuracy — the 08-06 review pull added 283 new images
from the neaq camera dirs, which is where right whale crops come from. If it improves,
repoint boem_conf/classification_model/finetune.yaml at the new hash.

## 38834235 (result addendum) — detection, 08-06 crops, posfrac 0.90
COMPLETED 2026-08-07T03:40, 12h49m, all 10 epochs. Best val_classification 0.01369,
beating 38458860's 0.0143 and 38309080's 0.0144 — the 08-06 crop set is a real gain.
zero_shot_evaluation_val_classification 0.01305 against pretraining 0.1656.
Next: boem_conf/boem_config.yaml detection_model.checkpoint is now three retrains stale
(still d1248ed8/epoch09). Deliberately left alone again — the 39211658 posfrac-0.5 run
supersedes this, so the pointer swap should happen once against whichever of the two wins.

## 39211655 — 2026-08-11 15:33 — submit_prepare_annotations.sh (--mem=128GB) — SUBMITTED
Why: fold the 2026-08-11 Label Studio pull into detection crops before retraining.
scripts/download_annotations.py wrote 10 new CSVs (train 514 rows / 73 images / 8
flights; validation none; review 122 / 32 / 2). Unlike the 08-06 pull these are all
genuinely new labels, not re-downloads — the 08-06 session ran delete_completed_tasks.py
so nothing stale was left on the server. All 10 mirrored into annotations_backup/ and
committed (cea8855); backup reports 10 new / 0 changed / 1,271 total.
Used submit_prepare_annotations.sh so the zero-shot holdout stays PINNED to
JPG_20260202_141900 + JPG_20260201_134000, matching 38834217. That keeps this a clean
data-only delta against the 08-06 crops (train 621,844 rows / test 132,578 /
zero_shot 22,996).
--mem 128GB on the command line, as with 38834217. Stage 0 is incremental, so expect
roughly the ~105 touched images to re-crop, not the full tree. 38834217 took 28m35s.
Result: PENDING.
Next: none; 39211658 releases on afterok.

## 39211658 — 2026-08-11 15:33 — submit_USGS_detection_augs.sh — SUBMITTED (afterok:39211655)
Why: **POSITIVE_BATCH_FRACTION=0.5, down from the 0.90 default.** Every detection run to
date (38309080, 38458860, 38834235) reserved 90% of each batch for annotated images; the
resulting models are miscalibrated on empty frames, which is the bulk of real survey
imagery. 0.5 gives an even split of annotated images and hard negatives per batch.
Passed as `--export=ALL,POSITIVE_BATCH_FRACTION=0.5` rather than by editing the script,
so the file's 0.90 default is untouched and this run is reproducible from the ledger
alone. EXP_NAME auto-derives to detection_posfrac0.5_39211658.
Everything else is unchanged from 38834235 (batch 64, workers 32, lr 0.001, 10 epochs,
300GB, hpg-b200), so the only deltas against it are the 08-11 labels and the batch
fraction.
Result: PENDING. Expect ~13h.
Next: compare best val_classification against 38834235's 0.01369. Note this is NOT a
clean comparison — val_classification is a loss on the val set, and changing the batch
composition changes what the model is optimized for, so a slightly worse number here can
still be the better field model. Weight zero_shot_evaluation_val_classification and
per-image false positives on empty frames more heavily than val_classification.
Do NOT compare box_recall across runs — see deepforest_box_recall_issue.md.
Whichever run wins, repoint boem_conf/boem_config.yaml detection_model.checkpoint once;
it is currently three retrains stale at d1248ed8/epoch09.

## 39223916, 39225777 — 2026-08-11 — benchmark_inference.py — COMPLETED
Why: Size inference throughput on one B200 ahead of the incoming Globus survey, to
answer GPU utilization / sustainability / hours-per-TB at the configured batch size.
60-100 images from JPG_20260712_100400 (6464x4852, 5.6 MB, 35 patches at patch_size
1000). 39225777 runs each batch size in a FRESH process; 39223916 reused one process
and produced a false OOM at batch 32 from the prior case's fragmentation.
Result: THE CONFIGURED predict.batch_size=64 CANNOT RUN. deepforest's
predict_tile(dataloader_strategy="batch") batches IMAGES, not patches — MultiImage
__len__ is len(paths) and collate_fn flattens every crop — so batch_size=64 means
64*35=2240 patches in one forward pass, needing ~416 GB on a 179 GB card. 32/48/64 all
OOM. The comment in boem_conf/boem_config.yaml ("64 fits all patches of an image in one
pass") has the unit wrong; batch_size=1 already puts all 35 patches in one pass.
Throughput is INVERSELY related to batch size — bigger is strictly worse on both axes:
  batch  img/s  peak GB  h/TB      batch  img/s  peak GB  h/TB
      1   3.88      6.5  13.5         16   2.50    101.5  20.9  (from 39223916)
      2   3.69     12.8  14.2         24   1.49    152.2  35.0
      4   3.28     25.5  15.9         32+  OOM
      8   2.66     50.8  19.7
Peak memory is linear at ~6.4 GB per image in the batch. GPU SM utilization sampled
1-30% and bursty; memory at batch 1 is 6.5/179 GB = 3.6%. The workload is data-bound
(JPEG decode + Lustre read + patch construction), not GPU-bound — which is exactly why
raising the batch size hurts. One B200 at batch 1 = ~3.9 img/s = ~22 MB/s = ~13.5 h/TB.
Detection only; the CropModel pass was not measured.
NOTE: the batch_size=16 case in 39225777 died on `Disk quota exceeded` writing
hparams.yaml, not on OOM — /blue/ewhite is 33T/33T full. Its numbers above come from
39223916. Same quota failure killed 39211655 (prep_ann) and stranded 39211658.
Next: set predict.batch_size to 1 (or 2) and fix the misleading comment. Free /blue
before the Globus transfer lands. For a full survey, fan out across the 8 GPUs per
b200 node rather than raising the batch size.

## 39271997-39272011, 39272127-39272145 — 2026-08-12 11:31 — submit_all_flights.sh — RUNNING
Why: First pass over the 20 new JPG_202607* flights (533,893 images, ~3.3 TB) with the
detector repointed to a09c6933/epoch16 and the classifier to 56e8585a. Full coverage:
active_learning.pool_limit=null, so every image is scored rather than a 500-image sample.
Also the first run of the redefined human-review band (see below) at human_review.n=30.

Checkpoints: detection a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt;
classification buffer_30/56e8585add144d1eabba1f00c411b985.ckpt. The classifier was
verified to be model.* only with no metadata_encoder and 70 classes — same architecture
as the e79ca03e it replaces — so use_metadata stays false and nothing else moved.

Human review was redefined before this run. It used to be
  score >= 0.8 AND cropmodel_score < 0.3, then a second cut at cropmodel_score <= 0.6.
The second cut was implied by the first, so uncertain == filtered and
confident_predictions was ALWAYS EMPTY — every confident_predictions.csv logged to Comet
before today is an empty table, and "Images auto-annotated: 0" meant nothing. The 0.8
detection floor was also dead, since the pool is already cut at predict.min_score=0.85.
Worse, the rule excluded the genuinely ambiguous cases: it kept only cropmodel_score
< 0.3 and threw away the 0.3-0.6 band where the classifier is actually torn.
Now: score >= human_review.min_detection_score (0.85) AND
review_low (0.3) <= cropmodel_score <= review_high (0.6) -> review;
above review_high -> confident/auto-annotate; below review_low -> ignored as a likely
spurious detection. New keys live under human_review so that active_learning's
min_classification_score keeps its real meaning as a floor in select_images and report.

GOTCHA — check_annotations does not survive a parallel fan-out. All 20 jobs launched
with check_annotations=True, and each one downloads completed Label Studio tasks AND
DELETES them from the same shared project. The 7 that started together each pulled a
different count (914 / 894 / 894 / 893 / 710 / 245 / 241 / 236) because they were eating
each other's queue, and three logged HTTP 500 "Save with update_fields did not affect any
rows" deleting tasks a sibling had already removed. Not fatal — writes are per-task under
each task's own flight_name — but the pull got split at random across 7 jobs. The 10
still-pending jobs were cancelled and resubmitted as 39272127-39272145 with
check_annotations=False; the 10 already running kept the flag.
Next: pull annotations ONCE before a fan-out, then submit every flight with
check_annotations=False. Worth enforcing in submit_all_flights.sh rather than by hand.
Throughput lands in /blue/ewhite/b.weinstein/BOEM/throughput.csv (one line per flight:
images, bytes, wall seconds, img/s, hours-per-TB) to answer the per-TB-per-GPU question.

## 39272217 (L4/hpg-turin), 39272218 (B200/hpg-b200) — 2026-08-12 — submit_worker_sweep.sh — SUBMITTED
Why: Test whether the DataLoader is the bottleneck behind 39225777's result (throughput
inversely related to batch size, GPU utilization in the single digits). Sweeps
predict.workers 0/2/5/10 at batch_size=1 on the SAME 60 images of JPG_20260712_100400,
run on both an L4 (24 GB) and a B200 (179 GB). If the workload is data-bound, workers
should move throughput and the two GPUs should land close together; if the L4 is much
slower, the forward pass matters more than 39225777 suggested.
batch_size pinned to 1 because batch 4 peaked at 25.5 GB and does not fit an L4. A
trailing batch=2 case probes L4 headroom (12.8 GB on the B200, so it should just fit).
Each case runs in a fresh process. Fixed en route: the benchmark warmed up at a
hardcoded batch_size=32, which would have OOM'd the L4 before any case ran; it now warms
up at the smallest batch in the sweep.
NOTE: detection checkpoint changed under this work — config now points at
a09c6933/epoch16, while 39223916/39225777 measured d1248ed8/epoch09. Same RetinaNet and
same 256.8 MB file, and timing is dominated by decode+I/O, but this sweep is the first
measurement on the current checkpoint.
Result: COMPLETED (L4 9m58s, B200 3m59s). Workers is the lever ON THE B200 ONLY.
  B200 (batch=1): w0 1.47 img/s (35.6 h/TB) · w2 2.11 · w5 4.11 (12.7 h/TB) · w10 4.03
  L4   (batch=1): w0 0.65 · w2 0.79 · w5 0.78 · w10 0.78 (67 h/TB)
The two GPUs did NOT land close together: the B200 is 5.3x the L4 at the same batch size.
B200 sampled 12-26% util at 6.5/179 GB (data-bound, workers=5 is the knee, 10 adds
nothing — the job only had 12 cores). L4 sampled 92% util with p90 100% (compute-bound;
workers buy nothing past 2). batch=2 on B200 gave 3.93 img/s, again worse than batch=1,
confirming 39225777.

THE "5.3x PIPELINE OVERHEAD" IS NOT REAL — it is a hardware mismatch, not overhead.
The 4.11 img/s above is the B200; submit_all_flights.sh requests `--gpus=l4:1`, so the
20 production flights ran on L4 hardware. The like-for-like comparison is L4 detection-
only 0.78 img/s vs L4 full-pipeline 0.782 img/s. Checked against throughput.csv for all
20 flights: predicted seconds at 0.78 img/s vs actual is -0.3% across 682,578 s
(189.6 GPU-h actual vs 190.1 predicted). Classification + crop writing + Label Studio +
flythrough + report cost nothing measurable and there is no overhead left to profile.
The reason they are free: the whole survey produced 91 boxes >=min_score 0.85 across
533,893 images (~0.017%). Jobs 39272011 and 39272144 found ZERO detections and still
ran at 0.783/0.786 img/s — identical to every other flight. Per-flight fixed cost is
~50 s (39271997: 5616 images, 7253 s actual vs 7200 s predicted).

BUG in scripts/benchmark_inference.py gpu_ceiling(): it builds `torch.rand(batch_size,
3, patch_size, patch_size)`, i.e. ONE patch per forward call at batch_size=1, while real
inference pushes all 35 patches of an image in one pass. So the reported "ceiling" is
latency-bound on kernel launches and is not a ceiling — the B200's measured 144 patch/s
EXCEEDS its own reported 136 patch/s ceiling. The L4 "1.12 img/s equivalent" number is
the same artifact and should not be quoted. The L4 compute-bound conclusion still holds,
but it rests on the 92% mean / 100% p90 utilization, not on the ceiling figure.
Next: the lever is the GPU, not the pipeline and not predict.workers. Move production
off `--gpus=l4:1` to hpg-b200 (5.3x: ~190 GPU-h -> ~36). B200 nodes are 112 cores /
8 GPUs = 14 cores per GPU, and a process needs ~6 (workers=5 + main), so ~2 concurrent
flights per GPU and ~16 per node; at 6.5/179 GB memory is nowhere near binding. Fix
gpu_ceiling to feed batch_size * patches_per_image patches before quoting it again.
SEPARATE CONCERN, not a throughput issue: 91 detections from 3.3 TB deserves a look at
whether min_score=0.85 is too high. The cache cannot answer it — pool_predictions.csv is
written AFTER the min_score filter, so the discarded score distribution is unrecoverable
without a rerun at a lower threshold.

## 39385379 — 2026-08-13 — submit_threshold_sweep.sh — SUBMITTED
Why: re-derive predict.min_score for the checkpoint that will actually go live. The
20-flight run inherited 0.85 from a differently-calibrated model. Same-flight control on
JPG_20260202_141900: a1c5649 at cut 0.85 gave 406 boxes (score mean 0.950, max 0.998,
59% above 0.95); a09c6933/epoch16 on the same flight at cut 0.50 gave 103 boxes with
**max 0.798** — it would have returned ZERO at 0.85. On the 20 new flights it barely
clears: mean 0.883, max 0.935, 80 of 100 boxes in the single bin just above the cut.
Ben's criterion: favour recall — accept false positives so annotators are not handed a
queue that is already missing real objects.
Confirms the cache note above with the code cite: pipeline.py:448-450 states the cache
stores only boxes >= the min_score it was written at, and the same block forces
re-prediction when the current min_score is lower. All 20 min_score.txt read 0.85 and the
lowest score present anywhere is 0.85001.
Substrate instead: the 648 human-reviewed full frames of the PINNED zero-shot holdout
(JPG_20260202_141900 + JPG_20260201_134000) — 1,279 annotation rows = 1,229 real objects
plus 50 human-marked FalsePositive, all 648 frames resolving on disk. Both flights are
excluded from training for 38834235, so it is a clean holdout, and within a reviewed
frame the human marked every object including ones the detector missed — which is what
makes recall measurable at all. Unlabelled survey imagery cannot answer a recall
question; you can count boxes but never what was missed.
scripts/threshold_sweep.py forces score_thresh 0.01 (otherwise the model's own default
silently truncates the low end and every recall number below it is wrong) and sweeps
0.05-0.99 offline: box recall, precision, FP/image, image-level recall, greedy IoU
matching at 0.4. FalsePositive-labelled frames are excluded from both the positive set
and the FP count so they bias neither metric.
Second measurement in the same job: 2,000 random UNSCREENED frames from the same flights
-> images and boxes per 1,000 at each threshold, i.e. the annotator queue size. Needed
because the reviewed frames were themselves selected by a detector, so they
under-represent empty ocean and their precision/FP numbers are optimistic. Recall is
unaffected by that selection bias; queue cost is not.
Two checkpoints in one job: 55d29b2c/epoch08 (job 38834235, best val_cls 0.01369, the
candidate) and a09c6933/epoch16 (the incumbent in boem_config). batch_size=1, workers=5.
Submitted to hpg-b200 rather than l4 per the entry above.
Result: **COMPLETED, 27m22s.** 648 labelled frames / 1,229 objects, 2,000-frame unscreened
queue sample, greedy IoU 0.4. JSON in /blue/ewhite/b.weinstein/BOEM/threshold_sweep_*.json.

a09c6933/epoch16 (the incumbent, raw score range 0.0100-0.9310):
   thresh  box recall   prec   FP/img  img recall  queue img/1k  queue box/1k
     0.05      67.8%   31.6%    2.78      100.0%        379.5        5144.0
     0.10      67.0%   60.6%    0.83      100.0%        201.5         923.0
     0.15      66.1%   70.7%    0.52       99.8%        112.0         290.0
     0.20      65.0%   74.7%    0.42       99.0%         51.0         104.0
     0.30      63.0%   78.4%    0.33       97.8%         15.5          26.5   <- CHOSEN
     0.50      57.9%   84.4%    0.20       94.1%          1.0           1.0
     0.85      15.9%   98.0%    0.01       29.9%          1.0           1.0   <- was live

**0.85 was discarding ~70% of occupied frames.** That is the headline; it is not a
marginal miscalibration.

**Image-level recall is the metric that matters, not box recall.** An annotator labels
every object in a frame they are handed, so a box missed inside a QUEUED frame is
recovered for free — only a frame that is never queued is truly lost. Box recall is the
wrong yardstick for a review-queue threshold.

**Box recall ceilings at 67.8% at ANY threshold (even 0.05).** A third of ground-truth
objects are invisible to this detector no matter where the cut goes, so the "NOT REACHABLE
at any threshold" lines for the >=80/90/95/99% recall targets are a detector finding, not
a threshold one. Worth its own investigation (patch size? architecture?); min_score
cannot touch it.

**Do NOT swap to 55d29b2c/epoch08.** The 38834235 candidate has the better
val_classification (0.0137 vs 0.0163) and is WORSE on this holdout at every single
threshold — 97.8% img recall at 0.20 vs a09c6933's 99.0%, at a higher queue cost
(20.0 vs 51.0 img/1k is not a fair trade once recall is matched: 55d29b2c at 0.10 gives
65.7% box recall for 125 img/1k, where a09c6933 at 0.15 gives 66.1% for 112). The
three-retrains-stale pointer turns out to be the correct one — val_cls was misleading.
Caveat on the queue columns: they come from Feb Gulf frames and the July imagery is much
sparser (91 boxes >=0.85 across 533,876 images, below the 2,000-frame sample's 1-per-1,000
resolution floor), so treat them as upper bounds. Everything at/below thresh 0.50 in those
two columns is at the sample's resolution limit and should not be read precisely.
Next: DONE — see the 2026-08-14 config entry below.

## (no job) — 2026-08-14 — threshold recalibration: three gates, not one
Why: act on 39385379. Chose **0.30** (97.8% image recall, 78.4% precision, ~15.5 img/1k)
rather than the more permissive 0.15/0.20 — deliberately trading ~1-2 points of image
recall for a queue 3-7x smaller.
There turned out to be THREE independent detection gates, and lowering only the obvious
one would have re-predicted the whole 3.45 TB survey and then filtered every new box back
out before it reached a human:
  predict.min_score                    0.85 -> 0.30   boem_config.yaml
  human_review.min_detection_score     0.85 -> 0.30   active_learning.py:161
  active_learning.min_detection_score  0.80 -> 0.30   starved the TRAINING queue; the
                                                      survey yielded 91 boxes >= 0.85
The third one was not in the original plan and is easy to miss — it lives in the
active_learning block, is spelled the same as the human_review key, and had a different
value (0.8, not 0.85). Grep for `min_detection_score` before changing a threshold again.
The chosen value is now recorded NEXT TO the checkpoint hash in boem_config.yaml with the
full sweep table inline, since that missing pairing is exactly what broke here.
Per-flight Label Studio upload is capped regardless of threshold: active_learning.n_images
100 + active_testing.n_images 1 + human_review.n 30 = <=131 images/flight, <=2,620 across
20 flights. Lowering the threshold does NOT flood Label Studio; it means those ~131 slots
are drawn from a pool that actually has something in it.
Note pipeline.py:638 takes `.head(human_review.n)` after sorting by DETECTION score
descending, so the review queue still skews to the confident end of the 0.30-0.95 range.
Verified: hydra resolves all three to 0.3 and both checkpoints exist on disk.
Committed 6461c64 on branch review-band-and-202607-run.

## 39396376-39396399 (20 jobs) — 2026-08-14 09:4x — submit_all_flights.sh — SUBMITTED
Why: re-run the 20 JPG_202607* flights (533,876 images, 3.45 TB) at the recalibrated
min_score 0.30. The 2026-08-12 pass at 0.85 returned 91 boxes across the entire survey and
two flights (39272011, 39272144) returned literally zero — that was the threshold, not the
imagery.
Cache: no manual cleanup needed. All 20 .prediction_cache/min_score.txt read 0.85 and
pipeline.py:449-459 force re-prediction whenever the current min_score is BELOW the cached
one, so this is automatically a full re-predict. (Verified before submitting.)
Annotations were pulled ONCE beforehand (scripts/download_annotations.py: 154 train rows /
1 image, 390 review rows / 218 images; validation none) and all 20 jobs run with
check_annotations=False, per the 08-12 fan-out gotcha where 7 concurrent jobs ate each
other's Label Studio queue. download_annotations.py does not delete server-side, so
nothing was consumed. 71 new CSVs mirrored to annotations_backup/ and committed.
HARDWARE — stayed on L4 despite the measured 5.3x B200 advantage. hpg-b200 was 85% drained
at submit time (51 nodes `drain` with "Kill task failed", 538 jobs pending, 14 running,
**0 free B200 GPUs**) against **101 free L4 GPUs** on hpg-turin. submit_all_flights.sh now
takes a `--b200` flag instead of hardcoding the partition, so this is a one-flag change
when b200 recovers; check `sinfo -p hpg-b200` first. Expect ~190 GPU-h again on L4 vs ~36
on B200.
Result: **ALL 20 COMPLETED** by 2026-08-15 12:49 (longest JPG_20260710_163500, 21.8h).
Note the job IDs are NOT contiguous: 39396384/385/391/398 belong to other users. The 20
BOEM jobs are 39396376-383, 386-390, 392-397, 399.
**41,694 boxes across 19,080 images (3.57% of 533,876 frames), up from 91.** No flight
returned zero; JPG_20260713_141400 went 0 -> 1,025. Runtime was unchanged despite ~458x
more boxes (192.0 GPU-h vs 189.6 on 08-12), confirming the 08-12 finding that
classification + crop writing cost nothing measurable — the workload is decode+I/O bound.

**THE 0.30 NEVER TOOK EFFECT — the live cut is 0.40.** Lowest score anywhere in the survey
is 0.400001 and the (0.30, 0.40] bin is empty across all 20 caches. Cause: the checkpoint
a09c6933/epoch16 carries `score_thresh: 0.4` in its saved hyper_parameters, and
`src/detection.py:load()` uses `main.deepforest.load_from_checkpoint()`, which restores
that config — so the RetinaNet head never emits a box below 0.4 and `predict.min_score=0.3`
is a no-op below it. This is exactly the trap scripts/threshold_sweep.py:104-114 guards
against by forcing `m.model.score_thresh = floor`; the production load path does not.
Cost per the 39385379 sweep: image recall 96.6% at 0.40 vs 97.8% at 0.30 (box recall 60.8%
vs 63.0%). Real but small — do NOT re-run 3.45 TB for it alone; fix the load path and pick
it up on the next pass. There are now FOUR gates, not the three logged on 08-14.

Label Studio (live API read 2026-08-17): 633 tasks from this run — 490 review, 107 train,
36 validation. 19 of 20 flights contributed a review batch (<=30 each).

**JPG_20260711_141200 (39396382) exited rc=1 but SLURM says COMPLETED.** The sbatch wrapper
records `$?` to throughput.csv and keeps going, so the failure is invisible in sacct — read
the rc column. `paramiko.ssh_exception.SSHException: Server connection dropped` at
label_studio.py:545 (`sftp_client.put`) after 30 of its images; the retry loop there only
catches EOFError, not SSHException. Predictions are safe (2,584 boxes cached), but the
flight contributed 0 train (of 42) and 0 review (of 30) tasks — only 1 validation task
landed. Re-upload only; nothing needs recomputing.

**First human review is 34 FalsePositive vs 3 Object** across the 30 annotated review tasks
(29 from JPG_20260710_155800, 1 from JPG_20260711_090800). Those annotations are still
server-side only — they postdate the 08-14 pull and are NOT yet in annotations_backup/.
Caveat: the review band is by construction the classifier's uncertain slice, so a high FP
rate there is expected and does not directly indict the 93% auto-accepted bucket — but that
bucket (38,769 boxes) has had zero human checking, and the species counts all come from it.
Two flights (JPG_20260712_113200, JPG_20260712_100400) hold 25,280 of the 41,694 boxes;
their detections are ~20-25px and cluster on bright specks in chop that are hard to
separate from whitecaps by eye.
Next: (1) pull the 30 new review annotations and back them up; (2) re-upload
JPG_20260711_141200 and catch SSHException in upload_images' retry loop; (3) fix
detection.py:load() to override score_thresh from predict.min_score; (4) get review
coverage across more flights + spot-check the auto-accepted bucket before anyone quotes
these as counts.
Summary written up at https://claude.ai/code/artifact/cdea0c3f-0f0e-40de-a778-82d522c30b6a

## 39551409, 39553349, 39555477 — 2026-08-17 — classifier confusion matrix (56e8585)
Why: the 20-flight JPG_202607* survey reports 50.3% Somateria mollissima (20,979 of 41,694
boxes) and 88.4% cold-water North Atlantic species, in a **Gulf of Mexico** survey where
none of them occur. Hypothesis under test was that eider is absorbing misclassified
Morus bassanus. Needed the classifier's own val-set confusion matrix to check.
First two submissions failed fast: `CropModel.get_transform()` takes `augmentations=`,
not `augment=` (39551409), and `groupby.apply(..., include_groups=False)` drops the
grouping column so `d.true` is gone inside the lambda (39553349). Third COMPLETED in
~1 min on one B200; 3,610 val crops, 70 classes.
Note `predict_step` reads a 2-tuple batch as `(images, metadata)`, so a bare ImageFolder
feeds LABELS in as metadata — scripts/classifier_confusion.py wraps it in an images-only
dataset. Harmless on this metadata-free checkpoint, a silent corruption on d8995ca8.

**The gannet hypothesis is wrong, and the truth is worse.** Morus->Somateria is 1/100 and
Somateria->Morus is 0/100; gannet holds 0.90 recall / 0.94 precision. The two classes
separate cleanly in-distribution. The survey's eiders are not hidden gannets.
**They are not birds.** 200px context crops of conf>0.99 Somateria show dense linear
wind-rows of foam blobs — 50+ "birds" per 400px patch. Fratercula arctica (2,336 boxes) is
a near-perfectly stereotyped artifact: a vertical pair of chromatically-fringed dots, the
same shape every time. The classifier partitions foam-blob geometry into species by
orientation and count.
Root cause is structural: **70 classes, no background/FalsePositive class.** A softmax must
name a species for every crop the detector hands it, and out-of-distribution foam lands on
the Atlantic sea ducks with saturated confidence (Somateria median score 1.000, 92.8% above
0.9). Classifier confidence is not a usable filter even in-distribution — val accuracy only
moves 0.759 -> 0.844 from score>=0 to score>=0.99.
**Detection score is the discriminator, not classification score.** On the 37 reviewed
boxes, real objects sit at det 0.71-0.89 and the 34 FalsePositives at median 0.48;
det>=0.60 keeps 3/3 real and drops 31/34 FP, where cls>=0.90 keeps 1/3 real and 2/34 FP.
Survey-wide the cold-water share falls 88.2% -> 22.0% and top-1 flips from Somateria to
Thalasseus maximus (Royal Tern, with Laughing Gull / Brown Pelican / Larus behind it — a
coherent Gulf-in-July list) once det>=0.70. 82.3% of the survey sits in det 0.40-0.50.
Two flights carry 25,280 of 41,694 boxes and 120 of them survive det>=0.70.
**Marine mammals are separately broken** and this is not foam — it is in-distribution val.
Megaptera novaeangliae precision 0.02 (82 predicted, 2 right) — it is the sink for
Delphinus delphis (recall 0.15, 46/59 -> Megaptera). Balaenoptera acutorostrata and
Eubalaena glacialis are 0.00 recall. **Tursiops truncatus, the configured
active_learning.target_label, is 0.29 recall / 0.33 precision.**
Metadata: **no 202607 captures CSVs exist anywhere** on /blue or /orange — the directory
ends at 202602 — and the JPGs carry zero EXIF, so July positions are unrecoverable without
the Globus pull. The metadata-capable checkpoint d8995ca8 is on disk and intact.
Outputs: /blue/ewhite/b.weinstein/BOEM/classifier_confusion_56e8585_matrix.csv and
_predictions.csv; annotator crop sheets in /blue/ewhite/b.weinstein/BOEM/qc_crops_202607/.
Next: (1) Globus the 202607 metadata; (2) add a background class before any recount — a
range filter alone cannot fix a detector that emits foam; (3) do NOT quote species counts
from this run.

## (no job) — 2026-08-17 — reconciling the 0.85 finding with the foam finding
The two are not in conflict and the distinction is easy to lose: `predict.min_score` gates
the DETECTION score (`score`), while the 0.99/1.00 confidences in the eider write-up are
CLASSIFICATION scores (`cropmodel_score`). Across the survey they correlate at **0.0085** —
detection median 0.441 / max 0.935, classification median 0.9994. The detector barely firing
and the classifier being certain are simultaneously true because the classifier never sees
the detection score and has no class with which to reject a crop.
The 08-12 result stands: at detection >= 0.85 this same 0.30 rerun yields 100 boxes across
533,876 frames (vs the 91 counted then), and they are mostly REAL — 76 Thalasseus maximus
plus Leucophaeus atricilla, Pelecanus occidentalis, Larus argentatus/delawarensis. 0.85 was a
high-precision, near-zero-recall corner, not a garbage filter.
**What did not transfer was the calibration substrate.** Measured at a common 0.50 floor:
  Feb holdout (JPG_20260202_141900 + JPG_20260201_134000)   79.1% of boxes >= 0.85, median 0.946
  July survey (20 flights)                                   1.4% of boxes >= 0.85, median 0.537
The 39385379 sweep asked "what is lost by cutting at 0.85" on imagery where four-fifths of
detections already sat above 0.85. July imagery has essentially nothing there. The Feb mix at
>=0.50 is a clean winter-Gulf list (Larus delawarensis 104, Thalasseus maximus 74, Leucophaeus
atricilla 55, Pelecanus occidentalis 53) including 21 Morus bassanus, which are CORRECT for the
Gulf in February. Cold-water share 20.5% Feb vs 81.9% July.
Lowering the threshold did not cause the foam problem, it revealed it — the defect is the
missing background class and is orthogonal to the cut. Returning to 0.85 would re-hide it and
discard the real birds. The 08-14 entry's caveat correctly flagged the Feb queue columns as
upper bounds, but framed it as queue SIZE; precision was the part that did not survive the move.
Actual misstep: 190 GPU-h committed to 3.45 TB on a February calibration with no spot-check of
a few hundred July frames at the new cut. Do that before the next re-predict.

## (no job) — 2026-08-17 — the 19 Morus bassanus boxes independently validate the det gate
Pulled crops for all 19 survey Morus bassanus boxes (the class the original gannet hypothesis
was about). **Eight are unmistakable birds in flight** — long pointed wings, slender body,
shadow on the water. Eleven are foam. The eight real ones are EXACTLY the eight with detection
score >= 0.65 (range 0.664-0.831); all eleven foam crops fall below (0.400-0.573). No exceptions
either way. This class played no part in deriving the 0.60-0.70 gate from the 37 reviewed boxes,
so it is independent confirmation on a held-out class.
The real birds are probably not gannets — H-CAST calls them Sterna hirundo / Thalasseus maximus,
and terns are what belongs in Gulf water in July. Species label still wrong, but there is a real
bird underneath, which was never true of the eiders.
Share of each class clearing detection 0.65 separates real species from foam artifacts cleanly:
  Thalasseus maximus     218 boxes  median det 0.812   82.1% >= 0.65
  Morus bassanus          19        median det 0.515   42.1%
  Pelecanus occidentalis 141        median det 0.479   16.3%
  Leucophaeus atricilla  464        median det 0.447    4.7%
  Somateria mollissima 20,979       median det 0.442    0.5%
  Clangula hyemalis     7,717       median det 0.438    0.4%
  Fratercula arctica    2,336       median det 0.435    0.1%
Royal Tern behaves like a real species; the three dominant classes behave like nothing at all.
This per-class ">= 0.65 share" is a cheap screening statistic worth computing on any future run
before anyone reads the species table.
Crop sheet: /blue/ewhite/b.weinstein/BOEM/qc_crops_202607/Morus_bassanus.png

## 39613239 — 2026-08-18 — classifier retrain adding a sea turtle class
**Why:** No turtle class existed despite turtles being common in the annotations. Cause is the
two-word (binomial) label filter in `filter_annotations` / `USGS_classification.py`: sea turtles
are annotated at family/order rank — `Cheloniidae`, `Testudines`, `Chelonioidea`, `Dermochelyidae`
— which are single words, so every one of them was silently dropped before the >25-per-class
filter ever ran. The explicit drop list (`"Turtle"`, `"Reptile"`) removed the rest. The only
turtle labels that were binomial (`Dermochelys coriacea` 16 crops, `Caretta caretta` 2,
`Lepidochelys kempii` 2) were then too rare to clear >25.
Species-level common names DO exist in the AWS manifests in quantity (`Loggerhead Turtle` 3,241,
`Kemp's Ridley Turtle` 762) but every one is a Tallgrass/Normandeau MACHINE prediction — the
`source.startswith("private.")` human filter in prepare_USGS correctly drops them. Human-reviewed
turtle rows are family/order rank only.
**Change:** `TURTLE_LABELS` / `map_turtle_labels()` in `src/classification.py` collapse all turtle
taxa (both taxonomic and common-name forms) onto one two-word class `Chelonioidea sp`, applied
BEFORE the >25 and two-word filters in `filter_annotations`, `scripts/USGS_classification.py`, and
both filter sites in `src/pipeline_evaluation.py` so eval scores the class it trains.
`Reptilia`/`Reptile` deliberately NOT mapped in — class-rank catch-all, not a turtle ID (487 crops
left on the table; one-line change if wanted).
**Pool check (dry run of the real filter chain):** 3,081 raw turtle crop rows in
`/blue/ewhite/b.weinstein/BOEM/training/crops` (Testudines 1,590 + Cheloniidae 1,356 + Chelonioidea
80 + Caretta 18 + Dermochelyidae 15 + binomials 22); **1,826 survive `filter_annotations`** across
1,638 parent images. The ~40% loss is the pre-existing `xmin/ymin != 0` rule dropping boxes on a
patch edge, and applies to every class. 1,638 parents is far above the min_test_images=5 threshold,
so the class will not be dropped by `train_test_split_by_image`.
**Config:** unchanged from the previous run (`use_metadata=False`, expand=30, 45 epochs, lr 1e-5,
batch 96) so the turtle class is the only variable vs the 56e8585 baseline (67 classes).
**Next:** expect 68 classes. Check the `[turtles]` line in the log for the relabel count, then read
turtle precision/recall off the Comet confusion matrix — the class lumps hardshells and leatherbacks
together, so confusion with Mola mola (similar size/shape from altitude) is the thing to look for.
