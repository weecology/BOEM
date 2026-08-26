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

## 39614374 — 2026-08-18 — submit_USGS_hierarchical.sh — COMPLETED
Why: user asked to refresh the hierarchical (H-CAST family/genus/species) model, using the same
new annotations (including the new turtle class) that motivated 39613239's flat CropModel retrain.
39613239 was still RUNNING (mid-45-epoch) at submit time, but that's fine here: `USGS_classification.py`
writes `usgs_train_split.csv`/`usgs_val_split.csv` to `checkpoint_dir/buffer_30/<comet_id>/` *before*
`train()` is called (scripts/USGS_classification.py:347-351), so the split for the turtle-inclusive
run was already final and on disk (`buffer_30/a3dc30a085f5442393736ecd96b564c5/`, written 09:44,
54,765 train rows, 1,647 of them `Chelonioidea sp`) — no need to wait for training/checkpoint to finish.
**Found and fixed a bug before submitting**: `scripts/USGS_hierarchical.py` builds its label vocabulary
by matching split-CSV labels against `taxonomy.json` species leaves (`taxonomy_hier.py: _walk_species`
only returns rank=="Species" nodes). `Chelonioidea sp` is a synthetic label invented by
`src/classification.py`'s turtle fix (real taxonomy has no such scientificName), so it silently failed
to match and every turtle crop would have been dropped from the hierarchical train/val sets — the exact
same silent-loss bug 39613239 just fixed for CropModel, reappearing in the parallel pipeline. Fixed in
`scripts/USGS_hierarchical.py` (uncommitted): after `load_taxonomy_restricted_to_species`, if
`TURTLE_CLASS` ("Chelonioidea sp") is in the discovered labels but missing from `name_to_ids`, assign it
a fresh id at every level (species/genus/family) rather than dropping it. Checked the ancestor-extension
path (`_build_hierarchical_train_with_ancestors`, only active because `--annotations-dir` is passed) too:
it re-reads raw annotation CSVs and only 20 raw two-word turtle rows exist there (mostly binomials like
`Caretta caretta`), all far below the >25-per-class filter, so that path was already a non-issue and
was left alone.
Submitted with `BOEM_HIER_SPLIT_DIR` pinned explicitly to `buffer_30/a3dc30a085f5442393736ecd96b564c5`
(rather than relying on the script's auto-discover-newest-split logic) because an unrelated second
classification job (39613593, `submit_USGS_classification.sh`, submitted 09:43:42 by a separate,
unexplained invocation) was also queued on hpg-b200 at submit time; pinning avoids the hierarchical
run silently picking up whatever split that job writes if it finishes first.
Result: **COMPLETED rc=0 at 2026-08-20 04:01:10, elapsed 1-17:40:41 — ran its full 100-epoch
budget (`--epochs 100` in submit_USGS_hierarchical.sh:52, not visible in the log itself).
Best Species@1 76.72 (epoch 96); final epoch 99 Species@1 76.44, Species@5 92.46, Genus@1 82.21,
Family@1 89.09.** Still improving slowly at the cap — the last 10 epochs move 75.4-76.7, so more
epochs would buy little but the curve had not flattened. Checkpoint (86.9 MB) written to
/blue/ewhite/b.weinstein/src/BOEM/output/usgs_hier/best_checkpoint.pth at 02:59, consistent with
the epoch-96 best. Log /home/b.weinstein/logs/classification_hier_BOEM39614374.out;
.err is warnings only (the amp FutureWarning, DeiT registry overwrites, a DtypeWarning on the
annotation-CSV concat) — no errors.
**Comparable to the flat CropModel it shares a val split with:** a3dc30a0 val accuracy 0.756
vs Species@1 0.767 here, on the identical 3,621-crop val set. The hierarchy buys ~1 point at
species and adds usable genus/family heads; it does not change the species-level picture.
Train set is much larger because of ancestor extension (176,338 rows vs 54,765) — the extra
121,808 rows carry only family/genus labels.
**The turtle fix landed as designed** (this was the open verification): the log's first lines are
`[hierarchical] added synthetic taxonomy entry for 'Chelonioidea sp' (species_id=67)` and
`Hierarchy sizes: species=68 genus=49 family=18`, i.e. +1 at every level.
**Turtle species-head accuracy is NOT answerable from this log** — `USGS_hierarchical.py` prints
only aggregate Species@1/@5, Genus@1, Family@1 per epoch, no per-class breakdown and no confusion
matrix. Getting it needs a separate eval pass over `best_checkpoint.pth` against
`buffer_30/a3dc30a085f5442393736ecd96b564c5/usgs_val_split.csv`.
Earlier snapshot (kept for the trajectory): at 2026-08-19 14:00, 63 epochs done,
val Species@1 69.21, Species@5 90.42, Genus@1 74.59, Family@1 83.84 — so the last 37 epochs
added ~7.5 points at species.
Next: (1) run a per-class eval over `output/usgs_hier/best_checkpoint.pth` to read turtle
species-head accuracy — family/genus heads are meaningless for this synthetic class since it
isn't a real taxon, only the species-level "is it a turtle" signal is informative; (2) decide
whether the hierarchical model is worth adopting at all given it matches the flat CropModel at
species level — its value would be the genus/family heads for the n<30 large-whale classes that
39696287 showed cannot support species-level prediction; (3) the pipeline DOES already consume an H-CAST model
(`src/pipeline.py:494-506` -> `active_learning.generate_pool_predictions` -> `hierarchical.classify_dataframe`),
but `boem_conf/hierarchical/hierarchical.yaml` still points at the Dec-2025
`output/usgs_hvit_c2f_b128/best_checkpoint.pth` (37/30/14 heads, epoch 278), NOT this run.
Swapping in `output/usgs_hier/best_checkpoint.pth` (68/49/18) ALSO requires a new `label_csv`:
`output/species.csv` has exactly 37 species rows matching the old head, so against a 68-wide head
every index >= 37 falls through `species_numeric_to_label.get()` to a literal `species_<int>`
string. Build the CSV from `scripts/taxonomy_hier.py` ordering (sorted-alphabetical 0-based over
the species present in the split) plus `Chelonioidea sp` appended at species_id=67.

## 39696287 — 2026-08-19 — confusion matrix on the NEAQ-free classifier (a3dc30a0)
Why: 56e8585 had Megaptera novaeangliae at precision 0.02 (82 predicted, 2 right) acting as
the sink for Delphinus delphis (recall 0.15). Standing hypothesis was that NEAQ — boat/
variable-distance imagery mixed into fixed-altitude aerial surveys — gave the classifier crop
scale as a shortcut for species identity. `scripts/USGS_classification.py` now drops `NEAQ_*`
crop CSVs (`UBFAI_CROPS_EXCLUDE_NEAQ_PREFIX`, committed in 5080197), so a3dc30a0/39613239 is
the first NEAQ-free checkpoint and tests it directly. Parameterised
`scripts/classifier_confusion.py` to take a comet id and added crop-pixel-size, NEAQ-source,
and cetacean blocks; added `submit_classifier_confusion.sh`.

**The hypothesis is right, and the old number was almost entirely a NEAQ artifact.**
Re-reading the 56e8585 predictions CSV by source (NEAQ_* basename vs not):
  - 70 of 3,610 val crops were NEAQ, at **7.1% accuracy** vs 77.3% for BOEM crops.
  - Delphinus recall splits **0/48 NEAQ vs 9/11 BOEM (0.82)**. The headline 0.15 was one
    parent image, NEAQ_20220812_00005, contributing 48 of the 59 val crops.
  - 46 of the 82 Megaptera predictions were NEAQ Delphinus; 55 of 82 were NEAQ crops of
    some class. Strip NEAQ and the "Megaptera sink" is two-thirds gone by construction.

**Direction of the scale effect is the opposite of how it was stated.** NEAQ dolphins are not
small boxes read as big whales — they are *large* crops: median 379px (266-584) against 158px
for BOEM dolphins, sitting squarely inside the BOEM large-whale band (median 540px). Close-range
imagery makes a dolphin fill a whale-sized box, and the classifier believed the size.

**On a3dc30a0 the whale/dolphin confusion is gone.** Megaptera predictions 82 -> **8**, and not
one cetacean row lands there except 2 genuine Balaenoptera acutorostrata (whale->whale).
Delphinus recall 0.15 -> **0.50** (5/10) and its errors are now Tursiops x4 + Fratercula x1 —
dolphin->dolphin, the confusion you would expect from a working model. Overall val accuracy is
unchanged (0.759 -> 0.756), so the exclusion cost nothing.
Crop size still tracks the real size ordering on BOEM-only data (median px): Delphinus 153,
Halichoerus 176, Phoca 176, Stenella 202, Tursiops 218 | Balaenoptera acutorostrata 383,
Megaptera 507, B. physalus 543. Size remains a legitimate cue once ground resolution is constant.

**What we traded for it.** NEAQ was carrying the large whales, and they have now collapsed to
sample sizes that cannot support a class: Megaptera train crops 29 -> 11 and **val recall 0.00**
(6 crops, top confusion Larus delawarensis x4); B. acutorostrata 0.167, B. physalus 0.200;
**Eubalaena glacialis dropped out entirely** (70 -> 69 classes, offset by the new Chelonioidea sp,
so the count reads 70 either way). Tursiops truncatus — still the configured
`active_learning.target_label` — is 0.296 recall, no better than before. So we removed a false
Megaptera *sink*, we did not gain whale detection: the model now has essentially no large-whale
capability rather than a wrong one. That is the better failure mode for survey counts, but any
right-whale/humpback work needs NEAQ back under scale-aware handling (its own model, or a
GSD-normalised crop) rather than mixed in raw.
Unrelated but unchanged from 56e8585: confidence is still not a filter (accuracy 0.756 at
score>=0, 0.831 at score>=0.99), and the missing background/FalsePositive class is untouched.
Outputs: /blue/ewhite/b.weinstein/BOEM/classifier_confusion_a3dc30a0_matrix.csv and
_predictions.csv; log /home/b.weinstein/logs/cls_confusion_39696287.out.
Next: (1) decide whether to keep declaring large-whale classes at n<30 or fold them into a
`Cetacea sp` coarse class; (2) if NEAQ returns, normalise crops by ground sample distance before
pooling; (3) the 202607 survey recount still blocks on the background class, not on this.

## 39700519 + 39700677-39700760 — 2026-08-19 — cache-only re-predict of JPG_202607* on a3dc30a0
Why: swap the classification checkpoint to the NEAQ-free a3dc30a0 (see 39696287) and refresh
the 20-flight July survey's species labels without re-running detection over 533,876 frames.
`src/pipeline.py:436` already supports exactly this: it reuses `<image_dir>/.prediction_cache`
when `detection_checkpoint.txt` matches `detection_model.checkpoint` **and** `min_score.txt`
<= `predict.min_score`, then narrows the pool to images that previously had a detection
>= min_score. All 20 July caches qualify (min_score 0.3, ckpt a09c6933/epoch16), unlike the 8
untrustworthy `imagery/` caches from 2026-08-04 that lack `min_score.txt`.
**Work saved: 19,080 of 533,876 images (3.57%)** carry a >=0.30 detection — 41,694 boxes.
Per-flight spread is wide: JPG_20260712_113200 is 21.1% of frames, JPG_20260712_145900 is 0.48%.

Config: `boem_conf/classification_model/finetune.yaml` checkpoint 56e8585 -> a3dc30a0. Both are
70 classes so nothing downstream changes shape, but the class LIST differs (Eubalaena out,
Chelonioidea sp in). `detection_model.checkpoint` deliberately untouched — changing it would
invalidate every cache and force the full 190 GPU-h pass again.
Added `--extra "<hydra overrides>"` to `submit_all_flights.sh` plus a CACHE-ONLY RE-PREDICT
block in its header documenting the invocation. Overrides used:
`active_learning.n_images=0 active_testing.n_images=0 human_review.n=0 flythrough_video.enabled=False`
  - The three n=0 gates leave train/validation/review image lists empty, so all three
    `annotator.upload()` calls are skipped. Without them this run would have pushed ~2,600 new
    tasks into the shared Label Studio project (100 train + 1 val + 30 review per flight).
  - flythrough off because `generate_flythrough` globs the WHOLE flight and decodes every 8th
    frame — a full-flight pass, which is the cost we are avoiding. The 20 existing videos are
    unchanged; regenerate later from the refreshed cache (the script falls back to reading
    `.prediction_cache/pool_predictions.csv` when `predictions=None`).

**Backed up all 20 pre-existing caches first** to
/blue/ewhite/b.weinstein/BOEM/prediction_cache_backup_56e8585_20260819/ (41,694 boxes, 6.2 MB),
because the run REWRITES `pool_predictions.csv` in place and the survey's published species
counts came from the 56e8585 version.

Smoke test 39700519 (JPG_20260712_092800, the smallest) COMPLETED in **1m51s**:
"running detection+classification on 58 images ... instead of 13911", then 58 images / 62 boxes
out — same box count as the cached run, detection scores identical to 1.5e-3 (GPU nondeterminism),
and "No images to upload" for all three instances. 16 of 60 matched boxes changed species label;
3 became the new Chelonioidea sp.
Then fanned out all 20 flights on hpg-b200.
**Scheduling note worth reusing:** submitted at `--time 08:00:00` they all sat PENDING behind a
274-deep queue with start estimates 10-14 h out. These jobs need ~2-30 min, and an 8 h request
makes them effectively unbackfillable. `scontrol update jobid=<id> TimeLimit=02:00:00` on the
pending jobs (no cancel needed) pulled the earliest estimate from 23:52 to 15:45. Submit
cache-only reruns at a realistic wall time.
Result: **all 21 jobs COMPLETED, rc=0, no errors.** Every flight logged "Using prediction cache"
and "No images to upload" x3. **6.2 GPU-h against the 189.6 GPU-h the original pass cost** —
e.g. JPG_20260710_163500 21.81h -> 0.19h. The two big flights (JPG_20260712_100400,
JPG_20260712_113200) took 1h45m each, so the 2h TimeLimit was closer than it looked; use 3h next
time. Box totals are stable: 41,694 -> 41,739 (+0.1%), 19,080 -> 19,008 images, drift only from
GPU nondeterminism at the score boundary.

**The whale fix transfers to real survey data**, independent of the val split:
  Megaptera novaeangliae  92 -> 8    Balaenoptera acutorostrata  29 -> 5
  Eubalaena glacialis      5 -> 0    Balaenoptera physalus        4 -> 1
**But the foam did not go away, it MOVED — and the prediction was wrong about which classes.**
  Somateria mollissima 20,979 -> 1,820   |   Chelonioidea sp        0 -> 9,699
  Gavia immer           2,635 -> 8,089   |   Clangula hyemalis  7,717 -> 10,719
  Tursiops truncatus      323 -> 1,379   |   Fratercula arctica 2,336 -> 1,978
The new sea turtle class became the single largest foam sink on its first contact with survey
imagery. "Cold-water share 79.4% -> 39.2%" is therefore a MIRAGE: the foam relabelled onto
turtles and loons, which are not on the cold-water list. Do not quote that number as progress.
The det>=0.65 screen is unchanged as expected (611 -> 608 boxes), confirming the swap did not
touch the detector — but Chelonioidea sp 58 and Tursiops truncatus 43 now appear inside the
screened set and need eyeballing before either is believed.
**Tursiops truncatus 4x-ing (323 -> 1,379) matters operationally**: it is
`active_learning.target_labels`, so it drives image selection. Its val recall is 0.296 and it was
NOT improved by the NEAQ removal.

**Latent cache trap found:** every July `min_score.txt` records 0.3, but the actual minimum score
in the CSVs is **0.4000** — `detection.py:load()` still does not override score_thresh from
`predict.min_score` (open item from the 08-14 entry). The cache-validity check
(`cached_min_score <= predict.min_score`) would therefore ACCEPT the cache for a future run at
min_score 0.30-0.40 while silently missing every box in that band. Fix score_thresh, or write the
observed floor rather than the configured one into min_score.txt.
Next: (1) pull crops for the Chelonioidea sp and Tursiops boxes that clear det>=0.65 and look at
them before anything is recounted; (2) the background/FalsePositive class remains THE blocker —
this swap redistributed the foam without reducing it; (3) fix the score_thresh/min_score.txt
mismatch before the next cached rerun.

## (no job) — 2026-08-20 — predict.min_score 0.30 -> 0.70, and score_thresh actually enforced
**Decision:** the pipeline detection gate is 0.70 everywhere. Three config gates moved together
(`predict.min_score`, `active_learning.min_detection_score`, `human_review.min_detection_score`),
as their comments have always required.

**The 0.30 gate was never real.** The July caches were stamped `min_score.txt` = 0.3 but their
actual floor was 0.4000, because `detection.load()` never overrode the `score_thresh` baked into
the checkpoint at construction (`scripts/USGS_backbone.py:154`, 0.4). So the "all detection
scores" species table everyone was reading WAS already the >=0.30 table — filtering it at the
configured gate is a no-op. Fixed two ways:
  - `detection.load(score_thresh=...)` now sets BOTH `model.score_thresh` (the live torchvision
    module, which is what predict_tile reads) and `config["score_thresh"]` (the inert dict copy).
    Setting only the latter — the previous behaviour everywhere but the sweep scripts — changes
    nothing about what the detector emits.
  - `min_score.txt` now records `max(predict.min_score, model.score_thresh)`, the floor that
    ACTUALLY applied, so a cache can never again claim coverage of a band it does not hold.

**Why 0.70 and not 0.30.** The 0.30 value came from the Feb holdout sweep (39385379), where 79.1%
of boxes already sat above 0.85; the July survey has 1.4% there. On survey imagery, detection
score is the ONLY signal that separates animals from foam:
  - 19 Morus bassanus crops: the 8 real birds are exactly the 8 with score >= 0.65 (0.664-0.831),
    all 11 foam crops fall in 0.400-0.573. No exceptions either way.
  - Share clearing 0.65 (a3dc30a0, 20 flights): Thalasseus maximus 86.7% (median det 0.811) vs
    Clangula 0.3%, Chelonioidea 0.6%, Gavia 0.5%, Somateria 2.0%, Fratercula 0.2%.
  - Classifier confidence cannot substitute: median `cropmodel_score` on the foam sinks is
    Chelonioidea 0.990, Clangula 0.994, Somateria 0.988, Tursiops 0.983. The classifier is
    maximally confident ON THE FOAM.
Pool cost on the 20-flight survey: 41,739 boxes (>=0.40) -> 7,405 (>=0.50) -> 608 (>=0.65) ->
388 (>=0.70). 98.5% of the old pool was below the only threshold that discriminates.

**The species table at the gate** (det>=0.65, 56e8585 -> a3dc30a0, 611 -> 608 boxes) is a
different document from the ungated one. Thalasseus maximus 179->170 dominates; Somateria
102->36; Larus argentatus 29->10. The classifier swap's whale fix holds AT the gate — Megaptera
10->0 and Phoca vitulina 5->0 confident boxes — but the turtle sink appears there too:
Chelonioidea 0->58 and Tursiops 25->43 inside the screened set. Those two still need eyeballing.

**Existing caches stay valid, no re-predict needed.** Every July cache holds everything >=0.4,
a superset of >=0.7, and `cached_min_score (0.3) <= 0.70` passes the validity check. `report.py`
reads `config.predict.min_score` directly, so the species table regenerates at 0.70 for free.

**Deliberately NOT moved** (asked and confirmed): `submit_seals_array.sh`,
`submit_seals_missing_metadata.sh` and `submit_neaq.sh` keep `predict.min_score=0.5`. Seal
detections peak ~0.89 even on obvious animals; NEAQ is retired and its imagery is already a
0.5-screened subset, so a higher floor there recovers nothing. `scripts/collect_screened_images.py`
MIN_SCORE stays 0.5. Recorded in the `predict.min_score` comment so it does not read as an
oversight later.
Also: `active_testing.min_score` is a DEAD key (nothing reads it); set to 0.70 and labelled.
`src/active_learning.py` fallback defaults (0.85/0.1/0.3) realigned to 0.70 — config still wins.

Tests: 43 passed, 1 skipped. `tests/test_detection.py::test_detection_preprocess_and_train` fails
on HEAD too (DeepForest API drift: `deepforest.__init__() got an unexpected keyword 'label_dict'`
at `src/detection.py:275`) — pre-existing, unrelated, still open.

## (no job) — 2026-08-20 — human_review.review_high 0.6 -> 0.99: the review band was inert
Why: the classifier is visibly overconfident (predict.min_score notes median cropmodel_score
0.98-0.99 on foam), so the question was whether `review_high` carries any information at all,
or whether the review queue is fed by accident. Answered off the existing a3dc30a0 val
predictions — /blue/ewhite/b.weinstein/BOEM/classifier_confusion_a3dc30a0_predictions.csv,
n=3,695 — plus the 20 cached July pools. No new GPU time.

**Two questions, opposite answers, and the confusion between them is what broke the band.**
LEVEL is hopeless: mean confidence 0.965 against 0.756 accuracy, **ECE 0.209**, and 80.9% of
all val crops score >= 0.99. `cropmodel_score` is not a probability and no threshold should be
read as one. RESOLUTION is fine: **AUROC(score -> correct) = 0.789**, and accuracy rises
monotonically straight through the spike — 0.51 in [0.99,0.995), 0.56, 0.66, 0.77, and 0.90 at
exactly 1.0. So the score ranks honestly; it just does all its ranking inside the top 1%.

**That is why 0.6 was doing nothing.** Below 0.99 accuracy is essentially FLAT: 0.36 at
0.6-0.7, 0.48 at 0.8-0.9, 0.49 at 0.95-0.99. Every cut in that range was separating
equally-wrong boxes from each other. The old [0.3, 0.6] band held 2.6% of val crops and
**7.0% of all errors**; on the 20-flight July survey it held **28 boxes / 27 images** against
`human_review.n=30`. The queue was not prioritised, it was whatever happened to land there —
the same failure shape as the always-empty `confident_predictions` bug from 2026-08-12, one
level up.

**0.99 chosen** as the first cut that lands where the error mass is: 19.1% of val crops,
**43.9% of all errors**, 56.1% of the reviewed batch actually wrong (**2.30x** the 24.4% base
rate). Survey pool goes 28 -> **185 boxes / 173 images** reviewed, auto-annotate 357 -> 188
boxes, and the val accuracy of the auto-annotated set (the errors we silently accept) rises
**0.767 -> 0.831**. Nothing above 0.99 buys much more: 0.999 gets 59.4% of errors but reviews
27.4% of the pool at a *lower* 2.17x lift, and would leave only 131 survey images auto-annotated.

**0.99 is a percentile, not a probability** — roughly p80 of this checkpoint's score
distribution (p10=0.892, p20=0.993, p30=0.9995, p50=1.000). It is as checkpoint-specific as
`predict.min_score` is detector-specific and must be re-derived when the classifier moves.
`scripts/classifier_confusion.py` now prints the reliability table, ECE, AUROC and the full
`review_high` sweep so the re-derivation is one job, not one analysis.

**The band widening also turns the queue sort back on.** `src/pipeline.py:654` orders the band
by DETECTION score descending, which is right (detection score is the only foam discriminator)
but was near-inert against a 27-image pool. Against 173 images it selects: queue detection
scores go 0.700-0.883 -> **0.868-0.935**, and the queue shifts off the mass classes toward
Thalasseus maximus 6 -> 15 and Morus bassanus 1 -> 5 — the two classes eyeballing has confirmed
behave like real animals at the gate.

**Caveat, and it is a real one.** The val split is all real annotated animals — there is no foam
and no background class in it, so every number above is measured on a cleaner distribution than
the survey. The 173-image figure comes from the actual survey pools and is the one to trust for
queue sizing; the 43.9% error recall is the val-set optimistic case. The missing background/
FalsePositive class remains the blocker underneath all of this (see 39696287).

Changed: `boem_conf/boem_config.yaml` human_review.review_high 0.6 -> 0.99 with the derivation
inline; `src/active_learning.py` `human_review()` default 0.6 -> 0.99 and docstring warning that
the value is a percentile, not a probability; `scripts/classifier_confusion.py` calibration +
threshold-sweep block replacing the old 5-line "does high confidence mean correct?" print.
`review_low` stays 0.3 — it is nearly inert (1 of 3,695 val crops below it, survey pool min
0.316) and the spurious-detection job it was written for now belongs to `min_detection_score`.
Next: (1) re-derive on the next classifier checkpoint before trusting the queue; (2) the honest
fix for the level miscalibration is temperature scaling on the val split, which would make
`review_high` mean something across checkpoints instead of needing a re-sweep each time.

## (no job) — 2026-08-20 — taxonomic rollup in the report: agreement decides rank
Why: reporting should inherit the hierarchical classification and roll statistics up the
taxonomic tree, rather than reporting the flat CropModel species as if it were unopposed.
**The rollup did NOT previously exist anywhere**, contrary to standing assumption. What existed
was the *predicate*, in two places, neither of which assigns a label:
  - `active_learning.crop_hcast_supported_match_or_genus_consistent` (src/active_learning.py:22) —
    "species match OR genus consistent", used as a VETO on Label Studio image selection under
    `ensemble_target_mode: match_or_genus_consistent`. Currently inert (mode defaults to `crop_only`).
  - `PipelineEvaluation._evaluate_hierarchical_metrics` (src/pipeline_evaluation.py:112) — computes
    `species_agreement` / `genus_agreement` as Comet METRICS. Numbers, not labels.
`src/report.py` set `predicted_label = cropmodel_label` unconditionally and carried hcast_* as
inert extra columns. Confirmed never implemented: `git log -S consensus_label / genus_rollup /
reported_label` all return nothing.

Implemented in `src/hierarchical.py` (`resolve_row_rank`, `resolve_taxonomic_rank`,
`summarize_taxonomic_rollup`, `load_species_to_ranks`), wired into `src/report.py`, config block
`report.taxonomic_rollup`, 10 tests in `tests/test_taxonomic_rollup.py` (all pass).
Ladder: species agree -> species | species differ, genus same -> genus | genus differs, family
same -> family | no agreement -> `unresolved` (crop label retained for traceability, rank marks it
uncountable). Missing/null hcast columns pass through at species rank, so `hierarchical.checkpoint:
null` runs are byte-identical to before. Joint confidence = min(cropmodel_score, hcast_<rank>_score),
matching `active_learning._row_min_class_confidence`. Optional `min_consensus_score` demotes one
further rank when joint confidence is below the floor (agreement sets how far up we CAN report,
confidence how far up we MUST) — default null, agreement alone.
Outputs: `consensus_label`/`consensus_rank`/`consensus_score` in both observations tables and the
shapefile (`cons_lbl`/`cons_rank`/`cons_scr`), plus a new `taxonomic_summary.csv` per flight.
The existing species-composition figures, maps and sample crops still key off `cropmodel_label` —
deliberately left alone so this change adds a view rather than silently restating published counts.

**The rollup is correct but currently not usable, and the reason is the checkpoint, not the code.**
Measured on the real JPG_20260712_113200 cache at the live det>=0.70 gate (26 boxes):
**4 species (15.4%), 5 genus (19.2%), 17 unresolved (65.4%)**. Across all 12,771 cached boxes it is
worse: 11 species, 20 genus, 95 family, **12,645 unresolved**. Crop-vs-hcast species agreement on
this flight is 0.09%. That is the wired H-CAST being a Dec-2025 37-class bird model whose
vocabulary cannot express the CropModel's top classes (no turtle, no Tursiops) — see the
39614374 entry for the checkpoint/label_csv swap this blocks on. Do NOT publish rollup counts to
collaborators until the 68-class checkpoint and a matching 68-row label CSV are wired in; until
then `unresolved` is measuring vocabulary mismatch, not genuine taxonomic uncertainty.
Next: (1) build the 68-row label CSV and swap `boem_conf/hierarchical/hierarchical.yaml`, then
re-measure the rank split on the same flight — it is the cleanest single test of whether the new
hierarchical model is worth adopting; (2) decide whether `unresolved` rows should be dropped from
abundance counts or reported as an explicit uncertainty bucket; (3) if the rollup is adopted as
the headline statistic, switch the PDF species-composition figures onto `consensus_label`.

## (no job) — 2026-08-20 — decision: 0.70 is the reporting band, 0.40 is disposable
Ben's call, explicit: **everything below det 0.70 is garbage detections; he does not want the
0.40 band preserved.** This retires the cache-overwrite objection raised against re-running the
July flights — rewriting `.prediction_cache/pool_predictions.csv` at a 0.70 floor is accepted,
and the 20 caches do NOT need backing up first (unlike the 2026-08-19 run, which backed up the
56e8585 versions to prediction_cache_backup_56e8585_20260819/).
What the rerun will actually produce, measured off the current caches:
  41,739 boxes / 19,008 images (0.40 floor)  ->  **388 boxes / 360 images** at 0.70.
Per-flight the survivors are thin — JPG_20260713_141400 keeps 2 boxes, JPG_20260711_102100 keeps 3;
the largest is JPG_20260712_100400 at 94. The taxonomic-agreement measurement will therefore rest
on 388 boxes survey-wide, which is the live reporting gate but a thin basis for characterising
model agreement. Noted, not objected to.

### Staged and waiting on the crop-geometry sweep
Decided to wait for `BOEM_hcast_expand` before touching anything. Two jobs are queued —
**39811016 (start 14:22:48) and 39811745 (start 14:27:09)**, both hpg-b200, both TimeLimit 4h.
**CORRECTION — they were NOT duplicates**, as first assumed from the identical job name and
script. They swept different checkpoints via the `BOEM_HCAST_*` env overrides and wrote to
different files: 39811016 -> the new usgs_hier model (`output/usgs_hier/expand_sweep.csv`),
39811745 -> the old usgs_hvit_c2f_b128 (`output/usgs_hvit_c2f_b128/expand_sweep.csv`). No race,
nothing to cancel. Both COMPLETED rc=0 in 6:08 and 4:17. Neither is mine (a concurrent session
owns them, along with in-flight edits to `src/hierarchical.py`).

Readiness audit for the rerun — everything except geometry is done:
  - `output/usgs_hier/best_checkpoint.pth` (68/49/18) loads clean, `nb_classes [68, 49, 18]`.
  - `output/usgs_hier/species.csv` built 13:21: 68 species / 49 genus / 18 family, index 0-67,
    with explicit `genus_index`/`family_index` columns — which `load_hcast_model` now honours
    (src/hierarchical.py:233,249), closing the positional-ordering risk flagged earlier.
    Verified: species 67 -> `Chelonioidea sp`, genus 48 -> `Chelonioidea`, family 17 -> `Chelonioidea`.
  - Crop geometry is plumbed config -> inference (`src/pipeline.py:250-252` and `506-508` pass
    expand/square/eval_crop_ratio into load/classify).
  - Rollup + `taxonomic_summary.csv` landed, 10/10 tests pass.
  - **NOT done:** `boem_conf/hierarchical/hierarchical.yaml` still points at the Dec-2025
    `usgs_hvit_c2f_b128` checkpoint and the old 37-row `output/species.csv`, and still carries the
    placeholder geometry `expand: 0 / square: false / eval_crop_ratio: null`. The sweep decides
    what those three become; the yaml's own comment predicts 30/true/0.875 but that is a reading
    of the training script, not a measurement.
Next, once the sweep lands: (1) read `output/usgs_hier/expand_sweep.csv` and take the winning
geometry; (2) swap checkpoint + label_csv + the three geometry keys in hierarchical.yaml;
(3) resubmit the 20 July flights with the documented cache-only invocation from
`submit_all_flights.sh`'s header (`--b200 --extra "active_learning.n_images=0
active_testing.n_images=0 human_review.n=0 flythrough_video.enabled=False"`), at a REALISTIC wall
time — the 08-19 lesson was that an 8h request makes these unbackfillable; ~360 images total means
minutes per flight, so 1h is generous; (4) read the rank split out of `taxonomic_summary.csv`.

## 39811016 + 39811745 — 2026-08-20 — H-CAST crop-geometry sweep — both COMPLETED
Why: H-CAST re-crops from the parent image at inference, and the geometry it uses must match the
geometry it trained on. `boem_conf/hierarchical/hierarchical.yaml` carried the historical no-op
(`expand 0 / square false / eval_crop_ratio null`) inherited from the Dec-2025 checkpoint, which
was trained outside this repo on pre-made crops. `scripts/USGS_hierarchical.py` instead pads by
--expand-pixels 30, squares, and validates at ratio 0.875. Sweep measures rather than assumes.

**New checkpoint (39811016, output/usgs_hier, 68/49/18, 3,621 val crops) — Species@1:**
    expand=30 square=true  ratio=0.875 -> **76.72**  (training-matched; best)
    expand=30 square=false ratio=0.875 ->   76.53
    expand=15 square=true  ratio=0.875 ->   75.75
    expand=60 square=true  ratio=0.875 ->   69.21   (over-expansion hurts)
    expand=0  square=true  ratio=0.875 ->   68.79
    expand=0  square=false ratio=null  -> **62.66**  (the OLD pipeline default)
76.72 reproduces 39614374's best-epoch val exactly, which is the proof that the inference path now
matches the training path. **The geometry alone is worth +14.06 points of species accuracy**, and
it was being silently thrown away on every pipeline run to date.

**Old checkpoint (39811745, usgs_hvit_c2f_b128, 37/30/14), restricted to its own 37 species
(3,621 -> 2,717 crops) — Species@1:**
    expand=60 square=false ratio=0.875 -> 29.96  (its best)
    expand=30 square=true  ratio=0.875 -> 29.74
    expand=0  square=false ratio=null  -> **12.81**  <- what the pipeline has actually been running
**This is the real explanation for the 0.09% crop-vs-hcast agreement on survey data**, not
vocabulary mismatch alone as previously concluded: the deployed hierarchical model was running at
12.8% species accuracy on an easier 37-class subset. Its genus head is worse than useless
(Genus@1 ~12-13% at every geometry, BELOW its own species accuracy — a real ordering anomaly worth
a look if that checkpoint is ever revived; it is not worth reviving on these numbers).
Best-vs-best across checkpoints is 76.72 vs 29.96 on different denominators (3,621 vs 2,717 crops,
68 vs 32 classes present), so not a clean head-to-head — but the direction is not in doubt.

Acted on it: swapped `boem_conf/hierarchical/hierarchical.yaml` to the new checkpoint +
`output/usgs_hier/species.csv` + expand 30 / square true / eval_crop_ratio 0.875, with the sweep
table inlined as the justification. Backup of the prior yaml is in the session scratchpad.
Next: measure the rollup rank split on the real 388-box 0.70 reporting set before submitting the
20 flights — running now off the cached boxes, no detection needed.

## 39811016 + 39811745 — 2026-08-20 — H-CAST inference crop geometry sweep — both COMPLETED
Why: user observed that H-CAST sees a different crop than the CropModel (its own square resize, no
`classification_model.expand` buffer), so it is not a second opinion on identical input, and predicted
a domain problem because the configured checkpoint was trained without expansion. Confirmed and worse
than stated: `output/usgs_hvit_c2f_b128/best_checkpoint.pth` (Dec 2025) has
`args.data_path=/scratch/user/u.sp270400/USGS_crops` — it was NOT trained by this repo at all, but by
an external H-CAST run on pre-made crop dirs, with no `expand_pixels` in its namespace.
**Three mismatches, not one.** Training (`HierarchicalCropDataset`) pads by `--expand-pixels`, squares
the box, then resizes at eval_crop_ratio 0.875. Inference (`InferenceCropDataset`) took the raw box,
did not square, and squashed straight to 224x224. The squaring and resize gaps were independent of
expand and applied to both checkpoints.
**The "train a new model first" step was already done**: job 39614374 (finished 08-20 02:59) trained
`output/usgs_hier/best_checkpoint.pth` via `submit_USGS_hierarchical.sh` with `--expand-pixels 30`
(the script's default since 5772341). No new training was needed to run the test.

Code: `src/hierarchical.py` — `expand_bbox_to_square` is now the canonical helper (was duplicated in
`scripts/USGS_hierarchical.py`, which now imports it so the two paths cannot drift again);
`InferenceCropDataset` takes `expand_pixels`/`square`; `_default_transform` takes `eval_crop_ratio`;
`classify_dataframe` plumbs all three. Config `expand`/`square`/`eval_crop_ratio` in
`boem_conf/hierarchical/hierarchical.yaml` wired at `src/pipeline.py:250,506`.
**Two prerequisite bugs fixed** (both would have silently corrupted the test): (1) `load_hcast_model`
assigned genus/family ids by CSV row order, which only accidentally matches a trained head — it now
honours explicit `genus_index`/`family_index` columns; (2) the new checkpoint had no usable label_csv
(`output/species.csv` is the 37-species one), so `scripts/build_hcast_label_csv.py` replays the
training vocabulary construction — produced 68/49/18, matching the checkpoint's stored `nb_classes`.

**Harness validated:** the `expand 30 / square / 0.875` row reproduces 76.72 Species@1, exactly the
best logged by training job 39614374 on the same 3,621-crop val split.

**Result 1 (39811016, new checkpoint) — expansion is the whole story, squaring is a wash at 30.**
Square effect, each expand at its own best resize: expand 0 **+6.13** (62.66->68.79), expand 15 +1.63,
expand 30 **+0.19** (76.53->76.72, ~7 crops = noise), expand 60 **-2.70** (71.91->69.21, squaring HURTS).
Mechanism is the box aspect distribution: raw boxes are elongated (median long/short 1.46, 45% >1.5),
but after +30 px per side the median aspect falls to 1.13 and only 2.5% exceed 1.5 — expansion has
already done the squaring's job, and by 60 the square only adds background.
Path from the pipeline's current setting to best: (0,noSq,squash) 62.66 -> +expand30 75.45 (**+12.79**)
-> +square 75.67 (+0.22) -> +0.875 crop 76.72 (+1.05). **+14.06 points from inference geometry alone,
no retraining.** Worst cell is (0, noSq, 0.875) at 49.07: centre-cropping a NON-square crop discards the
long axis. If no-square is ever used it must pair with the squash resize, never the centre-crop.

**Result 2 (39811745, the checkpoint the pipeline actually uses), restricted to the 32 of its 37 species
present in the split (2,717 crops): 12.81% at the current pipeline setting, 29.96% at its own best
geometry — vs 76.72% for the new model.** Its optimum is at expand **60**, not 30, consistent with the
user's domain hypothesis: trained on pre-made crops carrying more context than raw detection boxes.
**Not a label-mapping artifact** (checked, because the claim is strong): a 500-crop confusion dump at
its best geometry shows Morus bassanus recall 90.9%, Larus argentatus 71.4%, Fulmarus glacialis 63.6%,
and taxonomically coherent errors (Sterna forsteri<->S. hirundo, Larus argentatus->L. delawarensis,
Rissa tridactyla->Chroicocephalus philadelphia). A scrambled mapping would sit near chance at every
geometry and could not produce congener confusions. The model is genuinely weak, not mis-indexed;
failures are cross-order (Fratercula->Calonectris 14%, Gavia stellata->Morus 20%, Oceanites 12%),
which is the signature of domain mismatch rather than fine-grained difficulty.
Caveat: that checkpoint's Genus@1 is pinned near 12% across all 11 configs — the row-order genus
indexing described above. Only its species column is trustworthy.

`scripts/USGS_hierarchical.py` gained `--square/--no-square` and `--eval-crop-ratio` so the no-square
variant is trainable, defaults unchanged. On the +0.19 measured at expand 30, that run is NOT worth
queuing.
**NOT changed, deliberately:** `boem_conf/hierarchical/hierarchical.yaml` still points at the Dec-2025
checkpoint with `expand: 0 / square: false`, reproducing today's behaviour exactly.
Next: decide whether to swap to `output/usgs_hier/best_checkpoint.pth` +
`output/usgs_hier/species.csv` + `expand: 30 / square: true / eval_crop_ratio: 0.875`. That is a
~64-point species-accuracy change to a live pipeline stage whose `hcast_species` output feeds the
`match_or_genus_consistent` and `model-disagreement` strategies in `src/active_learning.py`, so it
needs an explicit call. Sweep CSVs: `output/usgs_hier/expand_sweep.csv`,
`output/usgs_hvit_c2f_b128/expand_sweep.csv`. Logs: /home/b.weinstein/logs/hcast_expand_sweep_3981{1016,1745}.out

## (no job) — 2026-08-20 — rollup measured on the real 388-box 0.70 reporting set
Why: settle whether the new hierarchical model makes the taxonomic rollup usable BEFORE spending
20 flight jobs. Run read-only off the 20 cached `pool_predictions.csv` at score>=0.70, re-running
only H-CAST over the existing boxes — no detection, no cache writes. (CPU login node, ~25 min;
would be ~1 min on a B200 but not worth queueing.)

                          BEFORE (37-class, old geom)      AFTER (68-class, expand30/sq/0.875)
  species                      17   ( 4.4%)                     **194  (50.0%)**
  genus                        70   (18.0%)                       17  ( 4.4%)
  family                       11   ( 2.8%)                       49  (12.6%)
  unresolved                  290   (74.7%)                      128  (33.0%)
  crop-vs-hcast species agreement   0.044                        **0.500**

**Resolved to some rank goes 25.3% -> 67.0%.** Both the checkpoint and the crop geometry changed
together, so this does not attribute the gain between them; the sweep (39811016) says geometry
alone is worth 14 points of species accuracy, so it is a large share of it.
Composition of the resolved set is consistent with everything else we know:
  - `Thalasseus maximus` is 146 of the 194 species-level records — the one class the 08-14
    threshold work found behaves like a real species (86.7% clearing 0.65, median score 0.811).
  - `Laridae` 41 at family level: gulls/terns where the two models pick different larids.
  - **`Tursiops truncatus` is 6 species-level vs 19 unresolved.** The configured
    `active_learning.target_labels` is still the class the hierarchical model least supports —
    consistent with its 0.296 val recall on the flat model. Anything driven off Tursiops
    selection should be read with that in mind.
  - `Chelonioidea sp` is down to 4 species + 1 family, from the 58 that cleared det>=0.65 in the
    08-19 run. The turtle foam sink does not survive contact with a second opinion.
Artifact: scratchpad/rollup_070_new.csv (388 rows, all consensus + hcast columns).

**Open design question, raised and NOT yet answered — do not submit the flights until it is.**
`final_predictions` (what the report is built from, pipeline.py:666-723) is the 0.70 pool PLUS the
existing human train/validation/reviewed annotations, concatenated with sentinel `score = 2.0` and
`cropmodel_score = 2.0`. Those rows carry no hcast columns, so the resolver passes them through at
species rank with the human label — correct behaviour, a human annotation must not be demoted
because a model disagrees. But it means `taxonomic_summary.csv` mixes verified records with model
predictions and averages real confidences against 2.0 sentinels, making `mean_score` meaningless.
Options put to Ben: (a) give human rows their own rank `verified`, ahead of `species`, so the
ordering verified > species > genus > family > unresolved is one honest confidence ladder;
(b) group the summary by the existing `set` column instead; (c) restrict the summary to
`set == "prediction"`. Recommended (a). The measurement above is unaffected — it ran off
pool_predictions only, which contains no human rows.

## 39825931 — 2026-08-20 — logits saved + temperature scaling on a3dc30a0
Why: `review_high=0.99` is a percentile of one checkpoint's score distribution, not a
probability, so it has to be re-swept on every retrain. Temperature scaling (Guo et al. 2017)
is the standard post-hoc fix. Question was whether it also improves the review queue.
`scripts/classifier_confusion.py` now captures raw logits, saves them, fits T, and reports
calibrated-vs-uncalibrated side by side. 1m33s on one B200.

**Logits are now persisted** — `classifier_confusion_a3dc30a0_logits.npy` (3695x70 float32,
row-aligned to `_predictions.csv`, columns in `_classes.txt`). `CropModel.predict_step`
softmaxes before returning, so the script overrides it with a logit-returning version;
recovering logits from the stored softmax afterwards is impossible because float32 saturates
(see below). Every future calibration question is answerable from this file with no GPU.

**T = 5.61.** Far outside the usual 1.5-3 for overconfident nets. ECE **0.209 -> 0.070**,
NLL **3.008 -> 1.147**, accuracy unchanged at 0.7564 (argmax is invariant to T, by
construction). The 0.99 pile-up dissolves: share >= 0.99 goes **0.809 -> 0.076**, and the
median confidence goes 1.0000 -> 0.797. The score finally uses its range.

**But it does not buy a better queue — it costs a little.** AUROC(confidence -> correct)
**0.796 -> 0.770**. Temperature is monotone in the logits but NOT in max-prob (max-prob
depends on the whole logit vector, so crops with different runner-up structure reorder), and
here that reordering is mildly harmful. At matched review budget: uncalibrated 0.99 reviews
19.1% of the pool for **43.9%** of errors at 2.30x lift; T-scaled 0.40 reviews 16.9% for
**38.2%** at 2.26x, T-scaled 0.50 reviews 24.6% for 51.1% at 2.08x. Interpolated to a common
19% budget the calibrated scale is ~2-4 points of error recall WORSE. Confirms the prediction
made when review_high moved to 0.99: temperature fixes LEVEL, not RESOLUTION.

**DECISION: T is NOT deployed to production. `review_high` stays 0.99 on the raw scale.**
Deploying it means patching the softmax inside the vendored `deepforest/model.py:610` (or
wrapping CropModel) for a threshold that would be slightly worse at triage. The portability
argument — a calibrated `review_high` meaning the same thing on the next checkpoint — is real
but untested: T has been fit on exactly one checkpoint, so there is no evidence yet that it is
stable enough to skip the sweep. Revisit after 2-3 checkpoints have a T on record. Until then
**T is a tracked diagnostic**: one number per checkpoint saying how overconfident it is, and
directly comparable across retrains in a way ECE-at-a-threshold is not.

**Margin and entropy are a dead end.** AUROC: max-prob 0.7962, logit margin (top1-top2)
0.7953, negative entropy 0.7968. No winner, differences well inside noise on n=3,695. This
closes the "a different uncertainty score might reorder the queue" idea — on this checkpoint
there is no more triage signal to extract from the logits, whatever function you apply. The
remaining lever is the model, not the score: background class, then the n<30 classes.

**Incidental find: float32 softmax saturation.** 43.2% of val crops come back as *exactly*
1.0 in float32 (float64: 4.5%), collapsing 1,598 crops into one tie group and costing
AUROC 0.7887 -> 0.7962. NOT an operational problem for `review_high=0.99` — the survey pool
is a gentler distribution (only 4.7% of all boxes and 10.6% of gated boxes hit exactly 1.0),
and the tie group is the easy end (accuracy 0.936) whereas review reads the low end. Recorded
because it makes the raw scale unusable ABOVE ~0.999: `review_high=0.9999` cannot be
implemented in float32 no matter what a sweep says.

Changed: `scripts/classifier_confusion.py` — `logit_predict_step` override, `_logits.npy` +
`_classes.txt` outputs, temperature-scaling section (fit, before/after calib table, AUROC
invariance check, calibrated review_high sweep, alternative-score comparison).
Next: (1) record T for each new checkpoint and see whether it is stable before considering
deployment; (2) the level miscalibration itself is a training artifact — label smoothing
0.05-0.1 or early-stopping on val NLL instead of val accuracy would attack the cause and,
unlike T, could actually change resolution.
UNRELATED, found while reading the training config: `fast_dev_run: True` is committed in
`boem_conf/classification_model/finetune.yaml` and flows to the Trainer via
`scripts/USGS_classification.py:385`; no `submit_*.sh` overrides it. a3dc30a0 plainly trained
properly so it was overridden by hand, but as committed the next retrain runs one batch.

## 39831139 — 2026-08-20 — 202607 re-predict on the new H-CAST checkpoint — SUBMITTED
Why: user confirmed the swap after 39811016/39811745 (see above). `boem_conf/hierarchical/hierarchical.yaml`
now points at `output/usgs_hier/best_checkpoint.pth` (job 39614374, 68/49/18, Species@1 76.72) with
`label_csv: output/usgs_hier/species.csv` and the training-matched geometry `expand: 30 / square: true /
eval_crop_ratio: 0.875`. Replaces the Dec-2025 `usgs_hvit_c2f_b128` checkpoint, which measured 12.81%
species accuracy at the geometry the pipeline had been feeding it.
Verified before submitting: hydra resolves all seven keys; `load_hcast_model` returns 68/49/18 with
**zero** unmapped `species_<int>` slots; the wrapper carries geometry 30/True/0.875.
**Ensemble alignment improved a lot**: the new label_csv can name 68 of the CropModel's 70 classes,
vs 32/70 for `output/species.csv`. The 2 it cannot are `Calonectris Puffinus` (222 train rows, a
genus-pair ambiguity label, not a real binomial) and `Columba livia` (13 rows) — neither is a species
leaf in taxonomy.json. Crops the CropModel calls those two will read as "H-CAST does not support" in
`crop_hcast_supported_match_or_genus_consistent` and as disagreement in `model-disagreement`.
Tests: tests/test_active_learning.py + test_taxonomic_rollup.py + test_classification.py, 23 passed.

Invocation: the documented cache-only re-predict from `submit_all_flights.sh`'s header, `--serial --b200`
at `--time 06:00:00`. All 20 `JPG_202607*` caches are VALID (`min_score.txt` 0.3 <= `predict.min_score`
0.7, `detection_checkpoint.txt` == the configured epoch16-val_cls0.0163.ckpt), so this is a genuine
cache-only pass. **Two different pool numbers, both real — do not confuse them.** The cached
`pool_predictions.csv` files hold 19,008 images across the 20 flights (those with a prior detection
>= the cached 0.3). The pipeline then filters at `predict.min_score` 0.7, and THAT pool is tiny: the
first two flights logged `Pool: 9 images with >=0.7 detections`. So the "~360 images total" in the
39700519 entry is the post-0.7 count and was correct; an earlier revision of this entry claimed it was
wrong, which it was not. 6 h was requested off the conservative 19,008-image reading (4.11 img/s ->
~1.3 h); observed throughput is far faster — 2 flights in ~3 min, so ~30 min for all 20. Future
cache-only 202607 passes can safely request ~1 h, which backfills much sooner (the 08-19 lesson was
that 8 h requests sit unscheduled).
Deliberately NOT included: the 8 non-202607 flights whose caches lack `min_score.txt` AND carry a stale
detector (`a1c5649615...pl`), which would force a full detection re-run over ~144,600 images and change
detections, not just species labels. Also excluded: 3 dirs with 0 jpgs. The 4 other valid-cache flights
(JPG_20241219_120500/131500/150200, min_score 0.5) were offered and not selected.
Logs: /home/b.weinstein/logs/BOEM_39831139.{out,err}

**Result: COMPLETED rc=0 at 19:02 elapsed.** All 20 flights hit the cache path
(`Using prediction cache` x20), 388 boxes >= 0.7 total. Verification over all 20 rewritten
`pool_predictions.csv`: 20/20 fresh, every row carries `hcast_species`, **0 literal `species_<int>`
placeholders** and **0 labels outside the 68-class vocabulary** — the new label_csv is in force.
**8 species the old 37-class model could not emit** now appear: Tursiops truncatus (n=22),
Pelecanus occidentalis (8), Chelonioidea sp (4), Aythya affinis (3), Anas rubripes, Ardea alba,
Branta canadensis, Puffinus griseus (1 each). Tursiops matters for
`scripts/find_tursiops_flythrough_targets.py` / `list_flights_without_tursiops.py`; Chelonioidea sp
is the turtle class from 5080197 reaching production for the first time.
Top predictions: Thalasseus maximus 189, Larus argentatus 37, Somateria mollissima 22,
Tursiops truncatus 22, Gavia immer 20, Clangula hyemalis 19, Morus bassanus 17.

**CAUTION — the cache is now narrowed.** `src/pipeline.py:563` rewrites `pool_predictions.csv` with
only the current run's rows and `:561` writes `min_score.txt` at the run's threshold. These caches went
from holding every box >= 0.3 (19,008 images) to only the >= 0.7 survivors (388 boxes), stamped 0.7.
This is self-protecting — a future run with a LOWER `predict.min_score` fails the
`cached_min_score <= predict.min_score` test and forces a full re-predict — but the cheap
"re-predict at 0.7" path is now built on a much smaller pool, and the pre-0.7 detections for these
20 flights exist only in Comet/report artifacts, not on disk.

Two pre-existing issues seen in the log, NOT introduced here and NOT fixed: (1) the report step cannot
find geospatial metadata, e.g. `/blue/ewhite/b.weinstein/BOEM/metadata_aflight_csvs/20260710_163500_captures.csv`
(non-fatal, geospatial columns end up empty); (2) counting flights by grepping the log for
`>=0.7 detections` DOUBLE-COUNTS, because that substring also appears in the
`Using prediction cache (previously had >=0.7 detections)` line — use `^Pool: [0-9]+ images with` instead.

## 39831142-39831330 (19 jobs) + 39831139 — 2026-08-20 17:15 — submit_all_flights.sh — SUBMITTED
Why: refresh the 20 July flights on the NEAQ-free 68-class hierarchical model with measured crop
geometry, so `taxonomic_summary.csv` and the consensus columns land in the per-flight reports.
Config as submitted: detection a09c6933/epoch16 (UNCHANGED — changing it invalidates every cache),
classification a3dc30a0, hierarchical `output/usgs_hier/best_checkpoint.pth` +
`output/usgs_hier/species.csv` + expand 30 / square true / eval_crop_ratio 0.875, predict.min_score
0.70. Invocation is the documented cache-only one from submit_all_flights.sh's header, `--b200`,
`--extra "active_learning.n_images=0 active_testing.n_images=0 human_review.n=0
flythrough_video.enabled=False"`, **`--time 01:00:00`** (the 08-19 lesson: 8h requests were
unbackfillable behind a 274-deep queue; ~360 images total means minutes per flight).
Cache validity confirmed live by 39831139's log: "Using prediction cache (same detection
checkpoint): running detection+classification on 9 images (previously had >=0.7 detections)
instead of 5587" — matches the 9 boxes/9 images predicted for JPG_20260710_155800.

**Human annotations are never rolled up.** Ben's call: a verified label is kept regardless of any
model prediction. Implemented as a `verified` rank ahead of species in the ladder
(`hierarchical._is_human_row`, keyed on the `set` column in {train,validation,reviewed} with the
score==2.0 sentinel as fallback), with `consensus_score` NaN so the 2.0 sentinel cannot poison
`mean_score`, and exempt from `min_consensus_score` demotion. 18/18 tests pass.

**39831139 was a `--serial` run of the ENTIRE batch, not a single flight.** First read of it was
wrong: seeing it "RUNNING JPG_20260710_155800" I took it for a one-flight duplicate and cancelled
only my matching 39831141. It is in fact the concurrent session's serial job (TimeLimit 06:00:00,
submitted 17:15:16, one job, `--serial`), and it processed all 20 flights sequentially, COMPLETED
rc=0 in **19m02s**. So my whole 20-job fan-out was redundant from the moment it was submitted, and
the other 19 were cancelled unrun. Lesson: check `grep -c 'Processing:' <log>` before concluding a
foreign BOEM job's scope — the job name is identical for serial and parallel modes.
Verified 39831139 did the real work before cancelling mine: all 20 caches rewritten (min_score.txt
now 0.7, actual floor 0.711 on the flight checked), hcast columns refreshed with the 68-class
model, and crop-vs-hcast agreement on JPG_20260712_113200 is **0.0009 -> 0.615**. It also logged
"No images to upload" x3, so nothing reached Label Studio.
Zero net GPU cost from my side; the whole survey refresh cost 19 minutes on one B200.
Result: **ALL 19 OF MINE CANCELLED, unrun and unneeded — 39831139 had already done the whole job.**
Monitor armed on all 20 job ids.
Next: on completion (1) confirm every flight logged "Using prediction cache" and a non-empty
`taxonomic_summary.csv` under <image_dir>/../reports/<flight>/; (2) aggregate the 20 summaries into
one survey-level rank split and compare against the 388-box local prediction
(species 194 / genus 17 / family 49 / unresolved 128) — they should agree closely, since the local
run used the same model, geometry and boxes; a large divergence means the pipeline path differs
from the direct path and must be explained before anything is published; (3) THEN decide whether
the PDF species-composition figures move onto consensus_label.

## 39832676 — 2026-08-20 17:5x — submit_all_flights.sh --serial — SUBMITTED
Why: **the 2026-08-20 survey refresh produced ZERO reports**, and the cause was not the model.
Every July flight in 39831139 logged `Report: could not load geospatial metadata: [Errno 2] ...
/blue/ewhite/b.weinstein/BOEM/metadata_aflight_csvs/20260710_155800_captures.csv`, so
`generate_report()` returned None at its first step (src/report.py:648-655) and no
observations table, shapefile, PDF or `taxonomic_summary.csv` was written for any flight.
**Root cause: a nested duplicate metadata directory.** `report.metadata_dir` pointed at
`/blue/ewhite/b.weinstein/BOEM/metadata_aflight_csvs`, which holds 230 `*_captures.csv` and
**zero** for 202607. The July files are one level deeper, in
`metadata_aflight_csvs/metadata_aflight_csvs/` (254 captures files, range 20201221-20260713).
The inner dir is a near-superset of the outer — the only file it lacks is
`20211201_125800_captures.csv`. Repointed `report.metadata_dir` at the inner dir (reversible,
touches no data) rather than reorganising the user's files. **These two dirs should be
consolidated properly**, and the 20211201 file carried across, before this bites again.
**One flight can never report: JPG_20260711_131000 has no capture CSV anywhere on disk**
(`find` over all of BOEM returns nothing for 20260711_131000). Its 18 boxes/18 images will be
absent from the survey rollup. Expect 19/20 reports, not 20/20 — that is not a failure.
Verified the whole report path locally before resubmitting, on JPG_20260712_113200 off its
refreshed cache: metadata loads (14,573 captures, 6464x4852), **26 of 26 boxes georeference**,
and the rollup gives species 16 / genus 2 / family 4 / unresolved 4 — 84.6% resolved, with
Thalasseus maximus 12 and Laridae 4 at family, as expected.
Submitted `--serial` this time (one job, all 20 flights) rather than a 20-way fan-out: 39831139
proved serial does the whole survey in **19m02s** off warm caches, so the fan-out only bought
queue churn. `--time 02:00:00`.
**Result: COMPLETED rc=0 at 18:21, elapsed 35m56s.** All three "Next" items are now answered.
(1) **19/20 reports, exactly as predicted.** 20/20 flights logged `Using prediction cache`, 388 boxes
>= 0.7 (identical to 39831139), and the repointed `report.metadata_dir` worked: the ONLY geospatial
failure in the whole log is the known-unfixable `20260711_131000_captures.csv`. 19 complete report sets
on disk (19 x PDF + HTML + shp/shx/dbf/prj/cpg + `taxonomic_summary.csv`, 57 csv, 127 png), all
uploaded to Comet. Zero tracebacks in the .err; its only content is per-flight Lightning GPU banners.
(2) **The pipeline rollup reconciles EXACTLY with the local 388-box prediction — no divergence to
explain.** Aggregating the 19 summaries: species 189 / unresolved 121 / family 46 / genus 14 /
**verified 3** = 373 observations. Local was species 194 / genus 17 / family 49 / unresolved 128 = 388.
The two gaps are both accounted for and neither is a discrepancy: JPG_20260711_131000 (18 boxes, no
capture CSV) is absent, and the per-rank deltas sum to precisely those 18 (5+3+3+7); the 3 `verified`
rows are human annotations the local box-only run never saw. 388 - 18 + 3 = 373. The pipeline path and
the direct path agree, so the rollup is safe to publish.
Survey composition is dominated by one species: Thalasseus maximus 141 of 192 species+verified
observations, then Larus (genus) 8, Gavia immer 6, Tursiops truncatus 6, Somateria mollissima 5,
Chelonioidea sp 4 (the turtle class, in a report for the first time).
(3) Decision on moving the PDF species-composition figures onto `consensus_label` is still open, but
now has the evidence it needed.

**BONUS — this run silently closed the "calibrated queue size is unmeasured" gap.** Temperature
scaling was already live when it launched (commit beb7001 landed 17:52, job started 17:45 off the
working tree), and the log confirms it on every flight: `[classification] temperature scaling active:
logits / 5.6136`. The 20 caches are stamped `temperature=5.6136`, and the rewritten
`pool_predictions.csv` carry calibrated scores — median `cropmodel_score` **0.398**, and only **2 of
388** boxes >= 0.99 (raw-scale share was ~81%).
**The measured calibrated review queue on survey imagery is 252 of 388 boxes / 236 of 360 images —
about 65%, not the ~24.6% / ~170 the val quantile mapping predicted.** So the val-set estimate
understates the real queue by ~2.5x: survey boxes are far less confidently classified than in-domain
val crops. `human_review.review_high: 0.50` is therefore a much wider gate in production than the val
numbers implied; the queue is still capped by `human_review.n`, and nothing was actually uploaded
("No images to upload for instance review" x20), so this is a free measurement with no side effects.
Worth revisiting whether 0.50 is the right band now that the survey-side number is known.
Next: (1) confirm 19/20 `taxonomic_summary.csv` and zero metadata failures other than
20260711_131000; (2) aggregate the 19 into a survey-level rank split and check it against the
local 388-box prediction (species 194 / genus 17 / family 49 / unresolved 128) — the local run
used the same model, geometry and boxes, so a large divergence means the pipeline path differs
from the direct path and must be explained before publishing; (3) decide on moving the PDF
species-composition figures onto consensus_label.

## (no job) — 2026-08-20 — temperature scaling DEPLOYED; every cropmodel_score threshold rescaled
Why: decided to take the interpretable/portable scale despite the measured triage cost
recorded in 39825931 (AUROC 0.796 -> 0.770). `cropmodel_score` is now a calibrated
probability everywhere in the pipeline, so `review_high` means "review anything the
classifier is worse than even odds on" instead of naming a percentile of one checkpoint.

**Where it happens.** `src/classification.apply_temperature()` binds a `forward` that divides
the logits by T before DeepForest's softmax (`deepforest/model.py:610`). Called at BOTH
CropModel load sites in `src/pipeline.py` — the `upload_full_flight` path and the main
train-or-load path, the latter placed after both branches of the if/else so a freshly TRAINED
model is scaled too. Bound to the class method rather than wrapping the previous `forward`, so
re-applying replaces rather than compounds. `temperature: null` restores raw behaviour.
Verified on a3dc30a0 (64 val crops, CPU): logits exactly divided, argmax identical on every
crop, mean score 0.943 -> 0.663, share >=0.99 0.812 -> 0.031, idempotent, rejects T<=0.

**T = 5.6136** lives in `boem_conf/classification_model/finetune.yaml` next to the checkpoint
it belongs to, on the `predict.min_score` model: paired with one checkpoint, re-derived on
every change, never inherited.

**THE DANGEROUS PART — five thresholds read cropmodel_score, and all five moved.** Changing
the scale without moving all of them is the same silent-filter failure this ledger already
records twice (the always-empty `confident_predictions`, the stale 0.85 review gate).
Mapped by val-set quantile equivalence, which is the honest method available:
| key | raw | calibrated |
| `human_review.review_low` | 0.3 | 0.05 |
| `human_review.review_high` | 0.99 | 0.50 |
| `active_learning.min_classification_score` | 0.3 | 0.07 |
| `pipeline_evaluation.classification_threshold` | 0.5 | 0.12 |
| `report.rare_species_min_score` | 0.7 | 0.21 |
`review_high` is the one exception to strict quantile-matching: the equivalent of raw 0.99 was
0.4293, and 0.50 was taken instead because it is semantically defensible and the queue is
capped by `human_review.n` anyway, so a wider pool only improves selection. At 0.50 on val:
24.6% reviewed, 51.1% of all errors caught, 2.08x lift, auto-annotated remainder at 0.842
accuracy. The list of all five is written into the `human_review` block so the coupling is
discoverable from the config rather than from this file.

**Caches are now scale-stamped.** `.prediction_cache/classifier.txt` records checkpoint +
temperature. The pipeline itself was never at risk there — it reads only the detection `score`
out of that cache and re-runs classification on the narrowed pool — but
`collect_screened_images.py`, `find_tursiops_flythrough_targets.py` and
`upload_mola_sample_to_review.py` read `cropmodel_score` straight off `pool_predictions.csv`,
and nothing distinguished a raw-scale file from a calibrated one. `.full_flight_predictions.csv`
was the genuine hazard: reused VERBATIM with no validity check of any kind, feeding Label
Studio and the flythrough video. It now carries a `.classifier` stamp and is ignored (with a
printed reason) when the checkpoint or temperature differs. **An unstamped cache is raw-scale**
— every cache on disk today predates this and will be bypassed on next run.

**Not done, and it is a real gap.** Queue sizes on the July survey CANNOT be predicted from the
existing caches. `raw -> calibrated` is not a function (max-softmax depends on the whole logit
vector), so a monotone fit over the val pairs carries a median absolute error of 0.088 and its
survey estimates are worthless — it put 12 images in the queue where the val quantile mapping
says ~170. Getting the real number needs crop logits for survey boxes, i.e. a classifier-only
re-predict over the cached detections (the 39700519 path). Until that runs, the calibrated
queue size on survey imagery is unmeasured.

**CORRECTION to the H-CAST warning first written here.** The original entry claimed the
crop/H-CAST scale mixing was a live hazard held back only by `hierarchical.checkpoint` being
unset. Both halves were wrong. `hierarchical.checkpoint` IS set
(`output/usgs_hier/best_checkpoint.pth`) and H-CAST runs on every flight. But the ensemble does
not compare confidences at all — `resolve_row_rank`'s ladder is pure LABEL comparison
(species match -> species, else genus match -> genus, else family), as are
`format_ensemble_suggestion_line` and `ensemble_target_mode: match_or_genus_consistent`.
Confidence enters in exactly two places and both are currently inactive:
  - `report.taxonomic_rollup.min_consensus_score` demotion (hierarchical.py:638-645), the only
    place joint confidence rewrites an output LABEL — set to `null`, so agreement alone decides.
  - `mean_joint_conf` (active_learning.py:408), a SECONDARY sort key after `disagreement_count`
    and only under `strategy: model-disagreement`; the active strategy is `taxonomy`.
So there is no live impact, and the scale change did not break the ensemble.

**What that review did surface is a SIXTH cropmodel_score threshold, missed in the table
above: `report.taxonomic_rollup.min_consensus_score`.** It is null today so nothing is
mis-scaled, but it is a crop-scale floor whenever it is set, for two independent reasons: a row
with no H-CAST score resolves to `_joint(crop_score)` — the crop score alone
(hierarchical.py:587) — and with H-CAST it is min(crop, hcast) where H-CAST is not
temperature-scaled. On raw scores the crop score wins that min on 17.0% of the 388 gated survey
boxes (median crop 0.981 vs H-CAST 0.629); calibrated crop scores fall into H-CAST's range, so
the min would start returning a different model's number. Exact post-calibration share is
unmeasured for the same reason the queue size is — no calibrated scores exist for survey boxes
yet. Documented inline at the key. Fix if enabled: fit a T for H-CAST, or compare on quantiles.

Tests: 67 passed, 1 skipped. Added 5 tests in `tests/test_active_learning.py` covering the
divide, argmax preservation, idempotence, T<=0 rejection, null/1.0 no-ops, and a
`human_review` band test that pins the calibrated defaults. `test_detection.py::
test_detection_preprocess_and_train` still fails on HEAD for the unrelated pre-existing
DeepForest drift (`main.deepforest(label_dict=...)`).
Next: (1) ~~classifier-only re-predict of JPG_202607* to measure the real calibrated queue~~ **DONE —
39832676 already ran calibrated. Measured: 252/388 boxes, 236/360 images, ~65% of the gated set, vs the
~24.6%/~170 the val quantile mapping predicted. The val estimate understates production by ~2.5x;
reconsider whether `review_high: 0.50` is the right band.**;
(2) record T for each future checkpoint and check whether it is stable enough that the
thresholds stop needing to move; (3) fit a T for H-CAST (or switch to quantile comparison) BEFORE
setting min_consensus_score or strategy: model-disagreement — not before an ensemble run, which
is already happening harmlessly; (4) `fast_dev_run: True` still committed in finetune.yaml.

## (no job) — 2026-08-24 — land filter trained on the first 62 Label Studio labels

Pulled the Land Screen annotations and fitted the logistic regression. **62 of the 250 uploaded
frames are annotated so far** (37 Water, 25 Land, zero Mixed, zero Unusable), so this is an early
read, not the final fit.

**The model works: cross-validated ROC-AUC 0.929, stable across 10 CV seeds (0.916-0.942).** At the
chosen operating point it dominates the hand-tuned conjunction currently in `src/land_filter.py`
on *both* axes — same 84.0% land recall, but 1/37 water frames lost instead of 7/37 (2.7% vs 18.9%).
Standardised coefficients: fine_edge -2.129, bg_frac -1.957, chroma -0.366, struct +0.091. The
struct term the hand rule leads with is doing almost nothing; the fit is carried by low fine-scale
edge energy and the absence of a single dominant background colour.

**Fixed a real defect in `scripts/fit_land_filter.py`: its headline operating point was
meaningless.** It selected the highest-recall threshold losing *zero* water frames, which is a
`max()` over the water set and therefore set by one frame. That frame pinned it at 0.930, where
**land recall is 12%** — i.e. the script recommended a threshold at which the filter does nothing,
and printed it as the answer. Replaced with a threshold chosen at a stated water-loss budget
(`WATER_LOSS_BUDGET = 0.03`), which picks **0.625: 84.0% land recall, 1/37 water lost**. The script
now also prints CV AUC, the full recall/loss curve, and the hand-tuned rule as a baseline, and
stamps `cv_auc` / `cv_land_recall` / `water_loss_budget` into `land_model.json`.

**The frame that pinned the old threshold is not a label error — it is a hard negative class the
features do not cover.** Inspected it: `C1_L3_F378_T20260710_162752_835.jpg` is turbid green
shallows with dark seagrass patches and a boat wake. Correctly labelled Water; it looks like land
to these features (many "materials", coarse structure, low fine edge). Shallow-water bottom
structure is a genuine gap, and no threshold choice fixes it — it needs a feature or more labels.

**One label IS wrong, and it costs a lot.** `C5_L1_F1034_T20260202_124404_130.jpg`
(JPG_20260202_122400) is labelled **Land** but is plainly brown chop with foam specks — water.
It is an `anchor`-band frame, so the fit weights it heavily. Correcting it:
**AUC 0.929 -> 0.971**, and zero-water-loss recall 12% -> 42%. Not corrected in the fit — the
annotation should be fixed in Label Studio rather than patched in code.

**Biggest caveat: the label set is nearly single-flight.** 23 of the 25 Land frames come from
JPG_20260710_155800 (the coastal line), and 17 of 37 Water from JPG_20260711_141200. The Dec-2024
whitecap flights — the hard negative that motivated the whole learned-filter exercise — contribute
**4 frames total**. Cross-flight generalisation is therefore unproven, and the 0.929 AUC should be
read as "separates land from water *on one coastal flight*". Spot-checked the land labels visually
(buildings, parking lots, mown grass — unambiguous).

**The filter is still not wired into the pipeline.** `src/land_filter.py` has no caller anywhere
outside its own four scripts, so no land frame is being dropped in production today. Added a
learned-inference path (`load_model`, `land_probability`, `is_land_learned`) so `land_model.json`
has a consumer at all — it previously had none — verified to match sklearn's `predict_proba` to
2.2e-16 and to recompute features from JPEG exactly. Wiring it into the detector is a separate
decision.

Tests: 67 passed, 1 skipped, unchanged. Added `tests/test_land_filter.py` (5 tests) covering the
hand-computed logistic, bounds/monotonicity, feature-order independence, scaler application, and
the per-path cache; they use a synthetic model file so they do not depend on the /blue artifact.
`test_detection.py` still fails on HEAD for the unrelated pre-existing DeepForest drift.
Next: (1) fix the JPG_20260202_122400 label in Label Studio and refit; (2) annotate the remaining
188 frames, prioritising non-coastal flights — Land is currently one flight; (3) decide whether
shallow-seagrass water needs its own feature; (4) decide where in the pipeline the filter runs.

## (no job) + 40092220 + 40098526 — 2026-08-24 — land filter: refit, and a held-out validation set

Follow-on to the entry above. Three things: the rejected annotation is out, the filter was run
over flights it has never seen, and a 400-frame correction pass is queued in Label Studio.

**Refit without the mislabelled anchor frame: CV ROC-AUC 0.929 -> 0.968**, operating point 0.610
(land recall 87.5%, 1/37 water lost; the hand-tuned rule is 87.5% at 7/37). The exclusion lives in
`scripts/fit_land_filter.py:EXCLUDE`, not in Label Studio, so a re-pull cannot silently reintroduce
it. **No new annotations have appeared since 2026-08-22 — still exactly 62 of 250 uploaded frames.**

**Infrastructure trap worth remembering: an interactive shell on this cluster gets ONE cpu.**
`nproc` = 1 and `len(os.sched_getaffinity(0))` = 1, so every `ProcessPoolExecutor(16)` in the
land-filter scripts was inert. Measured throughput was identical at 16, 32 and 64 workers
(2.3 frames/s) because it was serial the whole time. Scoring is CPU-bound at ~440 ms/frame; on 64
real cores it runs at ~118 frames/s, a 50x difference. Anything frame-parallel belongs in sbatch.
Also: job 40091435 hung with `couldn't chdir ... transport endpoint shutdown` — node c0705a-s5 had
a dead /blue mount. It sat in RUNNING producing nothing. Cancelled and resubmitted with
`--exclude=c0705a-s5`.

**Land is much rarer than the mining exercise implied: 239 of 40,000 sampled frames, 0.60%.**
And 89.7% of ALL frames sit in a tight p=0.10-0.30 band — the model is decisive about open water,
and the entire interesting region is under 2% of the data. The first pass (5000/flight, 8 flights,
job 40092220, 5m49s) was badly unbalanced: 78% of the predicted land came from two July-13 flights
(JPG_20260713_101500 at 2.30%, JPG_20260713_160300 at 1.42%), while six flights gave 4-16 frames
each. Validating on that would have reproduced the exact single-flight weakness this exercise is
meant to test. A full-depth pass over those six (job 40098526, **307,763 frames in 43m18s**) found
681 more, bringing every flight to at least 16 thinned land frames. Merged pool: 317,763 frames,
867 predicted land (0.27%), 565 after 5-frame overlap thinning.

**Uploaded 400 frames to the new `BOEM - Land Screen Validation` project (id 214) as a CORRECTION pass** —
each task pre-filled with the model's guess and p(land), inverting the no-anchoring rule of
`upload_land_project.py` on purpose so we only pay for the frames the model got wrong. Bands
(sized against measured supply, ~2/3 on the land side): land_confident p>0.85 n=120,
land_marginal 0.610-0.85 n=140, boundary_below 0.30-0.610 n=90, water_anchor <=0.30 n=50. All 8
flights represented, 29-67 each. **Precision on the two land bands is the number to read**: every
predicted-land frame that is really water is a frame the filter would have discarded, and those are
the errors that can cost an animal. The land/water ratio in this set is NOT a prevalence estimate.

**Open concern the annotation pass exists to settle: the new predicted-land frames do not look like
the training land frames.** Median struct 0.029 vs 0.041 and chroma 0.037 vs 0.049 — less
structured, less colourful — and only 25% fall inside the 5-95% envelope of the 24 labelled land
frames on all four features jointly. Either these flights hold a different KIND of land (marsh,
dune, mudflat rather than the buildings and parking lots that dominate the coastal training flight),
or a real share of them are false positives. The fit is extrapolating off its training distribution
either way.

**Also fixed:** Label Studio caps project titles at 50 characters and rejects a longer one with a
bare `400 Bad Request` and no message — the first upload attempt died there, before transferring
anything. Title shortened and a length guard added that fails with an explanation.

Verified after upload: 400 tasks, all 400 carrying a pre-filled prediction (260 Land / 140 Water),
bands and all 8 flights as intended, 0 annotations yet.

Tests: 77 passed, 1 skipped. `tests/test_land_filter.py` now has 10 tests; 5 cover band selection,
including a regression test for a bug found here — drawing the outer bands nearest-threshold-first
made `water_anchor` entirely p~=0.30 frames, i.e. not a confident-water tripwire at all. Outer bands
now draw a random spread (anchor mean p is 0.07).
Next: (1) annotate the 400 and read precision per band and per flight; (2) if precision on
`land_confident` is high but `land_marginal` is poor, the threshold moves rather than the model;
(3) the two July-13 flights were only sampled at 5000 — deep-score them if more land is wanted;
(4) still nothing in the pipeline calls the land filter.

## (no job) — 2026-08-25 — rank-10 and 90% abundance accuracy, first measurement
Why: the boss-facing report table had two empty rows with no pipeline stage behind them.
Both are now computed by `scripts/survey_metrics.py` off the existing
`classifier_confusion_a3dc30a0_predictions.csv` (3,695 val crops, a3dc30a0, overall 0.7564).
No GPU, no retrain, ~4 min of CSV reading.

**Neither split carries abundance.** `train_test_split_by_image` caps val at 100 crops/class
and `gentle_class_balance` caps train at 4x median (1896), so 27 classes sit at exactly 100 in
val — ranking anything by split counts is meaningless. Abundance had to be recovered by
re-reading the 72,329 per-image annotation CSVs under `training/crops/` with the same filters
`USGS_classification.py` applies before splitting: **241,787 crops, 78 classes**, cached to
`output/corpus_abundance.csv`. 10 of those 78 are slash-classes or rarities the split drops
(`Larus delawarensis/argentatus` at 9,372 is the big one); 68 of the model's 70 classes match.
`Aythya affinis` and `Calonectris Puffinus` have no verbatim corpus row and are excluded —
both negligible on either metric.

**Rank 10 abundance = 0.799** abundance-weighted (0.791 unweighted — the two nearly coincide
because every top-10 class is capped at exactly 100 val crops). The top 10 are 76.1% of all
231,000 labelled individuals.

**90% abundance = 0.795** abundance-weighted (0.784 macro, 0.801 pooled-crop). **22 species**
reach 90% of individuals — the cut lands at 90.5% on Branta canadensis. 2,127 val crops, so
unlike most things in this ledger it is not sample-size limited.

Both beat the 0.7564 all-class average, and the reason is the whole point of the metric:
**the other 46 species are 9.5% of individuals and average 0.499 accuracy.** The long tail is
half the class list and a tenth of the animals. Reporting a flat 70-class average understates
what the pipeline does on the animals it actually encounters and overstates what it does on
rarities.

**Weak spot inside the abundant set: `Pelecanus occidentalis` at 0.240 on 6,724 individuals** —
the largest single accuracy liability in the abundance-weighted picture, and a bigger lever
than any rare class. `Tursiops truncatus` at 0.296 is second (1,940 individuals, and only 27
val crops); it is also the configured `active_learning.target_labels`, consistent with the
0.296 val recall recorded throughout this ledger. Everything else in the 22 runs 0.63-0.99.

Caveat shared by both: accuracy is measured on the USGS val split, abundance on the labelled
corpus. Corpus abundance reflects annotation effort, not true at-sea density — the July 2026
survey itself resolved only 16 species-level taxa, 141 of 192 of them Thalasseus maximus, far
too thin to rank a top 10 from.
Next: (1) `Pelecanus occidentalis` at 0.240 is the highest-value classifier fix by a wide
margin; (2) re-run for the Refined column with `python scripts/survey_metrics.py <comet_id>`;
(3) if a survey-abundance basis is ever wanted instead of corpus abundance, it needs a
human-labelled sample of the survey itself, which does not exist yet.

## 40240876 — 2026-08-25 19:12 — submit_USGS_classification.sh — SUBMITTED
Why: First flat-classifier retrain after commit 9d2d68c fixed the two label bugs.
  "A/B" ambiguity labels now resolve to "Genus sp" instead of to the first species
  (Larus delawarensis 6,070 -> 771 crops, new Larus sp 5,301), and family-rank
  "Delphinidae" is kept as "Delphinidae sp" instead of dropped by the single-token
  filter. Cetacean training crops 342 -> 2,281. Pool 76 -> 79 classes, 73 kept
  after the parent-image split. Testing whether this closes the dolphin ->
  Larus delawarensis confusion, where a3dc30a0 scored P(Delphinidae)=0.0000 on
  11 dolphin crops it called ring-billed gull.
Result: pending
Next: on completion, re-fit temperature (scripts/classifier_confusion.py prints it;
  T is checkpoint-specific and does NOT transfer from a3dc30a0's 5.6136), then
  rerun the flat-vs-H-CAST comparison. Do not reuse a3dc30a0's min_score/temperature.

## 40241441 — 2026-08-25 19:16 — submit_compare_flat_hcast.sh — SUBMITTED
Why: User asked whether to drop the flat CropModel for H-CAST. Need one fair
  head-to-head: both models scored on the SAME a3dc30a0 val crop PNGs (flat side
  read from the saved classifier_confusion_a3dc30a0 logits, so neither model gets
  its own crop geometry), reporting species/genus/family accuracy, three ensembles,
  the disagreement head-to-head, and the Delphinidae-vs-Laridae confusion.
  Note this scores the PRE-fix a3dc30a0 checkpoint; 40240876 retrains the flat model
  with the label fixes and this should be rerun against it.
Result: FAILED (exit 1), script bug not a model result -- see 40242362.
Next: sanity check is H-CAST Species@1 ~= 76.7 (what 39614374 logged on this split).
  If it is far off, the crop geometry is wrong and the rest of the output is void.

## 40242362 — 2026-08-25 19:25 — submit_compare_flat_hcast.sh — SUBMITTED
Why: Rerun of 40241441, which FAILED (exit 1) reporting H-CAST 0.00% at every rank.
  Not a model problem: HCastWrapper.species_numeric_to_label stores names prefixed
  ("species_Alca torda") and classify_dataframe strips that in a nested helper the
  comparison script did not reuse, so zero species names matched the taxonomy and the
  shared vocabulary was empty. Script now strips the prefix and hard-fails on zero
  overlap rather than printing a 0.00% that looks like a result.
Result: COMPLETED. Sanity check passed: H-CAST 76.39 on the matched subset vs 76.72 logged
  by 39614374, so the crop geometry is right. Species@1 flat 75.64 / H-CAST 74.86;
  Family@1 flat rollup 88.71 / H-CAST native 86.63. Ensembles beat both: product
  (log-average) 78.76 species, 91.88 family. Head to head on the 899 disagreements:
  flat right 301, H-CAST right 272, both wrong 326 -- near a coin flip, neither dominates.
Next: superseded by 40242586, which adds error concentration and ensemble cetacean numbers.

## 40242586 — 2026-08-25 19:27 — submit_compare_flat_hcast.sh — SUBMITTED
Why: 40242362 COMPLETED and gave the head-to-head (flat 75.64 / H-CAST 74.86 Species@1,
  product ensemble 78.76; cetaceans 32/63 vs 39/63 right). Two numbers were missing for the
  "should we switch to hierarchical" decision: how much of the flat model's error is
  concentrated in the disagreement bucket (i.e. is disagreement worth routing to review),
  and what the ensemble does on the Delphinidae/Laridae confusion specifically.
Result: COMPLETED. Disagreement is a strong review trigger: flat is 89.2% accurate where the
  models agree and 33.5% where they disagree, so 66% of all flat errors (598/900) sit in the
  24% of crops that disagree. On cetaceans the ensemble does NOT inherit H-CAST's advantage:
  of 63 cetacean crops, right/->Laridae is flat 32/16, H-CAST 39/3, product ensemble 35/15.
  The flat model's overconfidence (P(Delphinidae)=0.0000 on the crops it calls ring-billed
  gull) dominates the product, so ensembling raises the global number but does not fix the
  dolphin/gull error. H-CAST's family head is the only thing that does.
Next: this closes the comparison against the PRE-fix a3dc30a0. Rerun once 40240876's
  2b27e044 checkpoint lands to see how much the label fixes close the gap on their own.

## 40263760 — 2026-08-26 01:26 — submit_classifier_confusion.sh 2b27e044 — SUBMITTED
Why: 40240876 COMPLETED (3h48m), micro-average val accuracy 0.7815 vs a3dc30a0's 0.7564,
  so the two label fixes are worth +2.5 points on their own. Need the val logits, the
  confusion matrix and a re-fitted temperature for the new checkpoint before it can be
  wired in or compared. 'Delphinidae sp' class accuracy is 0.95, but Delphinus delphis and
  Stenella frontalis both fell to 0.0 and Tursiops truncatus to 0.28 -- the indeterminate
  class is absorbing the species-rank dolphins, which is the tradeoff flagged when it was
  added. Confirm that in the matrix.
Result: COMPLETED. **New temperature T=4.4613** (a3dc30a0's was 5.6136 -- does not transfer).
  Overall val accuracy 0.7815 on 3,964 crops, 73 classes.
  The label fixes largely solved the dolphin/gull confusion on their own, WITHOUT any
  ensemble or H-CAST change. True species-rank dolphin crops:
                        predicted some dolphin   predicted a GULL   exact species
    a3dc30a0 (46 crops)      25 (54%)                11 (24%)            14
    2b27e044 (39 crops)      33 (85%)                 1 (3%)              5
  Of 100 true 'Delphinidae sp' crops, 98 are predicted as some dolphin and ZERO as a gull.
  Larus delawarensis is no longer an attractor: 45 predictions against 45 true (precision
  0.76, was 0.45 on 143 predictions against 100 true), and it now pulls in only other
  gulls/terns -- no dolphins, no whales.
  ACCEPTED COST: the indeterminate class absorbs the species-rank dolphins. Delphinus
  delphis recall 0.0 (16/16 -> Delphinidae sp), Stenella frontalis 0.0, Tursiops truncatus
  0.278 (8/18 -> Delphinidae sp). For a survey that is the right trade -- "dolphin, species
  indeterminate" is a true record where "ring-billed gull" was a false one -- but it means
  species-rank cetacean counts from this checkpoint are not comparable to a3dc30a0's.
Next: every cropmodel_score threshold must be refitted to T=4.4613 before this checkpoint is
  wired in (human_review.review_low/review_high, active_learning.min_classification_score,
  pipeline_evaluation.classification_threshold, report.rare_species_min_score).

## 40263761 — 2026-08-26 01:26 — submit_USGS_hierarchical.sh — SUBMITTED
Why: H-CAST must be retrained on the 2b27e044 split before any flat-vs-H-CAST comparison
  against the new checkpoint is valid. The wired H-CAST (39614374) trained on a3dc30a0's
  split, whose val crops are not disjoint from 2b27e044's train, so scoring it on
  2b27e044's val would leak. Auto-discovery picks the newest split dir, which is
  buffer_30/2b27e044 (written 2026-08-25 19:16). This is also the first run where the
  ancestor path actually does anything: commit 9d2d68c stopped the single-token filter
  from discarding every family/genus row before the extension could use it, and gave
  indeterminate "X sp" classes real ancestor ids.
Result: pending
Next: expect the hierarchy sizes line to show more species than 68 (the new split has 73
  classes incl. Delphinidae sp / Larus sp / Sterna sp / Aythya sp / Calonectris sp).
  Then rerun scripts/compare_flat_vs_hcast.py --comet-id 2b27e0442e51469c9cce3fa51927d741.

## 40263761 — 2026-08-26 — submit_USGS_hierarchical.sh — FAILED
Why: retrain H-CAST on the 2b27e044 split (see entry above).
Result: FAILED (exit 1:0) after 3h32m, mid-epoch 7, with NO traceback in either log and
  MaxRSS 15.9G against 60G requested -- not memory, no diagnostic, cause unknown. Training
  itself was healthy: the learning curve tracks 39614374 almost exactly (epoch 6 Species@1
  24.14 here vs 22.95 there), so this is not a regression from the 9d2d68c label changes.
  Two things did confirm: the fixed ancestor path added 121,255 higher-taxon rows, and the
  hierarchy is now species=72 genus=50 family=18 (was 68/49/18) -- the indeterminate classes
  got their own ids as designed.
  Also surfaced a latent problem: at ~28 min/epoch, 100 epochs needs ~47h against the 48h
  wall this script asked for. 39614374 fit in 41h42m only because its train set was smaller.
  DESTRUCTIVE SIDE EFFECT, found 2026-08-26 during /checklog and NOT noticed at the time:
  this run overwrote the wired H-CAST checkpoint. USGS_hierarchical.py saves
  best_checkpoint.pth into --output-dir on every new best from epoch 0, and the submit
  script pointed --output-dir straight at output/usgs_hier, which is the exact path
  boem_conf/hierarchical/hierarchical.yaml serves in production. At 03:50:39 the epoch-5
  weights (Species@1 26.91, 72/50/18) replaced 39614374's (76.72, 68/49/18). No backup
  exists: no other .pth under output/ is newer than 2025-12/2026-03, and the script has no
  Comet logging, so nothing was uploaded. 76.72 is UNRECOVERABLE except by retraining.
  The surviving older checkpoints are not fallbacks -- Dec 2025 usgs_hvit_c2f_b128 measured
  12.81% at the pipeline's geometry, worse than the wreck now in place. Note also that
  output/usgs_hier/species.csv is still the 68-class vocabulary, so the file in production
  now has 72 heads against 68 labels.
Next: see 40272359, then 40306075 which fixes the overwrite.

## 40272359 — 2026-08-26 05:00 — submit_USGS_hierarchical.sh — SUBMITTED
Why: rerun of 40263761. Raised --time 48h -> 96h; the partition allows 14 days and the
  script has no --resume, so a wall-clock kill throws away the entire run. No code change,
  since 40263761's crash left no diagnostic and its learning curve was healthy -- if this
  dies at a similar point that is evidence of something systematic rather than transient.
Result: FAILED (0:53) at 05:00:52 with ZERO elapsed -- killed the same second it started, no
  .out or .err written at all. sacct shows Timelimit=4-00:00:00 accepted and Reason=None, so
  the request was not rejected outright; the batch step was CANCELLED after a node was
  assigned. The ewhite-b QOS MaxWall is exactly 4-00:00:00, so --time=96:00:00 sat precisely
  on the cap. That is the leading explanation but is NOT confirmed: SLURM normally accepts a
  request equal to MaxWall. If a sub-cap run dies the same way, the cause is the node/prolog
  on hpg-b200, not the wall.
Next: superseded by 40306075 (--time 72h, under the cap either way).

## 40306075 — 2026-08-26 13:0x — submit_USGS_hierarchical.sh — SUBMITTED
Why: third attempt at the H-CAST retrain on the 2b27e044 split (40263761 crashed at epoch 7,
  40272359 was killed at 0s). Now also the ONLY way back to a working H-CAST at all, since
  40263761 destroyed the 39614374 checkpoint and every surviving fallback is worse than the
  wreck it left behind -- see the 40263761 entry.
  Two changes, both in submit_USGS_hierarchical.sh, no change to the training code:
    1. --time 96:00:00 -> 72:00:00, clear of the ewhite-b QOS 4-day cap that 40272359 sat on.
       Still covers the ~47h that 100 epochs needs at ~28 min/epoch.
    2. --output-dir now output/usgs_hier/run_$SLURM_JOB_ID instead of output/usgs_hier, so a
       crashed run can never again overwrite the checkpoint production reads. Promotion into
       output/usgs_hier/ is now a manual step after a run finishes and its accuracy checks out.
  hierarchical.yaml is deliberately LEFT pointing at the broken checkpoint (user's call,
  2026-08-26) rather than nulled, with a loud warning comment naming this job.
Result: pending
Next: on completion (expect Species@1 ~76 and hierarchy sizes 72/50/18):
  1. Promote: copy output/usgs_hier/run_40306075/best_checkpoint.pth to
     output/usgs_hier/best_checkpoint.pth.
  2. Regenerate output/usgs_hier/species.csv with scripts/build_hcast_label_csv.py off
     buffer_30/2b27e044 -- the checked-in one is the stale 68-class vocabulary and WILL
     mismatch the 72-head checkpoint. Never hand-edit it.
  3. Delete the BROKEN warning block from boem_conf/hierarchical/hierarchical.yaml.
  4. Rerun scripts/compare_flat_vs_hcast.py --comet-id 2b27e0442e51469c9cce3fa51927d741
     --hcast-checkpoint/--hcast-label-csv at this run's output, to decide whether the product
     ensemble still buys ~3 points now that the flat model's cetacean failure is fixed.
  Leave expand=30/square=true/eval_crop_ratio=0.875 pinned; that geometry is worth +14 points.

## (no job) — 2026-08-26 — land filter: dialled to zero measured water loss on collaborator request

Collaborators want the land screen maximally gentle: only unambiguous land (houses, trees,
suburban) should be dropped, nothing with water, a lagoon, or a beach that could hold birds.

**Checked Label Studio before touching anything, since the ask referenced "new annotations".**
There aren't any yet: `Bureau of Ocean Energy Management - Land Screen` is still 62/250 tasks
annotated (25 Land / 37 Water, unchanged since 2026-08-24), zero of them Mixed or Unusable, and
`BOEM - Land Screen Validation` (the 400-frame correction pass from the entry above) has only 2/400
done. So today's change is fit on the same 61 frames as the 0.610 operating point, not on new data.

**`scripts/fit_land_filter.py:WATER_LOSS_BUDGET` 0.03 -> 0.0.** Refit operating point: 0.610 ->
**0.920**, land recall 87.5% -> **33.3%**, measured water loss 1/37 -> **0/37**. This is the
"most extreme frames only" end of the curve the fit script already exposes — going further
(t>0.97) zeroes out land recall entirely, i.e. the filter stops doing anything. cv_auc unchanged at
0.968 (only the threshold moved, not the model). Note the zero-loss point is a max() over 37 water
frames, same instability the earlier 0.03 budget was chosen to avoid — one additional hard-negative
water frame could move it again.

**Added a Mixed-frame check to `fit_land_filter.py:main()`**: after fitting, it scores every frame
labelled Mixed and reports how many the chosen threshold would flag as land. Currently prints "none
labelled yet" since there are zero Mixed annotations to check against — this is a report, not a
gate, and becomes the real test of "nothing mixed gets caught" once collaborators produce some.

Tests: `tests/test_land_filter.py` unchanged, 10 passed (uses a synthetic model file, not the
refitted artifact).
Next: (1) once `BOEM - Land Screen Validation` and/or Mixed labels accumulate, rerun
`scripts/fit_land_filter.py` and read the new Mixed-frame line; (2) if Mixed frames still get
flagged at 0.920, the fix is a higher threshold or a Mixed-aware fitting objective, not more Land/
Water data; (3) still nothing in the pipeline calls the land filter.
