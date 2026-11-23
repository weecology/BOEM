# Species geographic priors — status and experiment plan

Branch `species-geo-priors`, worktree `.claude/worktrees/species-geo-priors` (cut from
`main` 337cf31, 2026-11-23). Goal: stop the classifier naming species that do not occur at
a flight's location and season, first using our own flight metadata, then using open
occurrence data (eBird, GBIF, OBIS, MDAT).

## Decisions (2026-11-23)

1. **GBIF is the pretraining source** for the location encoder (E4), reached after a few
   experiments. BOEM may supply its own survey data, which would come first or alongside
   it. eBird is not needed for now.
2. **Captures CSVs for the unconverted aflights are coming over Globus** (user triggered
   the transfer). No converter will be written. E0 waits for the transfer.
3. **DeepForest PR #1334 is on hold.** Metadata embeddings don't belong in DeepForest yet,
   and the PR will probably be closed eventually. How to hold that code inside BOEM is an
   open question, deferred for now. BOEM still pins DeepForest to the PR branch until then.

## 1. Where things stand (2026-11-23)

**DeepForest PR [#1334](https://github.com/weecology/DeepForest/pull/1334)**
("Add spatial-temporal metadata embeddings to CropModel", branch `claude/friendly-beaver`)
- Open. CHANGES_REQUESTED by jveitchmichaelis on 2026-07-14. No commits since 2026-07-07.
- Merge conflicts: 12 commits ahead, **53 behind** DeepForest main. Needs a squash.
- BOEM's `pyproject.toml` pins DeepForest to this branch, so BOEM runs a DeepForest that is
  53 commits stale. Unpin once it merges.
- Reviewer asks:
  - Docs: state the lat/lon convention (decimal degrees), which fields are required, and
    how to load each checkpoint flavour. Remove the LLM-written prose.
  - Dataset: parse dates with `date.fromisoformat`, fall back per field (lat/lon present
    but no date), and store per-file metadata as a `NamedTuple`.
  - Tests: metadata columns parse correctly, and the embeddings fall in range.
  - Architecture (optional, "later"): subclass `CropModel` instead of adding conditionals.
    `src/size_metadata_model.py` (in the size worktree) already uses this pattern.

**Metadata has never been trained into any BOEM checkpoint.** Every run up to July logged
`0 rows -> Disabling metadata` because the join used the per-frame timestamp, not the
flight-start key. The join was fixed in 4f02f51, which is on main. Production then went
metadata-free (`use_metadata: false`, current checkpoint `0139af8c…`). **So a fair
with-vs-without-metadata comparison has never been run.** Every earlier "metadata does
nothing" result was measured on an encoder that never saw training data.

**Uncommitted work in progress:** `.claude/worktrees/size-metadata-classification` has
uncommitted changes from 2026-08-18 that feed object size (m) into the metadata branch
(`src/size_metadata_model.py`, `submit_meta_size.sh`). It was never run or committed.
Leave it alone. It is a natural later arm (size plus location).

**What we learned about the common eider (job notes in JOB_LEDGER, commit a06af0e):** the July 2026 Gulf
survey reported 50% *Somateria mollissima*. The boxes were sea foam at det 0.40–0.50, not
misclassified gannets. The cause is that a softmax over 70 classes with no background
class has to name something. Raising the detection floor to 0.70 fixed most of it
(cold-water share fell from 88% to 22%). **A geographic prior does not fix foam.** It only
changes which species the foam gets called. The prior's real job is choosing between
species for real birds, so evaluate it on human-verified real objects, separately from
false positives.

## 2. Data facts that shape the design

Flights with `*_captures.csv` in `metadata_aflight_csvs/`: 255 after the 2026-11-23 Globus
transfer (25 new flights, 75 new files, no existing file modified). Grouped by median position:

| Region | Flights | Months covered |
|---|---|---|
| Atlantic (FL east coast to ME) | 200 | every month except Oct |
| Gulf | 35 | Jan, Feb, Dec (LA/TX); **Jul 2026 (20 flights, FL Panhandle to the Keys)** |
| Great Lakes (Green Bay, 2021) | 12 | Nov–Dec |
| Nevada/Idaho | 8 | Mar, Dec |

- **The July 2026 Gulf survey now has coordinates** (Globus, 2026-11-23). Join check:
  more than 99.9% of the frames in its image folders match a captures `Basename`. Those
  July flights run along the eastern Gulf (Florida), not LA/TX, so "Gulf" covers two
  sub-regions.
- Two aircraft fly on the same day (SEABIRD1 in the Gulf, SEABIRD2 in Maine on
  2025-01-25). Group by `flights.csv` Asset plus Folder, never by date alone.
- 26 aflights still have no CSV: 2017–2018, 2022, 2023-04/06, `20241219_120500/131500/150200`,
  `20250217_115700`, `20260202_141900`, and `20260203_094600/142000`. Coordinates for these
  would have to come from the `.aflight` XML. Low priority unless they hold labelled crops.
- **Region and season are still partly confounded.** Gulf flights cover only Dec–Feb and
  July, while Atlantic flights cover the whole year. A learned lat/lon/day encoder
  trained on 35 Gulf flights can memorize flight identity instead of learning range. Any
  learned-metadata experiment must split by flight.
- Label Studio labels with a resolvable region: 8.5k Gulf and 6.1k Atlantic. 17k (54%) use
  old flight naming and do not resolve yet. *Somateria* has 82 Atlantic labels and 1 Gulf.
  The classifier's training crops (mostly USGS) are a larger pool; region coverage there is
  task E0.
- About 15 of the 70 classes are above species level (Laridae, Sterninae, *Larus*,
  Anatidae, Delphinidae, Aves, …) and about 10 are not birds (dolphins, seals, sharks,
  *Mola*). eBird covers neither, so we need a taxon-to-source map.

## 3. Two ways to add location

1. **Post-hoc prior (no retraining).**
   `logit_c + λ · log(max(prior_c(cell, week), ε))`.
   - The prior comes from outside data. λ and ε are tuned on validation data.
   - It is transparent, and you can switch it per survey.
   - Higher taxa take the max of their member species' priors.
   - The floor ε keeps vagrants possible.
   - This is the expected first win.
2. **Learned location embedding (PR #1334).** The encoder is trained end to end on our
   labels only.
   - Its ceiling is our own label coverage, which is thin for the Gulf.
   - **Variant: pretrained encoder.** Train a small presence-only location encoder (SINR
     style, Cole et al. 2023) on GBIF/eBird points for our 70 classes over the Atlantic and
     Gulf. Freeze it and feed its embedding into the CropModel metadata slot. The PR's
     metadata tensor is opaque to `CropModel`, so this needs no DeepForest change beyond
     a subclass (same pattern as `size_metadata_model.py`).
   - This is how millions of outside records reach the model without us labelling more.

## 4. Open data sources (rules checked 2026-11-23)

| Source | What it gives | Access | Fit |
|---|---|---|---|
| **eBird Status & Trends** | Modeled weekly relative abundance, 3 km grid, with 9/27 km versions | Request form at ebird.org/st/request. Key issued at once for non-commercial use. `ebirdst` R package. Citation and disclaimer required. | **Best first prior for birds.** Already a smoothed, effort-corrected range/season surface. Offshore values are extrapolated from shore-biased data, so pool coarsely. |
| eBird API 2.0 | Recent observations (last 30 days) | Free API key | Not useful for priors |
| eBird Basic Dataset | Raw checklists with effort, so true absences | Data request, terms prohibit redistribution | Only if we build our own model. Too heavy for a first pass. |
| **GBIF** | Presence-only occurrences, includes the eBird observation dataset | Search API needs no key but caps at 100k records per query. Bulk downloads need a free account and get a DOI. | Training points for the pretrained encoder, plus non-bird taxa |
| **OBIS** | Marine mammals, sharks, *Mola*, and at-sea seabird surveys (e.g. AMAPPS) | Open API | Non-bird classes |
| **MDAT (NOAA NCCOS)** | At-sea bird abundance models, 47 species, 10 km, US Atlantic only, v3 2024 | Free download (Duke SEAMAP) | BOEM-funded and built from at-sea survey data, so the best offshore Atlantic prior. Gulf products not confirmed; check GoMMAPPS. |

## 5. Grid resolution

These are mobile animals, and our flights sit offshore where occurrence data is thin.
Start coarse and let the data argue for finer. Treat resolution as a swept variable, not a
fixed design choice:

- **Space:** region (Atlantic/Gulf/Great Lakes) → 2° → 1° → 0.5°. H3 res 2/3 is an
  equal-area alternative to degrees. Make the coarse cell the main one, and smooth finer
  cells with neighbours.
- **Time:** annual → season (4) → month. Use circular week bins so December sits next to
  January.
- **Guardrails:** keep the ε floor, cap the prior's log-ratio, and pick the setting on
  flight-held-out data. Expectation: the region × season effect carries almost all the
  signal, because the eider case is a range-level error, not a 10 km one.

## 6. Experiments

| ID | What | Compute | Output |
|---|---|---|---|
| **E0** | Infrastructure: once the Globus captures CSVs land, build one global `Basename → (lat, lon, date, flight, region)` index; report metadata coverage per train/val split and per class × region | CPU | Coverage table; July survey gets coordinates |
| **E1** | Baseline "out-of-range" audit: current classifier on Gulf vs Atlantic flights at det ≥ 0.70. Count predicted species × region against a range table (eBird S&T presence plus an expert list). Split human-verified boxes from unverified ones. | 1 GPU, short | Out-of-range rate per region; which species absorb errors |
| **E2** | Post-hoc prior sweep: source {expert list, GBIF density, BOEM survey data if supplied, MDAT} × resolution × λ. Score on flight-held-out labelled crops with coordinates: top-1, per-region accuracy, out-of-range rate. Report validation and zero-shot holdout as separate numbers. | CPU over cached logits | Whether a prior helps, by how much, and at what grain |
| **E3** | Fair learned-metadata comparison now that the join works: PR #1334 encoder vs none, 3 seeds, **flights split by group** | 6 GPU runs | First real measurement of learned lat/lon/day |
| **E4** | Pretrained, frozen GBIF location encoder (plus BOEM data if supplied) in the metadata slot vs E3 and vs E2 | Encoder pretrain + 3 runs | Whether outside data beats our own labels |
| E5 (later) | Location + size (resume the size worktree) | — | — |

**Run order:** E0 → E1 → E2 are cheap and decide whether E3 and E4 are worth GPU time. E2
needs only GBIF data and cached logits.

**DeepForest PR:** on hold (Decision 3). E3 and E4 need the metadata CropModel code
somewhere BOEM can import it. Decide that before E3. E0–E2 do not depend on it.

## 7. Coordination

- Never switch the shared BOEM checkout (currently `dense-flock-diagnostics`). All work
  happens in this worktree.
- Other active work: dense-flock (shared checkout), detector-architecture-comparison (own
  worktree, DeepForest PR #1473), and a session uploading Label Studio review tasks.
  None of them touches `src/spatiotemporal_metadata.py` or the classifier metadata path.
- Log every sbatch in `JOB_LEDGER.md`.

## Results

### E0 (2026-11-23, no SLURM job: `scripts/geo_priors_e0.py`, ~1 min on a compute node)

Outputs are in `/blue/ewhite/b.weinstein/BOEM/geo_priors/`: `frame_index.csv` (6.81M frames),
`e0_crops_a3dc30a0.csv`, and `e0_class_by_region_a3dc30a0.csv`.

- **Frame index:** 6,814,779 frames, all from captures CSVs. Every `metadata_aws` frame was
  already covered, so it adds nothing. By sub-region: AtlanticNorth 3.89M, MidAtlantic
  0.98M, SouthAtlantic 0.46M, GulfEast 0.53M, GulfWest 0.52M, GreatLakes 0.26M,
  InlandWest 0.17M.
- **Coverage of the a3dc30a0 classifier crops (58,460 train+val):**

  | Source | Crops | Share |
  |---|---|---|
  | Captures (exact coordinates) | 34,284 | 58.6% |
  | Old-format NYSERDA (NY Bight, region only, 2016–2019) | 16,874 | 28.9% |
  | Old-format `boemy1` (2018) and `boemy2` (2019–20) | 6,883 | 11.8% |
  | Unknown (2023-04-25 truncated captures, a few others) | 419 | 0.7% |

  Location is unknown for the `boemy` crops (question for the user). Region is known for
  87.5% of crops.
- **The classifier training data is almost entirely Atlantic.**
  - Gulf: 673 crops (1.2%), only from Feb and Dec, and **none from July**.
  - Only 13 classes have more than 10 Gulf crops.
  - So a learned lat/lon encoder (E3) trained on our own crops cannot learn Gulf ranges.
    At most it learns "Gulf means gulls, terns and pelicans" from 673 examples.
  - This strengthens the case for an external prior (E2) and GBIF pretraining (E4).
- **Evaluation data:** the Gulf boxes in Label Studio (about 8.5k, mostly LA/TX in Dec/Feb,
  plus the July Review queue) are *not* in classifier training. They are the out-of-sample
  Gulf test set for E1 and E2. Val-split logits for a3dc30a0 are already cached at
  `/blue/ewhite/b.weinstein/BOEM/classifier_confusion_a3dc30a0_logits.npy`, but those are
  99% Atlantic.
