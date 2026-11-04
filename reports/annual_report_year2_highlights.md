# Year 2 Highlights — Source Bullets

**Award:** GM-24-02C / IAA M24AS00417 — *Artificial intelligence, open data, and accessible
visualization for monitoring protected birds in the Northern Gulf of Mexico*
**Period:** Sept 16, 2025 – Sept 15, 2026
**Compiled:** 2026-09-11 from `/blue/ewhite/b.weinstein/src/BOEM/JOB_LEDGER.md`,
`/blue/ewhite/b.weinstein/src/BOEM/annotations_backup/`, SLURM logs in `/home/b.weinstein/logs/`,
and a live Label Studio query.

> Bullets only — raw material to translate into prose. Every number is traceable to the file or
> job ID named beside it. Items marked **[PENDING]** are not yet measured and should not be
> claimed in the report.

---

## 1. Annotations

### Cumulative corpus (all years)
- **45,203 distinct annotation boxes** across **12,163 unique images** from **99 flights**, in
  **144 distinct labels** (`/blue/ewhite/b.weinstein/src/BOEM/annotations_backup/`, 1,408 CSVs,
  deduplicated on image + box geometry + label).
- Of those: **22,905 real animal boxes**, **17,286 human-marked FalsePositive boxes** (used as hard
  negatives), and **5,012 empty-frame confirmations** (one row = a human confirming a whole frame
  holds nothing).

### Produced during Year 2
- **29,833 distinct boxes on 6,662 images across 98 flights**, in **131 distinct labels** (Year 1:
  47 labels, 6 flights).
- Composition: **20,336 real animal boxes**, **8,146 rejections of machine detections**,
  **1,351 empty-frame confirmations**.
- **Real animal boxes went from 2,569 in Year 1 to 20,336 in Year 2 — a 7.9× increase**, and flight
  coverage from 6 flights to 98.
- By split: **27,250 train / 2,293 review / 290 validation**.
- Biggest months: **11,183 boxes in Feb 2026** and **6,647 in Aug 2026**.
- **3 human annotators** — B. Klossing (the large majority), S. Anacleto, B. Weinstein — plus
  **75 boxes from the SageMaker pilot**, the project's first machine-assisted annotation path.
- Most-annotated taxa in Year 2: *Phalacrocorax auritus* 3,342, *Larus argentatus* 2,220,
  Anatidae 1,932, Sterninae 1,654, Laridae 1,542, Delphinidae 1,242, *Pelecanus occidentalis* 978,
  *Halichoerus grypus* 710.

### How to characterise this effort honestly
- **The unit of human work is images reviewed, not boxes drawn.** Median is **1 box per image**;
  **4,889 of 6,662 images carry one box or a single empty-frame confirmation**. Meanwhile the
  **49 densest flock frames hold 10,455 boxes — 35% of the entire Year 2 count** (the largest single
  image has 1,076 birds). Box totals are dominated by a handful of dense aggregations and overstate
  clicks; "6,662 images adjudicated across 98 flights" is the defensible framing.
- **Boxes come from human annotation records, not model output** — the downloader reads Label
  Studio's `annotations` field, never the `predictions` field
  (`/blue/ewhite/b.weinstein/src/BOEM/src/label_studio.py:494`), and every row carries a named
  annotator.
- **But "human record" is not the same as "hand-drawn."** Review-queue tasks are uploaded
  pre-filled with model detections (`src/label_studio.py:453`); an annotator who accepts a
  pre-annotated task has its boxes copied into their annotation. So in the **review** split a box
  generally means *a human adjudicated a machine detection* — which is exactly what the 8,146
  FalsePositive rejections record. Hand-drawing dominates the **train** split. Both are real
  annotation labour; they are different labour, and the report should not blur them.

### Derived training corpus
- Classification corpus: **241,787 labelled crops across 78 classes** (`output/corpus_abundance.csv`).
- Detection corpus after the Sept 2026 refresh (job 40994108): **656,990 train rows / 133,272 test /
  22,996 zero-shot**, from **36,925 parent images**.
- Hierarchical model training pool: **182,350 rows** (122,871 of them ancestor-extended
  family/genus labels), **3,947 held-out validation crops**.

### Review throughput (live Label Studio read, 2026-09-11)
- **Bureau of Ocean Energy Management – Review: 336 of 336 tasks annotated (100%)** — the full
  July-2026 survey review pass is complete: **426 boxes, 374 real / 52 false positive**.
- **BOEM – Dolphin Species Review: 937 of 2,033 annotated (46.1%)** — species-level second pass,
  in progress.
- Training and Validation queues are empty **by design** — everything queued has been completed and
  pulled back into training.

---

## 2. Imagery processed

- **533,876 images / 3.80 TB examined — every frame of all 20 July 2026 survey flights, no
  sampling.**
- Processed **twice** end-to-end (initial pass, then a full re-predict after threshold
  recalibration): **~1.07 million image-passes**, **385 GPU-hours**
  (`/blue/ewhite/b.weinstein/BOEM/throughput.csv`).
- **27 hours wall-clock start-to-finish** for the whole survey by running **10 flights in
  parallel** — vs ~8 days sequentially.
- **~2,800 images/hour per GPU; ~50.6 GPU-hours per terabyte.**
- **33 flights of imagery on disk, 32 with cached predictions** — the archive now spans 2021–2026.
- **321,963 frames** scored by the land/water screening filter across 20 flights.
- **78 flight flythrough videos** rendered (`/blue/ewhite/b.weinstein/BOEM/flight_videos/`).
- Hardware benchmarking established the scaling lever: **a B200 GPU is 5.3× an L4** (4.11 vs
  0.78 img/s); the workload is **I/O- and JPEG-decode-bound, not GPU-bound** (12–26% SM
  utilization, 6.5 of 179 GB memory). Also disproved a suspected pipeline overhead: classification,
  crop writing, Label Studio upload, video and reporting together cost **under 0.3% of runtime**.
- Practical implication worth stating: **the entire survey can be re-processed in about a day
  whenever the model improves.**

---

## 3. Detection model — performance

### The headline: threshold recalibration
- The detection score floor inherited from an earlier, differently-calibrated model was
  **discarding ~70% of frames that contained an animal**.
- At the old cut, the 533,876-frame survey returned **91 boxes total**, and two flights returned
  **zero**. At the recalibrated cut: **41,694 boxes across 19,080 frames (3.57% of the survey)** —
  a **458× increase in recovered detections at identical runtime** (192.0 vs 189.6 GPU-h).
- Measured on a fully-reviewed holdout (**648 frames, 1,229 animals, every animal marked by a
  person**), job 39385379:

  | Threshold | Frame recall | Box precision | Queue size (img/1k) |
  |---|---|---|---|
  | 0.85 (was live) | 29.9% | 98.0% | 1.0 |
  | 0.70 (**adopted**) | **75.8%** | **90.2%** | — |
  | 0.30 | 97.8% | 78.4% | 15.5 |
  | 0.05 | 100.0% | 31.6% | 379.5 |

- **Independently confirmed by human reviewers.** Under the old setting the review queue came back
  **3 confirmed vs 34 rejected** — annotators were spending nearly all their effort on wave crests.
  After recalibration the same queue returns **92 confirmed vs 7 rejected = 93% confirmed**, against
  **90% precision predicted from held-out data**. Prediction and field experience agree.

### Start vs end of Year 2 — measured on the same holdout (jobs 41840299 / 41840640)

Both endpoints scored on the identical pinned holdout (648 fully-reviewed frames, 1,229
animals). No such comparison existed before 2026-09-12; it was run specifically for this report.

- **The recall ceiling — the share of animals the detector can find at ANY threshold — rose
  monotonically across the year: 64.2% → 67.8% → 71.4%.** This is the metric threshold tuning
  cannot touch, so **+7.2 points is genuine detector improvement**, not recalibration.
- **At matched precision (~85%), the fairest cross-model point:**

  | | Threshold | Recall (individuals) | Precision | Recall (frames) |
  |---|---|---|---|---|
  | Start of Year 2 (Sept 2025) | 0.70 | 51.5% | 85.4% | 87.5% |
  | Mid (July 2026, deployed) | 0.50 | 57.9% | 84.4% | 94.1% |
  | **End of Year 2 (Sept 2026)** | 0.60 | **61.0%** | 84.7% | **98.2%** |

  **+9.5 points of individual recall and +10.7 points of frame recall at equal precision.**
- **Annotator cost fell ~40× at equal recall.** To reach ≥97% frame recall the Sept 2025 model
  needed 40.0 flagged images per 1,000 surveyed; the Sept 2026 model needs **1.0 per 1,000**,
  and gets slightly better recall doing it.
- **Empty-water pass rate, measured on unscreened survey frames** (not a training log):
  99.80% → 99.90%; only 1 frame in 1,000 is flagged at the operating point.
- **Caveat to state plainly:** the start-of-year row is a per-flight pipeline fine-tune, the
  end-of-year row a corpus-wide retrain — different kinds of artifact, not just different
  vintages. The detector actually deployed in Sept 2025 (`f3c687c8`, trained 2025-06-11) **cannot
  be measured: its weights no longer exist on disk**, though Comet preserves its identity and
  training-set metrics (box precision 0.814, val_classification 0.127).

### Operating metrics of the current detector
- **Frame recall 75.8% · box precision 90.2% · empty-water accuracy 99.9%.**
- Two recall numbers are reported deliberately: **frame recall 75.8%** (what matters — a reviewer
  handed a frame labels every animal in it) vs **individual-box recall 44.3%**.
- Retraining improved held-out detection loss steadily across the year:
  **0.0144 → 0.0143 → 0.01369** best `val_classification`; the latest retrain (job 40994172) moves
  a pretrained model from **0.1656 → 0.0254** on this domain.

### Diagnostics that matter for Year 3
- **Box recall ceilings at 67.8% at any threshold, even 0.05** — a third of ground-truth animals are
  invisible to this architecture regardless of the cut. This is a detector limit (patch size /
  architecture), not a tuning problem.
- **Four independent score gates** were found scattered across the codebase, plus a fifth embedded
  in the checkpoint's own saved hyperparameters that silently overrode the configured value. All
  now consolidated and documented next to the checkpoint hash.
- Dense-flock under-detection was traced to the score floor, **not** a detection-count cap (caps
  measured and never bind).

---

## 4. Classification model — performance

### Accuracy
- **Flat species classifier: 76.5% micro accuracy / 65.3% macro precision across 75 classes**
  (job 40994173, Sept 2026, on the refreshed corpus). A parallel checkpoint (2b27e044, Aug 2026, never promoted) reached **78.2%** on its own split.
- **Hierarchical (H-CAST) model: Species@1 77.6%, Species@5 ~94%, Genus@1 ~84%, Family@1 ~91%**
  (job 40306075) — the production checkpoint. Reported figures in the survey brief are
  **76.7 / 82.2 / 89.1** species/genus/family.
- **Ensembling the two models beats either alone: 80.3% Species@1, 93.8% Family@1** — **+2.2 over
  the flat model, +3.2 over H-CAST** (job 40700993). Where the two disagree, the case is routed to
  a human rather than given a confident label.
- **Abundance-weighted accuracy is the honest headline for a survey product:**
  **79.9% across the 10 commonest species** (76.1% of all individuals), **79.5% across the 22
  species making up 90% of individuals** — against **49.9% on the 46-species long tail** that is
  under a tenth of the animals. Unweighted macro is 59.1%; the gap is the point.

### The biggest single gain of the year — no retraining required
- **Aligning inference crop geometry with training geometry was worth +14.06 points of species
  accuracy (62.66% → 76.72%)**, and had been silently thrown away on every pipeline run to date
  (jobs 39811016 / 39811745).
- The consequence was larger than that number suggests: **the hierarchical model actually deployed
  in the pipeline had been running at 12.81% species accuracy** on an easier 37-class subset. It
  has been replaced.

### Fixing what the model was getting wrong
- **Removed out-of-distribution close-range (NEAQ) imagery from training.** It had taught the model
  to read apparent size as species identity. Result: false *Megaptera novaeangliae* predictions
  **82 → 8**, *Delphinus delphis* recall **0.15 → 0.50**, at **no cost to overall accuracy**
  (0.759 → 0.756).
- **Fixed ambiguous-label collapse.** "A/B" uncertainty labels were being resolved to a single
  confident species, and single-word family labels were being dropped entirely. They now resolve to
  "*Genus* sp" / "Family sp". **Cetacean training crops 342 → 2,281**, and the dolphin-called-a-gull
  failure closed: **142 of 155 cetaceans now correct**.
- **Added a sea turtle class** (*Chelonioidea* sp) — previously turtles had no class and were forced
  into whatever bird looked nearest. It scores **96% accuracy** and the July survey produced the
  project's first turtle detections (4 agreed, 19 proposed).
- **Withdrew a false result rather than publish it.** The previous survey brief listed eight
  humpback whales; on the corrected model **no whale detection survives anywhere in the survey**.
  The earlier rows were an artifact of the scale shortcut and were retracted in the reissued brief.
- **Species vocabulary grew 37 → 70 → 75 classes** (hierarchy now 74 species / 52 genera /
  20 families).
- **Class-balance sweep settled an open tuning question:** downsampling factor 4.0
  (76.5% micro / 65.3% macro) beats factor 2.0 (75.5% / 60.6%) and no balancing at all
  (74.1% / 62.7%) — jobs 40994173 / 41142625 / 41142626.

### Known weak points (targeted for Year 3)
- ***Pelecanus occidentalis* 24% accuracy despite being the 7th commonest animal on record** — the
  single largest abundance-weighted liability.
- ***Tursiops truncatus* 24–30%** — and it is the configured active-learning target species.
- **Large whales have too few training images to be identified reliably either way** — absence in
  the survey is not evidence of absence at sea.
- **There is no background / "unknown" class.** A softmax must name a species for every crop it is
  handed, so out-of-distribution sea foam lands on Atlantic sea ducks with saturated confidence.
  This is structural and is the top classifier priority.

---

## 5. Workflow improvements

- **Active learning actually works now.** The human-review band contained a logic bug where the
  second filter was implied by the first — **`confident_predictions` was always empty and
  "images auto-annotated: 0" meant nothing**. The band was also excluding exactly the ambiguous
  cases it should capture. It now selects the genuinely uncertain range (classifier score 0.30–0.60)
  for review, auto-accepts above it, and discards spurious detections below.
- **Model disagreement is now a review trigger.** Where the flat and hierarchical classifiers
  disagree, the observation is reported at the rank they still share — or held back — rather than
  given a confident label the evidence does not support. In the July survey **121 of 370
  observations** were flagged this way instead of being guessed at.
- **Annotations are now version-controlled.** The Label Studio CSVs were the *only* copy of every
  human annotation on a filesystem with no snapshots, and completed tasks are deleted server-side
  after download. **1,408 CSVs are now mirrored into git** with a drift check that treats any
  in-place rewrite as a red flag.
- **Full job provenance.** Every SLURM job is logged with why / result / next in `JOB_LEDGER.md`
  (2,459 lines) — the project can reconstruct which model produced which number.
- **Checkpoint promotion is now a deliberate, manual step** into a run-scoped directory, after a
  crashed training run overwrote the production checkpoint in place.
- **Throughput is instrumented per flight** (images, bytes, wall seconds, img/s, hours/TB) so
  capacity questions are answered from data.
- **Parallel fan-out made safe** — 20 concurrent jobs were each deleting each other's Label Studio
  queue; annotations are now pulled once before a fan-out.
- **SageMaker second-pass export built** — the completed review queue is packaged for an external
  verification pass (100 images staged, first-pass verdicts attached).
- **Learned land/water screening filter** to keep shoreline frames out of the detection pool:
  **cv_AUC 0.971**, operating point tuned to **zero measured water loss across 38 water frames**,
  and validated visually against the frames it would cut. Honest caveat for the report: it is
  **not yet wired into the pipeline**, and visual audit found real limits (a parking lot scores
  0.672; the confident-land tail is beach sand, which should stay in). This is a feature-set
  limitation, not a threshold one.
- **Test suite maintained at 77 tests** through a year of substantial refactoring.
- **Map links added to review tasks**; flythrough video generation integrated into the pipeline.

---

## 6. Training / test validity

- **Split by parent image, not by crop** (36,925 parent images → 33,674 train / 1,636 test), so
  crops from the same photograph can never straddle the train/test boundary.
- **Zero-shot holdout pinned to two named flights** (JPG_20260202_141900 + JPG_20260201_134000)
  since 2026-08-06. Before pinning, each preparation run drew holdout flights at random and
  successive runs were not comparable. **Every run since is a clean data-only delta**, and the Sept
  2026 retrain deliberately kept the same holdout so "initial" and "refined" model columns are
  measured identically.
- **Recall is measured only against completely-reviewed frames** — 648 frames / 1,229 animals where
  a person marked *every* animal including the ones the model missed. Unlabelled survey imagery can
  tell you what was found but never what was missed.
- **A broken metric was identified and quarantined.** The framework's `box_recall` is ceilinged by
  the empty-frame ratio; it is documented (`deepforest_box_recall_issue.md`) and explicitly
  excluded from cross-run comparison rather than quietly reported.
- **Hard negatives are trained on, not discarded.** Human-marked false positives feed the negative
  bucket, and the positive batch fraction was moved **0.90 → 0.50** because models reserving 90% of
  each batch for annotated images are miscalibrated on empty water — which is the bulk of real
  survey imagery.
- **Empty-frame representation is controlled** (capped at 10% of train, 25% of test).
- **Classifier confidence was tested as a filter and rejected.** It is badly miscalibrated
  (**ECE 0.209**; mean confidence 0.965 against 0.756 accuracy) and accuracy is nearly flat below
  0.99. **Detection score is the discriminator** — the two correlate at **0.0085**, so a confident
  classification says nothing about whether the object is real.
- **Reporting distinguishes machine output from verified truth.** The survey brief states plainly
  that reviewers have confirmed *presence*, not *species*, and that every species figure remains
  machine output pending verification.
- **Provenance of every reported metric is written down** — which job, which checkpoint, which
  split, and which of two similarly-named quantities is meant.

---

## 7. Key figures / assets available for the report

| Asset | Full path | What it shows |
|---|---|---|
| Survey brief (interactive) | `/blue/ewhite/b.weinstein/src/BOEM/reports/summer_2026_aerial_survey.html` | The partner-facing deliverable: coverage, detections, species, accuracy, review status |
| Detection score by camera | `/blue/ewhite/b.weinstein/src/BOEM/reports/detection_score_by_camera.png` | Score distribution across cameras |
| Land filter boundary, 5 panels | `/blue/ewhite/b.weinstein/src/BOEM/reports/land_boundary_1_confident_water.png` … `_5_confident_land_CUT.png` | What the screening filter keeps vs cuts, at the operating point |
| Classifier confusion matrices | `/blue/ewhite/b.weinstein/BOEM/classifier_confusion_2b27e044_matrix.csv`, `…_a3dc30a0_matrix.csv` | Per-class accuracy; source of the NEAQ and cetacean findings |
| Threshold sweep (source of the recall/precision table) | `/blue/ewhite/b.weinstein/BOEM/threshold_sweep_a09c6933_e16.json` | Reproduces the detection row of the metrics table exactly |
| Corpus abundance | `/blue/ewhite/b.weinstein/src/BOEM/output/corpus_abundance.csv` | 241,787 crops / 78 classes; basis for abundance-weighted accuracy |
| Flythrough videos (78) | `/blue/ewhite/b.weinstein/BOEM/flight_videos/` | Per-flight visual review product |
| Throughput record | `/blue/ewhite/b.weinstein/BOEM/throughput.csv` | Per-flight images, bytes, seconds, img/s, hours/TB |

**Suggested figures to build if wanted:** (1) the threshold recalibration curve — frame recall vs
queue cost, with the before/after operating points marked; (2) annotation accumulation over Year 2
by month and split; (3) accuracy vs abundance rank, showing the common-species / long-tail gap.

---

## 8. Deliverable status

| Year 2 deliverable | Status |
|---|---|
| Detection + classification models for Gulf surveys, with evaluation metrics and model releases | **Delivered and measured.** Detection 75.8% frame recall / 90.2% precision / 99.9% empty accuracy; classification 76.7% species, 82.2% genus, 89.1% family; ensemble 80.3% species. Models on disk and pointed to from config. |
| Active learning and labeling workflow in use for identifying new images for partner labeling | **Delivered and in production use.** 20 flights queued through it; 336-frame review pass completed 100% by partners; a 2,033-task species-level second pass is 46% done. |

---

## 9. Open items — state as Year 3 plans, not Year 2 results

- **[RESOLVED 2026-09-12]** Refined-model detection metrics are now measured (job 41840299). The
  "Refined Model" column can be filled from the Sept 2026 checkpoint at the 0.70 operating point:
  **detection recall (frames) 94.2%, recall (individuals) 57.9%, precision 86.6%, empty-frame 99.9%.**
- **[DECISION NEEDED]** `boem_config.yaml` still points at the July checkpoint (`a09c6933`), and the
  Sept 2026 retrain beats it decisively **at the same 0.70 threshold — 94.2% vs 75.8% frame recall
  (+18.4 points)**, for a 3.6-point precision cost. Promoting it would recover roughly a fifth of
  occupied frames currently passed over, but invalidates every prediction cache, so it is a
  deliberate call rather than a routine swap.
- **[PENDING]** Flight position records for the 20 July flights are still outstanding — **no maps or
  density estimates are possible until they arrive**. 18 detections from one flight are already
  excluded from the taxonomic rollup for this reason.
- **[PENDING]** Species-level expert verification has not started; all species counts remain machine
  output.
- **[OPEN]** No background / "unknown" class in the classifier — top structural priority.
- **[OPEN]** Land filter is built and validated but not called by the pipeline.
- **[OPEN]** Box recall ceiling of 67.8% needs an architecture/patch-size investigation.
- **[OPEN]** Large-whale capability needs scale-aware handling (separate model or GSD-normalised
  crops), not raw mixing of close-range imagery.
