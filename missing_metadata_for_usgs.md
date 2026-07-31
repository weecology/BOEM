# Missing flight metadata — request list for USGS

We're missing `*_captures.csv` (and the corresponding `*_flights.csv`, `*_cameras.csv`,
and `.aflight`) for the flights below. Everything else we thought was missing turned out
to be a naming issue on our side and needs nothing from you.

Our metadata lives in `metadata_aflight_csvs/` (230 flights) and `metadata/aflights/`.
For each flight below we have the imagery but no metadata record of any kind.

## 1. Fully missing flights (4 flights, 133,969 images)

| Flight key | Images | Notes |
|---|---|---|
| `20241219_120500` | 20,964 | |
| `20241219_131500` | 37,942 | |
| `20241219_150200` | 4,544  | |
| `20260202_141900` | 70,519 | |

These are **partial-day gaps**, which is what makes us think it's an incomplete transfer
rather than data that was never collected:

- **2024-12-19** — we received `160100`, `164400`, `165600`, `172400`.
  We are missing `120500`, `131500`, `150200` (the earlier flights that day).
- **2026-02-02** — we received `094800`, `122400`.
  We are missing `141900` (the last flight that day).

Nothing for these four exists on our side: no captures/flights/cameras CSV and no
`.aflight` file.

## 2. Incomplete captures coverage (1 flight, ~5,000 images)

**2023-04-25** (`SE_JPG_2023_April25_NC`): we have 5,812 images, but only 817 of them
(14%) appear in any captures CSV.

We do have `20230425_094400`, `095500`, `095800`, `104500`, `135600`, `140400`.
The unmatched images are captured between roughly 10:00 and 11:59 — the `104500`
captures file exists but contains only 1 of our ~5,159 images from that hour, so it
looks truncated or belongs to a different camera/flight than we assumed.

Could you check whether the captures export for the 2023-04-25 late-morning flight(s)
is complete?

## 3. Minor residuals (not urgent, ~180 images total)

Small numbers of images don't appear in any captures CSV. Probably not worth chasing,
listing for completeness:

- `JPG_March29b` (2024-03-29): 74 of 4,712 unmatched
- `2023_Sept5_jpgs`: 54 of 3,141 unmatched
- `2023_Sept7_jpgs`: 27 of 787 unmatched
- `JPG_2024_Jan27`: 24 of 1,018 unmatched

---

## Not a USGS problem (for our records)

The other 14 "missing metadata" folders were a false alarm. Their images **are** in the
captures CSVs we already have — the folder names just don't encode a flight datetime, so
our lookup failed. Joining on the image `Basename` resolves them 100%:

| Folder | Resolves to |
|---|---|
| `JPG_2021_March24` | `20210324_135000` |
| `2023_June20b_JPG` | `20230620_130800` |
| `2023_June21_800ft_JPG` | `20230621_123800` |
| `JPG_2023_Dec14` | `20231214_114800` |
| `JPG_2024_Jan5_6b` | `20240105_104600` |
| `JPG_2024_Jan31` | `20240131_095100`, `20240131_115500` |
| `JPG_2024_Jan31b` | `20240131_132100` |
| `JPG_2024_Feb29a` | `20240214_150400` |
| `JPG_2024_Feb29b` | `20240214_112500` |
| `SE_JPG_2023_April24_NC` | `20230424_*` (6 flights) |

Note the folder names are unreliable: `JPG_2024_Feb29a` actually contains imagery captured
on **2024-02-14**. The image filename (`C1_L10_F3295_T20240214_161903_292.jpg`) is the
source of truth, joined to the captures `Basename` column.

Also `screened_images/JPG_20241219__150200` has a double underscore (typo) — same flight
as `20241219_150200` above.
