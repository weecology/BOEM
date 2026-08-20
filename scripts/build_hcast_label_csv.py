"""Write the label CSV that src/hierarchical.load_hcast_model needs for a checkpoint.

The species/genus/family head indices of a checkpoint trained by
scripts/USGS_hierarchical.py are decided by load_taxonomy_restricted_to_species,
which numbers the *sorted* set of species/genera/families present in the training
split. Nothing about that ordering is stored in the checkpoint, so the CSV must be
rebuilt by replaying the same construction against the same split CSVs.

Usage:
    uv run python scripts/build_hcast_label_csv.py \
        --train-split-csv <dir>/usgs_train_split.csv \
        --val-split-csv   <dir>/usgs_val_split.csv \
        --taxonomy taxonomy.json \
        --out output/usgs_hier/species.csv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.taxonomy_hier import load_taxonomy, load_taxonomy_restricted_to_species  # noqa: E402
from src.classification import TURTLE_CLASS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-split-csv", required=True)
    parser.add_argument("--val-split-csv", required=True)
    parser.add_argument("--taxonomy", default="taxonomy.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_split_csv, low_memory=False)
    val_df = pd.read_csv(args.val_split_csv, low_memory=False)
    unique_labels = set(train_df["label"].unique()) | set(val_df["label"].unique())

    # Mirrors USGS_hierarchical.main(): --annotations-dir is passed by
    # submit_USGS_hierarchical.sh alongside both split CSVs, so ancestors are on.
    nb_classes, name_to_ids = load_taxonomy_restricted_to_species(
        args.taxonomy, unique_labels, include_ancestor_labels=True
    )
    if TURTLE_CLASS in unique_labels and TURTLE_CLASS not in name_to_ids:
        sid, gid, fid = nb_classes[0], nb_classes[1], nb_classes[2]
        name_to_ids[TURTLE_CLASS] = (fid, gid, sid)
        nb_classes = [nb_classes[0] + 1, nb_classes[1] + 1, nb_classes[2] + 1]
        print(f"[labels] synthetic taxonomy entry for {TURTLE_CLASS!r} (species_id={sid})")
    print(f"Hierarchy sizes: species={nb_classes[0]} genus={nb_classes[1]} family={nb_classes[2]}")

    # name_to_ids also contains ancestor (Family/Genus) keys pointing at a
    # representative species; keep only the true species entries, one per species id.
    triples, _ = load_taxonomy(args.taxonomy)
    species_to_triple = {s: (f, g, s) for (f, g, s) in triples}

    rows = {}
    for name, (fid, gid, sid) in name_to_ids.items():
        if name == TURTLE_CLASS:
            rows[sid] = dict(index=sid, species=TURTLE_CLASS, genus="Chelonioidea",
                             family="Chelonioidea", genus_index=gid, family_index=fid)
            continue
        if name not in species_to_triple:
            continue  # ancestor key, or a "Genus species" alias of a species already covered
        family, genus, species = species_to_triple[name]
        rows[sid] = dict(index=sid, species=species, genus=genus, family=family,
                         genus_index=gid, family_index=fid)

    missing = [i for i in range(nb_classes[0]) if i not in rows]
    if missing:
        raise SystemExit(f"No label for species indices {missing}; CSV would mislabel those heads")

    out = pd.DataFrame([rows[i] for i in range(nb_classes[0])])
    for col, n in (("genus_index", nb_classes[1]), ("family_index", nb_classes[2])):
        got = out[col].nunique()
        if got != n:
            raise SystemExit(f"{col} covers {got} distinct ids but the head has {n}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} species rows to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    sys.exit(main())
