"""Load taxonomy.json and build hierarchical label mappings (family, genus, species).

Taxonomy is a nested JSON: Class -> Order -> Family -> Genus -> Species.
We extract (family, genus, species) for each species leaf and build
stable id mappings and species_name -> (family_id, genus_id, species_id).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _walk_species(
    node: dict[str, Any],
    path: list[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    """Recursively collect (family_scientificName, genus_scientificName, species_scientificName)."""
    rank = (node.get("rank") or "").strip()
    name = (node.get("scientificName") or "").strip()
    if not name:
        return []
    path = path + [(rank, name)]
    children = node.get("children") or []
    if not children:
        if rank == "Species":
            family = genus = species = None
            for r, n in path:
                if r == "Family":
                    family = n
                elif r == "Genus":
                    genus = n
                elif r == "Species":
                    species = n
            if family is not None and genus is not None and species is not None:
                return [(family, genus, species)]
        return []
    out = []
    for c in children:
        out.extend(_walk_species(c, path))
    return out


def load_taxonomy(taxonomy_path: str | Path) -> tuple[list[tuple[str, str, str]], dict[str, tuple[int, int, int]]]:
    """Load taxonomy JSON and build species list and name -> (family_id, genus_id, species_id).

    Returns:
        species_triples: list of (family_scientificName, genus_scientificName, species_scientificName).
        name_to_ids: dict mapping species_scientificName (and normalized "Genus species") to
            (family_id, genus_id, species_id). IDs are 0-based, stable order (sorted unique names).
    """
    path = Path(taxonomy_path)
    with path.open() as f:
        data = json.load(f)
    triples: list[tuple[str, str, str]] = []
    for root in data if isinstance(data, list) else [data]:
        triples.extend(_walk_species(root, []))
    families = sorted({t[0] for t in triples})
    genera = sorted({t[1] for t in triples})
    species_list = sorted({t[2] for t in triples})
    family_to_id = {f: i for i, f in enumerate(families)}
    genus_to_id = {g: i for i, g in enumerate(genera)}
    species_to_id = {s: i for i, s in enumerate(species_list)}
    name_to_ids: dict[str, tuple[int, int, int]] = {}
    for family, genus, species in triples:
        fid = family_to_id[family]
        gid = genus_to_id[genus]
        sid = species_to_id[species]
        name_to_ids[species] = (fid, gid, sid)
        # Also map "Genus species" (same as scientificName for binomials)
        name_to_ids[f"{genus} {species}"] = (fid, gid, sid)
    return triples, name_to_ids


def get_nb_classes(taxonomy_path: str | Path) -> list[int]:
    """Return [n_species, n_genera, n_families] for hierarchical model head sizes."""
    triples, _ = load_taxonomy(taxonomy_path)
    n_families = len({t[0] for t in triples})
    n_genera = len({t[1] for t in triples})
    n_species = len({t[2] for t in triples})
    return [n_species, n_genera, n_families]


def load_taxonomy_restricted_to_species(
    taxonomy_path: str | Path,
    species_names: set[str] | list[str],
    include_ancestor_labels: bool = False,
) -> tuple[list[int], dict[str, tuple[int, int, int]]]:
    """Build nb_classes and name_to_ids only for species that appear in the given set.

    Use this to match CropModel: classes = only those present in the training data
    (and in the taxonomy). species_names is the set of labels from your data
    (e.g. "Actitis macularius"). Only species in both the data and the taxonomy
    are included; their families and genera define the hierarchy sizes.

    If include_ancestor_labels is True, name_to_ids is also populated for Family
    and Genus scientificNames (e.g. "Delphinidae", "Tursiops"). Each ancestor
    maps to a representative (fid, gid, sid) from the first species in that
    family/genus in the restricted set (for use as training targets when the
    annotation is labeled only at family or genus level).

    Returns:
        nb_classes: [n_species, n_genera, n_families] for the restricted set.
        name_to_ids: dict from species (and optionally ancestor) name to (family_id, genus_id, species_id).
    """
    triples, _ = load_taxonomy(taxonomy_path)
    want = set(species_names)
    restricted = [(f, g, s) for (f, g, s) in triples if s in want]
    if not restricted:
        return [0, 0, 0], {}
    families = sorted({t[0] for t in restricted})
    genera = sorted({t[1] for t in restricted})
    species_list = sorted({t[2] for t in restricted})
    family_to_id = {f: i for i, f in enumerate(families)}
    genus_to_id = {g: i for i, g in enumerate(genera)}
    species_to_id = {s: i for i, s in enumerate(species_list)}
    name_to_ids: dict[str, tuple[int, int, int]] = {}
    for family, genus, species in restricted:
        fid = family_to_id[family]
        gid = genus_to_id[genus]
        sid = species_to_id[species]
        name_to_ids[species] = (fid, gid, sid)
        name_to_ids[f"{genus} {species}"] = (fid, gid, sid)
        if include_ancestor_labels:
            # Map ancestor names to this (fid, gid, sid); first occurrence wins.
            if family not in name_to_ids:
                name_to_ids[family] = (fid, gid, sid)
            if genus not in name_to_ids:
                name_to_ids[genus] = (fid, gid, sid)
    nb_classes = [len(species_list), len(genera), len(families)]
    return nb_classes, name_to_ids
