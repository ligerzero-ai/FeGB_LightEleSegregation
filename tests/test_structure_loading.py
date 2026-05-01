"""Verify that every pickled DataFrame yields valid pymatgen Structures.

Three things are checked per file:

1. ``structures`` column (raw VASP DataFrames + 04/07/09 checkpoints):
   for every row the **last** ionic-step entry is JSON-deserialised with
   ``Structure.from_str``. Unconverged rows (``convergence == False``) are
   allowed to carry a NaN/empty ``structures`` field.

2. ``final_structure`` column (04/07/09 checkpoints): each row holds a
   live ``Structure`` instance with at least one site.

3. ``structure`` column (Chargemol DataFrames): each row holds a live
   ``Structure`` instance with at least one site.

This is the regression test that originally lived as an inline smoke step
in the CI workflow; promoting it to pytest gives per-file failure
reporting and lets contributors run it locally with ``pytest -q``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Structure

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw_data"
CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"

ALL_PICKLES = sorted([*RAW_DIR.glob("*.pkl.gz"), *CHECKPOINT_DIR.glob("*.pkl.gz")])


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if hasattr(value, "__len__") and len(value) == 0:
        return True
    return False


@pytest.mark.parametrize("path", ALL_PICKLES, ids=lambda p: p.name)
def test_pickle_loads(path: Path) -> None:
    df = pd.read_pickle(path)
    assert len(df) > 0, f"{path.name} is empty"


@pytest.mark.parametrize(
    "path",
    [p for p in ALL_PICKLES if p.name != "chargemol_FeGBInterstitial.pkl.gz"
     and p.name != "chargemol_FeGBPure.pkl.gz"
     and p.name not in {"08_df_compare_pairwise.pkl.gz",
                        "08_df_sub_compare_pairwise.pkl.gz",
                        "KP_Vacancy_Formation_Energy.pkl.gz"}],
    ids=lambda p: p.name,
)
def test_structures_column_deserialises(path: Path) -> None:
    """Every converged row's last ionic-step Structure must deserialise."""
    df = pd.read_pickle(path)
    assert "structures" in df.columns, f"{path.name} missing 'structures'"

    failed: list[tuple[int, str]] = []
    missing_but_unconverged = 0
    deserialised = 0

    for idx, row in df.iterrows():
        structs = row["structures"]
        converged = bool(row["convergence"]) if "convergence" in df.columns else True

        if _is_missing(structs):
            if converged:
                failed.append((idx, "missing structures on a converged row"))
            else:
                missing_but_unconverged += 1
            continue

        try:
            s = Structure.from_str(str(structs[-1]), fmt="json")
        except Exception as exc:  # noqa: BLE001
            failed.append((idx, f"from_str raised: {exc!r}"))
            continue

        if s.num_sites == 0:
            failed.append((idx, "Structure has zero sites"))
            continue

        if "element_count" in df.columns and isinstance(row["element_count"], list):
            expected = sum(row["element_count"])
            if s.num_sites != expected:
                failed.append(
                    (idx, f"site count {s.num_sites} != sum(element_count)={expected}")
                )
                continue

        deserialised += 1

    assert not failed, (
        f"{path.name}: {len(failed)} bad row(s); first 5: {failed[:5]}"
    )
    assert deserialised > 0, f"{path.name}: no rows had loadable structures"


CHECKPOINT_FILES_WITH_FINAL = [
    CHECKPOINT_DIR / "04_df_KP_voronoi.pkl.gz",
    CHECKPOINT_DIR / "04_df_KP_voronoi_with_nn.pkl.gz",
    CHECKPOINT_DIR / "07_df_KP_filtered.pkl.gz",
    CHECKPOINT_DIR / "09_df_main_final.pkl.gz",
]


@pytest.mark.parametrize("path", CHECKPOINT_FILES_WITH_FINAL, ids=lambda p: p.name)
def test_final_structure_column(path: Path) -> None:
    df = pd.read_pickle(path)
    assert "final_structure" in df.columns
    bad: list[int] = []
    for idx, val in df["final_structure"].items():
        if not isinstance(val, Structure) or val.num_sites == 0:
            bad.append(idx)
    assert not bad, f"{path.name}: {len(bad)} non-Structure / empty rows in 'final_structure'"


CHARGEMOL_FILES = [
    RAW_DIR / "chargemol_FeGBInterstitial.pkl.gz",
    RAW_DIR / "chargemol_FeGBPure.pkl.gz",
]


@pytest.mark.parametrize("path", CHARGEMOL_FILES, ids=lambda p: p.name)
def test_chargemol_structure_column(path: Path) -> None:
    df = pd.read_pickle(path)
    assert "structure" in df.columns
    bad: list[int] = []
    for idx, val in df["structure"].items():
        if not isinstance(val, Structure) or val.num_sites == 0:
            bad.append(idx)
    assert not bad, f"{path.name}: {len(bad)} non-Structure / empty rows in 'structure'"


def test_09_cm_and_cleavage_structures() -> None:
    """09_df_main_final has cm_structure (Structure) and cleavage_final_structure_lst (array of Structures)."""
    df = pd.read_pickle(CHECKPOINT_DIR / "09_df_main_final.pkl.gz")

    bad_cm = [i for i, v in df["cm_structure"].items()
              if not isinstance(v, Structure) or v.num_sites == 0]
    assert not bad_cm, f"09: {len(bad_cm)} bad cm_structure rows"

    bad_cleav: list[tuple[int, int, str]] = []
    for idx, lst in df["cleavage_final_structure_lst"].items():
        if not hasattr(lst, "__len__") or len(lst) == 0:
            bad_cleav.append((idx, -1, "empty list"))
            continue
        for j, s in enumerate(lst):
            if not isinstance(s, Structure) or s.num_sites == 0:
                bad_cleav.append((idx, j, type(s).__name__))
    assert not bad_cleav, f"09: bad cleavage_final_structure entries: {bad_cleav[:5]}"
