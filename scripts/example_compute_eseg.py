#!/usr/bin/env python
"""Compute light-element segregation energies from the raw pickled VASP DataFrames.

Demonstrates the full pipeline:
    1. Load raw GB and bulk DataFrames from data/raw_data/
    2. Extract final-step energies and atom counts
    3. Apply the segregation-energy formula:
           E_seg = (E_GB[Fe,X] - E_GB[Fe]) - (E_bulk[Fe,X] - E_bulk[Fe])
                   - n_Fe_diff * mu_Fe
       where mu_Fe = E_bulk[Fe] / 128 = -8.2374 eV/atom.
    4. Print per-(GB, element) minimum E_seg

Run from anywhere; paths are resolved relative to this file.
"""
import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_data"

# --- Bulk Fe reference (128-atom bcc supercell) -----------------------------
HOST_NATOMS_BULK = 128
MU_FE = -1054.3935 / HOST_NATOMS_BULK   # = -8.2374 eV/atom (KS parameters)


def last_value(x):
    """Return the final entry of a per-ionic-step array, else x itself."""
    if hasattr(x, "__len__") and not isinstance(x, str) and len(x) > 0:
        return x[-1]
    return x


def count_fe(structure_or_count):
    """Count Fe atoms in a structure (or pass-through if already a count)."""
    if isinstance(structure_or_count, (int, np.integer)):
        return int(structure_or_count)
    sites = getattr(structure_or_count, "sites", None) or structure_or_count
    return sum(1 for s in sites if str(s.specie) == "Fe")


def main() -> None:
    # ---- Load raw DataFrames ------------------------------------------------
    df_gb_int  = pd.read_pickle(RAW / "FeGB_Int_vasp_database.pkl.gz")
    df_pure_gb = pd.read_pickle(RAW / "Fe_Pure_GB_SLAB_vasp_database.pkl.gz")
    df_bulk_int = pd.read_pickle(RAW / "FeBulk_Int_vasp_database.pkl.gz")
    df_bulk_sub = pd.read_pickle(RAW / "FeBulk_Sub_vasp_database.pkl.gz")

    # ---- Extract final-step total energies ---------------------------------
    for df in (df_gb_int, df_pure_gb, df_bulk_int, df_bulk_sub):
        df["E"] = df["energy"].map(last_value).astype(float)

    # ---- Pure GB lookup (per GB) -------------------------------------------
    # Each row in df_pure_gb is a pure GB with the GB name embedded in
    # job_name (e.g. 'S5-RA001-S310-d-GB'). We strip the '-GB' suffix.
    df_pure_gb["GB"] = df_pure_gb["job_name"].str.replace("-GB", "", regex=False)
    pure_gb_E = df_pure_gb.set_index("GB")["E"].to_dict()

    # ---- Bulk references per element ---------------------------------------
    # job_name pattern in FeBulk_Int / FeBulk_Sub: "Fe_bulk_<element>_<idx>"
    # We use the most-stable (lowest-energy) bulk site per element.
    def best_bulk_per_ele(df):
        df = df.copy()
        df["element"] = df["job_name"].str.extract(r"_([A-Z][a-z]?)(?:_|$)")
        return df.dropna(subset=["element"]).groupby("element")["E"].min()

    bulk_int_E = best_bulk_per_ele(df_bulk_int)
    bulk_sub_E = best_bulk_per_ele(df_bulk_sub)
    # Lowest of int/sub per element (the thermodynamically preferred site)
    bulk_E = pd.concat([bulk_int_E, bulk_sub_E], axis=1).min(axis=1).to_dict()

    # ---- GB segregation rows -----------------------------------------------
    # Job-name template e.g. "S5-RA001-S310-d-iH-site-37" / "...-sH-site-37"
    df_gb_int["GB"]      = df_gb_int["job_name"].str.extract(r"^([A-Za-z0-9-]+)-i[A-Z]")
    df_gb_int["element"] = df_gb_int["job_name"].str.extract(r"-i([A-Z][a-z]?)-")

    rows = []
    missing = 0
    for _, row in df_gb_int.dropna(subset=["GB", "element"]).iterrows():
        gb, ele = row["GB"], row["element"]
        if gb not in pure_gb_E or ele not in bulk_E:
            missing += 1
            continue
        # Composition difference: assume one solute X added vs pure GB =>
        #     n_Fe_diff (GB) = -1 if X replaces Fe, 0 if X is interstitial
        # The interstitial dataset always adds X without removing Fe.
        n_fe_diff = 0
        e_seg = (row["E"] - pure_gb_E[gb]) - (bulk_E[ele]) - n_fe_diff * MU_FE
        rows.append({"GB": gb, "element": ele, "Eseg": e_seg, "job_name": row["job_name"]})
    seg = pd.DataFrame(rows)
    print(f"Computed E_seg for {len(seg)} interstitial GB rows ({missing} dropped, missing reference).")

    # ---- Per-(GB, element) minimum -----------------------------------------
    summary = seg.groupby(["GB", "element"])["Eseg"].min().unstack().round(2)
    print("\nMinimum interstitial E_seg per (GB, element) [eV]:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
