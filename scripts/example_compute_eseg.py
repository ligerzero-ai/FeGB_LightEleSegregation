#!/usr/bin/env python
"""Compute light-element segregation energies from the raw pickled VASP DataFrames.

End-to-end demonstration:
    1. Load raw GB and bulk DataFrames from data/raw_data/
    2. Extract final-step total energies AND Fe atom counts per row
    3. Normalise GB job-names so segregation rows match their pure-GB references:
         - 'S5-RA100-...' (a typo in the seg-job naming) -> 'S5-RA001-...'
         - S5 segregation rows live in the *doubled* supercell; the pure-GB
           reference must therefore have a '-d' suffix
    4. Apply the segregation-energy formula
           E_seg = (E_GB[Fe,X] - E_pure_GB) - (E_bulk[Fe,X] - E_bulk[Fe,Fe])
                   - n_Fe_diff * mu_Fe
       where
           n_Fe_diff = (nFe_seg - nFe_pure_GB) - (nFe_bulk[X] - nFe_bulk[Fe,Fe])
       For light elements with substitutional bulk preference (He, B, P, S)
       this term is non-zero and applies a host-per-atom correction.
    5. (--verify) Cross-check the per-row E_seg against the production
       `Eseg_KS` column in `data/checkpoints/08_df_compare_pairwise.pkl.gz`
       and report the maximum absolute deviation.

Run from anywhere; paths are resolved relative to this file.

Usage:
    python example_compute_eseg.py            # compute and print summary
    python example_compute_eseg.py --verify   # also cross-check against production Eseg_KS
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_data"
CHK = REPO / "data" / "checkpoints"

DOUBLED_GBS = {"S5-RA001-S210", "S5-RA001-S310"}


def last_value(x):
    """Final scalar from a per-ionic-step array (NaN if empty/missing)."""
    if x is None:
        return math.nan
    if hasattr(x, "__len__") and not isinstance(x, str):
        if len(x) == 0:
            return math.nan
        x = x[-1]
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return math.nan
        x = x.item() if x.size == 1 else x.flatten()[-1]
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def last_int(x):
    v = last_value(x)
    return int(v) if not math.isnan(v) else -1


def count_fe(structure_or_count):
    """Count Fe atoms. Accepts pymatgen Structure or an integer (passes through)."""
    if isinstance(structure_or_count, (int, np.integer)):
        return int(structure_or_count)
    sites = getattr(structure_or_count, "sites", None) or structure_or_count
    return sum(1 for s in sites if str(getattr(s, "specie", "")) == "Fe")


def add_E_and_nFe(df: pd.DataFrame) -> pd.DataFrame:
    """Add scalar columns 'E' (final energy) and 'nFe' (Fe atom count) to df.

    The raw VASP databases store `element_list` (e.g. ``['Fe', 'X', 'Fe']``)
    and `element_count` (e.g. ``[1, 1, 126]``) as parallel lists, with one
    entry per contiguous same-element block in the POSCAR. The Fe count is
    the sum of `element_count[i]` for indices where `element_list[i] == 'Fe'`.
    """
    df = df.copy()
    df["E"] = df["energy"].map(last_value).astype(float)
    def _fe_count(row):
        el_list  = row["element_list"]
        el_count = row["element_count"]
        if not isinstance(el_list, (list, tuple)) or not isinstance(el_count, (list, tuple)):
            return 0
        return int(sum(c for sym, c in zip(el_list, el_count) if str(sym) == "Fe"))
    df["nFe"] = df.apply(_fe_count, axis=1).astype(int)
    return df


def canonical_gb(gb_raw: str) -> str:
    canon = gb_raw.replace("RA100", "RA001")
    if canon in DOUBLED_GBS:
        canon += "-d"
    return canon


def main(verify: bool = False) -> None:
    df_gb_int   = pd.read_pickle(RAW / "FeGB_Int_vasp_database.pkl.gz")
    df_pure_gb  = pd.read_pickle(RAW / "Fe_Pure_GB_SLAB_vasp_database.pkl.gz")
    df_bulk_int = pd.read_pickle(RAW / "FeBulk_Int_vasp_database.pkl.gz")
    df_bulk_sub = pd.read_pickle(RAW / "FeBulk_Sub_vasp_database.pkl.gz")

    df_gb_int   = add_E_and_nFe(df_gb_int)
    df_pure_gb  = add_E_and_nFe(df_pure_gb)
    df_bulk_int = add_E_and_nFe(df_bulk_int)
    df_bulk_sub = add_E_and_nFe(df_bulk_sub)

    # ---- Pure GB lookup ----
    pg = df_pure_gb[df_pure_gb["job_name"].str.endswith("-GB")].copy()
    pg["GB_key"] = pg["job_name"].str.removesuffix("-GB")
    pure_E   = pg.set_index("GB_key")["E"].to_dict()
    pure_nFe = pg.set_index("GB_key")["nFe"].to_dict()

    # ---- Pure-Fe 128-atom bulk reference ----
    fe_pure = df_bulk_sub[df_bulk_sub["job_name"] == "bulk_128_atom_bcc_Fe_Fe"]
    if fe_pure.empty:
        raise RuntimeError("'bulk_128_atom_bcc_Fe_Fe' not found in FeBulk_Sub.")
    E_BULK_FE  = float(fe_pure.iloc[0]["E"])
    nFE_BULK   = int(fe_pure.iloc[0]["nFe"])
    MU_FE      = E_BULK_FE / nFE_BULK
    print(f"Pure-Fe 128-atom bulk: E = {E_BULK_FE:.4f} eV, nFe = {nFE_BULK}, mu_Fe = {MU_FE:.4f} eV/atom")

    # ---- Bulk-with-X reference per element ----
    # Per the manuscript convention, the bulk reference is the
    # thermodynamically preferred bulk site for the element:
    #   * INTERSTITIAL bulk for H, C, N, O  (lower-energy at a tetrahedral/octahedral void)
    #   * SUBSTITUTIONAL bulk for everything else, including He/B/P/S
    # (For He the *absolute* energy is lower in the interstitial cell, but the
    # manuscript follows the noble-gas-occupies-vacancy convention.)
    BULK_PREFERENCE = {
        "H": "int", "C": "int", "N": "int", "O": "int",
    }  # default: 'sub'
    df_bulk_int["element"] = df_bulk_int["job_name"].str.extract(r"^Fe_bulk_interstitial_([A-Z][a-z]?)_")
    df_bulk_sub["element"] = df_bulk_sub["job_name"].str.extract(r"^bulk_\d+_atom_bcc_Fe_([A-Z][a-z]?)$")
    # Drop pure-Fe row from the substitutional table
    df_bulk_sub = df_bulk_sub[df_bulk_sub["element"] != "Fe"]

    int_best = (df_bulk_int.dropna(subset=["element"]).sort_values("E")
                .drop_duplicates("element", keep="first").set_index("element"))
    sub_best = (df_bulk_sub.dropna(subset=["element"]).sort_values("E")
                .drop_duplicates("element", keep="first").set_index("element"))

    bulk_E, bulk_nFe = {}, {}
    for ele in set(int_best.index) | set(sub_best.index):
        pref = BULK_PREFERENCE.get(ele, "sub")
        src = int_best if (pref == "int" and ele in int_best.index) else sub_best
        if ele not in src.index:
            src = int_best if ele in int_best.index else sub_best
        bulk_E[ele]   = float(src.loc[ele, "E"])
        bulk_nFe[ele] = int(src.loc[ele, "nFe"])

    light = ["H", "He", "B", "C", "N", "O", "P", "S"]
    print("Bulk-with-X reference per light element:")
    print("  ", {ele: ("int" if bulk_nFe.get(ele) == nFE_BULK else "sub") for ele in light if ele in bulk_E})

    # ---- Parse GB and element from seg job_name ----
    df_gb_int["gb_raw"]  = df_gb_int["job_name"].str.extract(r"^([A-Za-z0-9-]+?)-i[A-Z]")
    df_gb_int["element"] = df_gb_int["job_name"].str.extract(r"-i([A-Z][a-z]?)-")
    df_gb_int = df_gb_int.dropna(subset=["gb_raw", "element"]).copy()
    df_gb_int["GB"] = df_gb_int["gb_raw"].map(canonical_gb)

    # ---- Compute E_seg ----
    rows, missing = [], 0
    for _, row in df_gb_int.iterrows():
        gb, ele = row["GB"], row["element"]
        if gb not in pure_E or ele not in bulk_E:
            missing += 1
            continue
        n_fe_diff = (row["nFe"] - pure_nFe[gb]) - (bulk_nFe[ele] - nFE_BULK)
        e_seg = ((row["E"] - pure_E[gb])
                 - (bulk_E[ele] - E_BULK_FE)
                 - n_fe_diff * MU_FE)
        rows.append({"job_name": row["job_name"], "GB": gb, "element": ele,
                     "Eseg": e_seg, "n_Fe_diff": n_fe_diff})
    seg = pd.DataFrame(rows)
    print(f"Computed E_seg for {len(seg)} interstitial GB rows ({missing} dropped: missing reference).")

    summary = seg.groupby(["GB", "element"])["Eseg"].min().unstack().round(2)
    print("\nMinimum interstitial E_seg per (GB, element) [eV]:")
    print(summary.to_string())

    # ---- Verification against production Eseg_KS ----
    if verify:
        prod = pd.read_pickle(CHK / "08_df_compare_pairwise.pkl.gz")
        # The pairwise df is keyed by job_name (its index)
        prod = prod.reset_index()
        join_col = "job_name" if "job_name" in prod.columns else "index"
        prod = prod[[join_col, "Eseg_KS"]].rename(columns={join_col: "job_name"})
        merged = seg.merge(prod, on="job_name", how="inner", validate="one_to_one")
        merged["abs_err"] = (merged["Eseg"] - merged["Eseg_KS"]).abs()
        n = len(merged)
        max_err = merged["abs_err"].max()
        mean_err = merged["abs_err"].mean()
        print(f"\nVerification vs production Eseg_KS:")
        print(f"  Matched rows: {n}")
        print(f"  Max  |E_seg_repro - Eseg_KS| = {max_err:.6f} eV")
        print(f"  Mean |E_seg_repro - Eseg_KS| = {mean_err:.6f} eV")
        # Show worst-case rows
        worst = merged.sort_values("abs_err", ascending=False).head(5)
        print(f"\n  Worst-case offsets:")
        print(worst[["job_name", "GB", "element", "Eseg", "Eseg_KS", "abs_err"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true",
                   help="Cross-check reproduced E_seg against production Eseg_KS")
    args = p.parse_args()
    main(verify=args.verify)
