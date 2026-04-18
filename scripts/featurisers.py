import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Tuple
import numpy as np
from utils.structure_featuriser import VoronoiSiteFeaturiser

from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
# assumes your min_dist_to_other(structure, site_index, ...) is already defined/imported
# --- Helpers ---------------------------------------------------------------

def _normalize_voronoi_output(out: Any) -> Dict[str, Any]:
    """
    Accepts either a dict or a (keys, values) pair and returns a dict.
    """
    if isinstance(out, dict):
        return out
    if (
        isinstance(out, (list, tuple))
        and len(out) == 2
        and isinstance(out[0], (list, tuple))
        and isinstance(out[1], (list, tuple))
    ):
        keys, vals = out
        return dict(zip(keys, vals))
    # Fallback if unexpected
    return {"voronoi_error": f"Unexpected output type: {type(out)}"}

def _vorNN_worker(structure, site_idx) -> Dict[str, Any]:
    """
    Single-row worker. IMPORTANT: VoronoiSiteFeaturiser must be importable at top-level.
    """
    try:
        out = VoronoiSiteFeaturiser(structure, site_idx)
        return _normalize_voronoi_output(out)
    except Exception as exc:
        return {"voronoi_error": repr(exc)}

def featurise_VorNN_and_append(
    df: pd.DataFrame,
    structure_col: str = "final_structure",
    site_idx_col: str = "site_idx",
    n_jobs: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Parallelise VoronoiSiteFeaturiser over df rows and append features to df.
    Returns a NEW DataFrame (does not modify in place).
    """
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    # Prepare inputs
    idxs = df.index.to_list()
    structures = df[structure_col].to_list()
    site_idxs = df[site_idx_col].to_list()

    features = {}
    # Optional simple progress indicator without extra dependencies
    total = len(idxs)
    printed = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {
            ex.submit(_vorNN_worker, s, i): idx
            for s, i, idx in zip(structures, site_idxs, idxs)
        }
        for k, fut in enumerate(as_completed(futures), start=1):
            idx = futures[fut]
            features[idx] = fut.result()
            if show_progress and (k == 1 or k == total or k - printed >= max(1, total // 20)):
                print(f"[Voronoi] {k}/{total} done")
                printed = k

    # Build features DataFrame aligned to original index
    feat_df = pd.DataFrame.from_dict(features, orient="index")
    feat_df = feat_df.reindex(df.index)  # keep original order

    # Prefix (optional): avoids name collisions if you already have similarly-named columns
    # feat_df = feat_df.add_prefix("vor_")

    # Merge and return
    out_df = pd.concat([df, feat_df], axis=1)
    return out_df

#%%


# --- internal worker: compute for many sites on the same structure -----------
def _group_worker(structure, site_indices, return_index, strictly_positive, eps):
    try:
        dm = structure.distance_matrix  # compute once per structure
    except Exception as exc:
        # If structure is invalid, return NaNs for all sites
        if return_index:
            return [(float("nan"), None) for _ in site_indices]
        return [float("nan") for _ in site_indices]

    out = []
    n = len(structure)
    for si in site_indices:
        # guard bad indices
        if si is None or not (0 <= int(si) < n):
            out.append((float("nan"), None) if return_index else float("nan"))
            continue

        row = dm[int(si)]
        mask = np.ones(n, dtype=bool)
        mask[int(si)] = False
        if strictly_positive:
            mask &= (row > eps)

        if not np.any(mask):
            out.append((float("nan"), None) if return_index else float("nan"))
            continue

        sub = row[mask]
        idxs = np.arange(n)[mask]
        k = int(np.argmin(sub))
        j = int(idxs[k])
        d = float(sub[k])
        out.append((d, j) if return_index else d)
    return out

# --- public API --------------------------------------------------------------
def featurise_min_solute_idx_and_append(
    df: pd.DataFrame,
    structure_col: str = "final_structure",
    site_idx_col: str = "site_idx",
    *,
    return_index: bool = False,
    strictly_positive: bool = False,
    eps: float = 1e-8,
    n_jobs: int | None = None,
    new_dist_col: str = "min_nn_dist",
    new_idx_col: str = "min_nn_index",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Parallel apply `min_dist_to_other` to all rows of df using (final_structure, site_idx).
    Groups rows by the *same* structure object to avoid recomputing distance matrices.

    Returns a NEW DataFrame with appended column(s).
    """
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    # Build groups by object identity (best effort dedup if the same object is reused)
    groups = {}  # key -> {"structure": Structure, "rows": [df_index...], "sites": [site_idx...]}
    for idx, (s, si) in df[[structure_col, site_idx_col]].iterrows():
        key = id(s)
        g = groups.get(key)
        if g is None:
            groups[key] = {"structure": s, "rows": [idx], "sites": [si]}
        else:
            g["rows"].append(idx)
            g["sites"].append(si)

    # Submit one task per unique structure
    results_dist: dict[int, float] = {}
    results_idx: dict[int, int | None] = {}

    total = len(groups)
    printed = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {
            ex.submit(
                _group_worker,
                g["structure"],
                g["sites"],
                return_index,
                strictly_positive,
                eps,
            ): key
            for key, g in groups.items()
        }
        for k, fut in enumerate(as_completed(futs), start=1):
            key = futs[fut]
            g = groups[key]
            res = fut.result()

            # stitch back to row indices
            if return_index:
                for row_idx, (d, j) in zip(g["rows"], res):
                    results_dist[row_idx] = d
                    results_idx[row_idx] = j
            else:
                for row_idx, d in zip(g["rows"], res):
                    results_dist[row_idx] = d

            if show_progress and (k == 1 or k == total or k - printed >= max(1, total // 20)):
                print(f"[min-nn] {k}/{total} unique structures processed")
                printed = k

    # Build Series aligned to df and append
    dist_s = pd.Series(results_dist, name=new_dist_col).reindex(df.index)
    out = pd.concat([df, dist_s], axis=1)
    if return_index:
        idx_s = pd.Series(results_idx, name=new_idx_col).reindex(df.index)
        out = pd.concat([out, idx_s], axis=1)
    return out

def pca_whiten(
    X,
    n_components=0.95,
    method: str = "pca",  # "pca" (default) or "zca"
    model: dict | None = None,
    eps: float = 1e-12,
):
    """
    Center -> PCA -> (optional truncation) -> Whitening.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input feature matrix (e.g., SOAP vectors).
    n_components : float|int
        If float in (0, 1], keep enough PCs to explain this fraction of variance.
        If int >= 1, keep exactly that many PCs (capped at min(n_samples, n_features)).
    method : {"pca","zca"}
        "pca" -> return PCA-whitened scores in k-dim PC space (decorrelated, unit variance).
        "zca" -> return ZCA-whitened data in original feature space (rotated back).
        If k < d, this is a truncated ZCA using top-k PCs.
    model : dict or None
        If provided, use this fitted model to transform X (no refit).
        Expected keys: {"mu","V","eigvals","k","method","eps"} from a prior fit.
    eps : float
        Small ridge for numerical stability in whitening (adds to eigenvalues).

    Returns
    -------
    Z : np.ndarray
        Whitened data:
        - method="pca": shape (n_samples, k)
        - method="zca": shape (n_samples, n_features) (truncated if k < d)
    model_out : dict
        Fitted model with keys {"mu","V","eigvals","k","method","eps"}.
        Pass this as model= later to transform new data consistently.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if model is None:
        # center
        mu = X.mean(axis=0)
        Xc = X - mu

        # SVD of centered data: Xc = U S V^T
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        eigvals = (S ** 2) / max(n - 1, 1)  # eigenvalues of covariance; length = min(n,d)
        V_full = Vt.T  # principal axes (d x r), r=min(n,d)

        # choose k
        if isinstance(n_components, float) and 0 < n_components <= 1:
            cumvar = np.cumsum(eigvals) / np.sum(eigvals) if np.sum(eigvals) > 0 else np.ones_like(eigvals)
            k = int(np.searchsorted(cumvar, n_components) + 1)
        elif isinstance(n_components, int) and n_components >= 1:
            k = min(n_components, V_full.shape[1])
        else:
            raise ValueError("n_components must be float in (0,1] or int >= 1")

        V = V_full[:, :k]  # d x k
        lam = eigvals[:k]  # (k,)
        inv_sqrt = 1.0 / np.sqrt(lam + eps)  # (k,)

        if method.lower() == "pca":
            # PCA scores then scale to unit variance
            Z = (Xc @ V) * inv_sqrt  # (n x k)
        elif method.lower() == "zca":
            # ZCA: project -> scale -> rotate back
            Z = Xc @ (V @ (inv_sqrt[:, None] * V.T))  # (n x d) truncated ZCA if k<d
        else:
            raise ValueError("method must be 'pca' or 'zca'")

        model_out = {
            "mu": mu,
            "V": V,
            "eigvals": eigvals,
            "k": k,
            "method": method.lower(),
            "eps": eps,
        }
        return Z, model_out

    # ---------- transform using existing model ----------
    mu = model["mu"]
    V = model["V"]
    eig_all = model["eigvals"]
    k = model["k"]
    method = model["method"]
    eps = model.get("eps", eps)
    lam = eig_all[:k]
    inv_sqrt = 1.0 / np.sqrt(lam + eps)
    Xc = X - mu

    if method == "pca":
        Z = (Xc @ V) * inv_sqrt
    elif method == "zca":
        Z = Xc @ (V @ (inv_sqrt[:, None] * V.T))
    else:
        raise ValueError("Invalid method in model.")

    return Z, model

def drop_similar_sites_by_cosine(
    df,
    vector_col: str = "SOAP_pca",
    groupby_col: str = "element",
    threshold: float = 0.999,
    energy_col: str | None = "Eseg",
    job_name_col: str = "job_name",  # assumes job_name column exists
):
    """
    Drops similar sites by cosine similarity, and appends columns to the returned DataFrame
    listing the job_names and Eseg values of dropped (non-representative) sites for each representative.
    """

    def _dedup_block(block: pd.DataFrame):
        idx = block.index[block[vector_col].notna()]
        if len(idx) <= 1:
            clusters = [{"rep_index": (idx[0] if len(idx)==1 else None),
                         "members": list(idx)}] if len(idx)==1 else []
            return idx, clusters

        X = np.vstack([
            np.asarray(v) if hasattr(v, "__array__") else np.array(v)
            for v in block.loc[idx, vector_col]
        ])

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        Xn = X / norms

        radius = 1.0 - threshold
        nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(Xn)
        G = nn.radius_neighbors_graph(Xn, radius=radius, mode="distance")
        G.setdiag(0.0); G.eliminate_zeros()

        n = Xn.shape[0]
        parent = list(range(n))
        rank = [0]*n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra==rb: return
            if rank[ra] < rank[rb]: parent[ra] = rb
            elif rank[ra] > rank[rb]: parent[rb] = ra
            else: parent[rb] = ra; rank[ra]+=1

        coo = G.tocoo()
        mask = coo.row < coo.col
        for i, j in zip(coo.row[mask], coo.col[mask]):
            union(int(i), int(j))

        groups = defaultdict(list)
        for k in range(n):
            groups[find(k)].append(k)

        kept_local = []
        clusters_summary = []
        idx_list = idx.to_list()
        energies = (block.loc[idx, energy_col].to_numpy()
                    if energy_col is not None and energy_col in block.columns else None)

        for members in groups.values():
            if len(members) == 1:
                rep_local = members[0]
            else:
                rep_local = (members[int(np.argmin(energies[members]))]
                             if energies is not None else min(members))
            kept_local.append(rep_local)
            clusters_summary.append({
                "rep_index": idx_list[rep_local],
                "members": [idx_list[m] for m in sorted(members)],
            })

        kept_global_idx = [idx_list[i] for i in sorted(kept_local)]
        return pd.Index(kept_global_idx), clusters_summary

    # Run per group
    all_keep = []
    all_clusters = []
    if groupby_col is None:
        keep_idx, clusters = _dedup_block(df)
        all_keep.append(keep_idx)
        all_clusters.extend(clusters)
    else:
        for _, sub in df.groupby(groupby_col, sort=False):
            keep_idx, clusters = _dedup_block(sub)
            all_keep.append(keep_idx)
            all_clusters.extend(clusters)

    # ---- portable combine (no union_many) ----
    combined = pd.Index([], dtype=df.index.dtype)
    for ix in all_keep:
        combined = combined.append(ix)
    keep_idx = pd.Index(pd.unique(combined))  # preserves first-seen order

    filtered = df.loc[keep_idx].copy().reset_index(drop=True)
    summary = pd.DataFrame(all_clusters)

    # --- Append dropped job_names and dropped Esegs to new columns ---
    rep_to_dropped = {}
    rep_to_dropped_Esegs = {}
    for cluster in all_clusters:
        rep_idx = cluster["rep_index"]
        members = cluster["members"]
        dropped_indices = [m for m in members if m != rep_idx]
        # Dropped job_names
        if len(dropped_indices) == 0:
            rep_to_dropped[rep_idx] = []
            rep_to_dropped_Esegs[rep_idx] = []
        else:
            if job_name_col in df.columns:
                dropped_job_names = df.loc[dropped_indices, job_name_col].tolist()
            else:
                dropped_job_names = dropped_indices  # fallback: just indices
            rep_to_dropped[rep_idx] = dropped_job_names
            # Dropped Esegs
            if energy_col is not None and energy_col in df.columns:
                dropped_Esegs = df.loc[dropped_indices, energy_col].tolist()
            else:
                dropped_Esegs = [None] * len(dropped_indices)
            rep_to_dropped_Esegs[rep_idx] = dropped_Esegs

    dropped_col = []
    dropped_Esegs_col = []
    for i, row in filtered.iterrows():
        orig_idx = keep_idx[i]  # original index in df
        dropped = rep_to_dropped.get(orig_idx, [])
        dropped_Esegs = rep_to_dropped_Esegs.get(orig_idx, [])
        dropped_col.append(dropped)
        dropped_Esegs_col.append(dropped_Esegs)
    filtered["dropped_job_names"] = dropped_col
    filtered["dropped_Esegs"] = dropped_Esegs_col

    return filtered, summary
    # --- Append dropped job_names to a new column ---
    # Map from rep_index to dropped job_names
    rep_to_dropped = {}
    for cluster in all_clusters:
        rep_idx = cluster["rep_index"]
        members = cluster["members"]
        # Find dropped indices (all except rep)
        dropped_indices = [m for m in members if m != rep_idx]
        if len(dropped_indices) == 0:
            rep_to_dropped[rep_idx] = []
        else:
            # Get job_names for dropped indices
            if job_name_col in df.columns:
                dropped_job_names = df.loc[dropped_indices, job_name_col].tolist()
            else:
                dropped_job_names = dropped_indices  # fallback: just indices
            rep_to_dropped[rep_idx] = dropped_job_names

    # Now, for each row in filtered, set the dropped_job_names column
    dropped_col = []
    for i, row in filtered.iterrows():
        idx = row.name  # after reset_index, this is the new integer index
        orig_idx = keep_idx[i]  # original index in df
        dropped = rep_to_dropped.get(orig_idx, [])
        dropped_col.append(dropped)
    filtered["dropped_job_names"] = dropped_col

    return filtered, summary
