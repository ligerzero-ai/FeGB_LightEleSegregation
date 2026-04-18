import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Tuple, Dict, List, Collection

def _build_gb_colors(
    gbs,
    custom_colors: Optional[Dict[str, str]] = None,
    cmap_name: str = "tab20",
) -> Dict[str, str]:
    """
    Deterministically build a color map for the given GB labels.
    Respects any colors passed in `custom_colors` and fills the rest
    from a matplotlib colormap.
    """
    colors = {}
    if custom_colors:
        colors.update(custom_colors)

    cmap = plt.get_cmap(cmap_name)
    for i, g in enumerate(sorted(gbs)):  # stable mapping across runs
        if g not in colors:
            colors[g] = cmap(i % cmap.N)
    return colors

def _build_gb_markers(
    gbs,
    custom_markers: Optional[Dict[str, str]] = None,
    default_cycle: Optional[List[str]] = None,
    fallback: str = "o",
) -> Dict[str, str]:
    """
    Build a marker map for GB labels. Uses `custom_markers` where provided,
    then cycles through `default_cycle`, and finally falls back to `fallback`.
    """
    if default_cycle is None:
        # printer-friendly set
        default_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]

    markers = {}
    if custom_markers:
        markers.update(custom_markers)

    i = 0
    for g in sorted(gbs):  # stable mapping
        if g not in markers:
            if len(default_cycle) > 0:
                markers[g] = default_cycle[i % len(default_cycle)]
                i += 1
            else:
                markers[g] = fallback
    return markers

def scatter_by_gb(
    df: pd.DataFrame,
    x: str,
    y: str,
    gb_col: str = "GB_string",
    gb_colors: Optional[Dict[str, str]] = None,
    gb_label_map: Optional[Dict[str, str]] = None,
    legend_order: Optional[List[str]] = None,   # explicit legend order
    # --- marker style controls ---
    gb_markers: Optional[Dict[str, str]] = None,     # mapping GB -> marker
    default_marker: str = "o",
    marker_edgecolor: Optional[str] = None,
    marker_edgewidth: Optional[float] = None,  # changed: now Optional and default None
    s: float = 28,                                   # marker area (points^2)
    alpha: float = 0.85,
    # --- figure / text sizing ---
    figsize: Tuple[float, float] = (5.0, 4.0),
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    label_fontsize: int = 11,        # axis label fontsize
    tick_fontsize: int = 10,         # tick label fontsize
    # --- main legend (GB types) ---
    legend_fontsize: int = 11,       # legend text fontsize
    legend_markersize: float = 7.0,  # legend marker size (points)
    legend_ncol: Optional[int] = 1,  # <-- default to 1 column
    legend_title: Optional[str] = None,
    show_legend: bool = True,
    legend_labelspacing: Optional[float] = None,    # <-- added for label to label vertical (or horizontal) spacing
    legend_columnspacing: Optional[float] = None,   # <-- added for horizontal space between columns
    legend_borderpad: Optional[float] = None,       # <-- added padding between the legend and the axes
    legend_handletextpad: Optional[float] = None,   # <-- added padding between legend handle and text
    legend_handlelength: Optional[float] = None,    # <-- added length of legend handle
    legend_borderaxespad: Optional[float] = None,   # <-- added pad between legend & axes
    legend_titlepad: Optional[float] = None,        # <-- added pad above the legend title
    # --- style legend (filled vs hollow) ---
    show_style_legend: bool = True,
    style_legend_labels: Tuple[str, str] = ("substitutional", "interstitial"),
    style_legend_markersize: Optional[float] = None,  # defaults to legend_markersize if None
    style_marker: str = "o",
    style_legend_fontsize: Optional[int] = None,      # defaults to legend_fontsize if None
    style_legend_frameon: bool = True,
    style_legend_loc: str = "upper left",
    style_legend_anchor: Optional[Tuple[float, float]] = (0.02, 0.98),  # inside axes (x,y)
    style_legend_labelspacing: Optional[float] = None,    # <-- style legend additional spacings
    style_legend_columnspacing: Optional[float] = None,
    style_legend_borderpad: Optional[float] = None,
    style_legend_handletextpad: Optional[float] = None,
    style_legend_handlelength: Optional[float] = None,
    style_legend_borderaxespad: Optional[float] = None,
    style_legend_titlepad: Optional[float] = None,
    # --- layout & rendering ---
    tight_layout: bool = True,
    rasterized: bool = False,
    # --- legend placement (inside axes by default) ---
    legend_loc: str = "upper right",
    legend_anchor: Optional[Tuple[float, float]] = (0.98, 0.98),  # axes-fraction
    # --- axis limits ---
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    # --- fill/hollow control from a column (e.g., "calc_description") ---
    fill_by_col: Optional[str] = "calculation_description",
    hollow_values: Optional[Collection[str]] = ("GB_int",),
    ax=None,
    gb_sigma_mapping: Optional[Dict[str, int]] = None,    # <--- added for ordering by mapping
):
    """
    Single scatter plot; points colored and *optionally marked* by GB type.

    Interstitial-as-hollow:
      - If `fill_by_col` is present and df[fill_by_col] ∈ `hollow_values`
        (default: "GB_int"), those points are drawn hollow (facecolors='none',
        edgecolors=<GB color>). Others (e.g., "GB_sub") are filled.

    Legends:
      - Main legend lists GB types (default: **1 column** unless overridden).
      - Secondary "style" legend explains marker fill semantics:
          'substitutional' (filled), 'interstitial' (hollow).
        Control with `show_style_legend`, position via `style_legend_loc`/`style_legend_anchor`.

    Returns:
        gb_colors, (fig, ax)
    """
    df = df.dropna(subset=[gb_col, x, y]).copy()

    # Build maps for colors and markers based on GBs present in *this* df
    present_gbs = list(df[gb_col].unique())
    gb_colors = _build_gb_colors(present_gbs, custom_colors=gb_colors)
    gb_markers = _build_gb_markers(present_gbs, custom_markers=gb_markers, fallback=default_marker)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Setup marker linewidth logic
    # If marker_edgewidth is None, use matplotlib default (do not pass linewidths at all)
    # If marker_edgewidth is not None, pass it as linewidths to scatter/legend handles

    # Determine hollow/filled logic availability
    use_hollow_logic = (
        fill_by_col is not None and
        (fill_by_col in df.columns) and
        hollow_values is not None
    )

    # Plot grouped by GB type
    for gb_type, chunk in df.groupby(gb_col, sort=False):
        color = gb_colors.get(gb_type, "#808080")
        marker = gb_markers.get(gb_type, default_marker)

        if use_hollow_logic:
            hollow_mask = chunk[fill_by_col].isin(hollow_values)
            filled = chunk[~hollow_mask]
            hollow = chunk[hollow_mask]

            if not filled.empty:
                scatter_kwargs = dict(
                    s=s, alpha=alpha,
                    facecolors=color,
                    edgecolors=(marker_edgecolor if marker_edgecolor is not None else "none"),
                    marker=marker,
                    label=gb_type,
                    rasterized=rasterized,
                )
                if marker_edgewidth is not None:
                    scatter_kwargs["linewidths"] = marker_edgewidth
                ax.scatter(
                    filled[x], filled[y], **scatter_kwargs
                )
            if not hollow.empty:
                scatter_kwargs = dict(
                    s=s, alpha=alpha,
                    facecolors="none",
                    edgecolors=(marker_edgecolor if marker_edgecolor is not None else color),
                    marker=marker,
                    label=None if not filled.empty else gb_type,
                    rasterized=rasterized,
                )
                if marker_edgewidth is not None:
                    scatter_kwargs["linewidths"] = marker_edgewidth if marker_edgewidth > 0 else 1.0
                else:
                    if marker_edgewidth == 0.0:
                        scatter_kwargs["linewidths"] = 1.0
                ax.scatter(
                    hollow[x], hollow[y], **scatter_kwargs
                )
        else:
            scatter_kwargs = dict(
                s=s, alpha=alpha,
                facecolors=color,
                edgecolors=(marker_edgecolor if marker_edgecolor is not None else "none"),
                marker=marker,
                label=gb_type,
                rasterized=rasterized,
            )
            if marker_edgewidth is not None:
                scatter_kwargs["linewidths"] = marker_edgewidth
            ax.scatter(
                chunk[x], chunk[y], **scatter_kwargs
            )

    # Title & labels
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, pad=8)
    ax.set_xlabel(x_label or x, fontsize=label_fontsize)
    ax.set_ylabel(y_label or y, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    # Axis limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # ---- Main legend (GB types) ----
    if show_legend:
        gb_present = present_gbs

        legend_order_from_sigma_mapping = None
        if gb_sigma_mapping is not None and isinstance(gb_sigma_mapping, dict):
            legend_order_from_sigma_mapping = [g for g in gb_sigma_mapping.keys() if g in gb_present]
            legend_order_from_sigma_mapping += [g for g in gb_present if g not in legend_order_from_sigma_mapping]

        if legend_order is not None:
            ordered = [g for g in legend_order if g in gb_present]
            ordered += [g for g in gb_present if g not in ordered]
        elif legend_order_from_sigma_mapping is not None:
            ordered = legend_order_from_sigma_mapping
        elif gb_label_map:
            # Use dict order if user provides an ordered dict
            # If dict is Python 3.7+ (always ordered); preserves order of mapping
            ordered = [g for g in gb_label_map.keys() if g in gb_present]
            ordered += [g for g in gb_present if g not in ordered]
        else:
            ordered = gb_present

        handles = []
        for g in ordered:
            handle_kwargs = dict(
                marker=gb_markers.get(g, default_marker),
                linestyle='',
                markerfacecolor=gb_colors.get(g, "#808080"),
                markeredgecolor=(marker_edgecolor if marker_edgecolor is not None else "none"),
                markersize=legend_markersize,
            )
            if marker_edgewidth is not None:
                handle_kwargs["markeredgewidth"] = marker_edgewidth
            handles.append(Line2D([0], [0], **handle_kwargs))

        labels = [gb_label_map.get(g, g) if gb_label_map else g for g in ordered]

        if handles:
            # Always default to 1 column unless user overrides
            if legend_ncol is None:
                legend_ncol = 1

            main_legend_kwargs = dict(
                frameon=True,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                title=(legend_title if legend_title is not None else gb_col),
                loc=legend_loc,
            )
            if legend_anchor is not None:
                main_legend_kwargs.update(
                    bbox_to_anchor=legend_anchor,
                    bbox_transform=ax.transAxes
                )
            # Add all legend padding kwargs if set
            if legend_labelspacing is not None:
                main_legend_kwargs['labelspacing'] = legend_labelspacing
            if legend_columnspacing is not None:
                main_legend_kwargs['columnspacing'] = legend_columnspacing
            if legend_borderpad is not None:
                main_legend_kwargs['borderpad'] = legend_borderpad
            if legend_handletextpad is not None:
                main_legend_kwargs['handletextpad'] = legend_handletextpad
            if legend_handlelength is not None:
                main_legend_kwargs['handlelength'] = legend_handlelength
            if legend_borderaxespad is not None:
                main_legend_kwargs['borderaxespad'] = legend_borderaxespad
            if legend_titlepad is not None:
                main_legend_kwargs['title_pad'] = legend_titlepad

            main_leg = ax.legend(handles=handles, labels=labels, **main_legend_kwargs)
            ax.add_artist(main_leg)  # keep this when adding a second legend

    # ---- Secondary "style" legend: substitutional vs interstitial ----
    if show_style_legend:
        style_ms = legend_markersize if style_legend_markersize is None else style_legend_markersize
        style_fs = legend_fontsize if style_legend_fontsize is None else style_legend_fontsize

        # filled (substitutional)
        filled_handle_kwargs = dict(
            marker=style_marker,
            linestyle='',
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=style_ms,
        )
        hollow_handle_kwargs = dict(
            marker=style_marker,
            linestyle='',
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=style_ms,
        )
        # Set linewidth for style legend if set by user
        if marker_edgewidth is not None:
            filled_handle_kwargs["markeredgewidth"] = marker_edgewidth
            hollow_handle_kwargs["markeredgewidth"] = marker_edgewidth if marker_edgewidth > 0 else 1.0
        else:
            # For hollow, ensure edge is visible if marker_edgewidth==0
            if marker_edgewidth == 0.0:
                hollow_handle_kwargs["markeredgewidth"] = 1.0

        filled_handle = Line2D([0], [0], **filled_handle_kwargs)
        hollow_handle = Line2D([0], [0], **hollow_handle_kwargs)

        style_handles = [filled_handle, hollow_handle]
        style_labels = list(style_legend_labels)

        style_kwargs = dict(
            frameon=style_legend_frameon,
            fontsize=style_fs,
            ncol=1,
            title=None,
            loc=style_legend_loc,
        )
        if style_legend_anchor is not None:
            style_kwargs.update(
                bbox_to_anchor=style_legend_anchor,
                bbox_transform=ax.transAxes
            )
        # Add style legend padding kwargs if set
        if style_legend_labelspacing is not None:
            style_kwargs['labelspacing'] = style_legend_labelspacing
        if style_legend_columnspacing is not None:
            style_kwargs['columnspacing'] = style_legend_columnspacing
        if style_legend_borderpad is not None:
            style_kwargs['borderpad'] = style_legend_borderpad
        if style_legend_handletextpad is not None:
            style_kwargs['handletextpad'] = style_legend_handletextpad
        if style_legend_handlelength is not None:
            style_kwargs['handlelength'] = style_legend_handlelength
        if style_legend_borderaxespad is not None:
            style_kwargs['borderaxespad'] = style_legend_borderaxespad
        if style_legend_titlepad is not None:
            style_kwargs['title_pad'] = style_legend_titlepad

        ax.legend(handles=style_handles, labels=style_labels, **style_kwargs)

    if tight_layout:
        fig.tight_layout()

    return gb_colors, (fig, ax)
