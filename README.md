# universal_plot_maker_plus_with_subplot

This updated script introduces the ability to use sublots, so you can plot dataframes with more than 3 layers.

## Script

universal_plot_maker_plus.py
```py

#!/usr/bin/env python3
"""
Universal interactive plot maker (generalized)
- Reads a TSV/CSV into pandas
- Exports a standalone HTML with Plotly.js and a functional UI:
  * Axis selection (X/Y from CLI-provided candidates; type-aware)
  * Plot type: bar | scatter | line
  * Group coloring by a chosen column
  * Multiple dropdown filters with CLI defaults
  * Multiple search boxes with CLI defaults
  * Primary + secondary sort (asc/desc)
  * Initial zoom (#bars)
  * Click for details (columns chosen via CLI)
  * TSV export of selected points (lasso/box)
- Duplicate handling: overlay | stack | max | mean | median | first | sum | separate

Example (with your dataframe columns):
    python universal_plot_maker_plus.py \
      --file gene_celltype_table.tsv \
      --out gene_plot.html \
      --plot-type bar \
      --title "Cell type enrichment" \
      --x-choices "Gene name|Gene" \
      --y-choices "avg_nCPM|Enrichment score|log2_enrichment|Enrichment score (tau penalized)|log2_enrichment_penalized" \
      --default-x "Gene name" \
      --default-y "log2_enrichment" \
      --color-col "Cell type group" \
      --filter-cols "Cell type|Cell type group|Cell type class" \
      --filter-defaults "Cell type group=secretory cells" \
      --search-cols "Gene name|Gene" \
      --search-defaults "" \
      --details "Gene|Gene name|Cell type|avg_nCPM|Enrichment score|log2_enrichment|specificity_tau|Cell type group|Cell type class" \
      --initial-zoom 100 \
      --sort-primary "Enrichment score" \
      --sort-primary-order desc \
      --sort-secondary "avg_nCPM" \
      --sort-secondary-order desc \
      --dup-policy mean \
      --show-legend \
      --self-contained

"""

import json
import re
import sys
import argparse
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np
from plotly.graph_objects import Figure, Bar, Scatter

# ------------------------------
# Helpers: I/O & type checks
# ------------------------------

def _infer_sep(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    plu = path.lower()
    if plu.endswith(".tsv") or plu.endswith(".tab"):
        return "\t"
    if plu.endswith(".csv"):
        return ","
    return None

def load_table(path: str, sep: Optional[str]) -> pd.DataFrame:
    if sep is None:
        sep = _infer_sep(path)
    df = pd.read_csv(path, sep=sep if sep else None, engine="python")
    # Normalize column names: strip spaces/newlines
    df.columns = [str(c).strip() for c in df.columns]
    return df

def is_numeric_series(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(s)
    except Exception:
        return False

def coerce_numeric_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return s
    # Loose coercion (commas -> dots), drop "NA"/blanks
    s2 = (
        s.astype(str)
         .str.strip()
         .replace({"NA": None, "N/A": None, "null": None, "None": None, "": None})
         .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s2, errors="coerce")

def safe_str(s) -> str:
    try:
        return str(s)
    except Exception:
        return ""

# ------------------------------
# Duplicate handling
# ------------------------------

def dedupe_rows(
    df: pd.DataFrame,
    key_cols: List[str],
    value_col: Optional[str],
    policy: str
) -> pd.DataFrame:
    """
    Pre-collapse duplicates BEFORE plotting if policy in {max, mean, median, first, sum}.
    If policy == 'overlay': keep all duplicates (single trace, overlap).
    If policy == 'stack': keep duplicates; client will render multiple traces stacked.
    """
    policy = policy.lower()
    if not key_cols or any(k not in df.columns for k in key_cols):
        # Nothing we can do; return as-is unless policy requires collapsing
        return df.copy()

    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if not dup_mask.any():
        return df.copy()

    if policy in ("overlay", "stack"):
        # Plot duplicates as-is, handled in figure building
        return df.copy()

    if value_col is None or value_col not in df.columns:
        raise ValueError(f"--dup-policy '{policy}' requires --default-y (numeric) present in the data.")

    # Numeric coercion only for aggregation
    v = coerce_numeric_column(df, value_col)
    tmp = df.copy()
    tmp["__val__"] = v

    agg_map = {
        "max": "max",
        "mean": "mean",
        "median": "median",
        "first": "first",
        "sum": "sum",
    }
    if policy not in agg_map:
        raise ValueError(f"Unsupported --dup-policy '{policy}'. Use overlay|stack|max|mean|median|first|sum")

    # Aggregate by key
    gb = tmp.groupby(key_cols, dropna=False)
    agg_df = gb.agg({"__val__": agg_map[policy]}).reset_index()

    # Reattach the first row's other columns (for details)
    first_rows = gb.nth(0).reset_index()
    # Drop helper col in first_rows if present
    if "__val__" in first_rows.columns:
        first_rows = first_rows.drop(columns="__val__")

    merged = pd.merge(agg_df, first_rows, on=key_cols, how="left")
    # Replace original value_col with aggregated value for numeric
    merged[value_col] = merged["__val__"]
    merged = merged.drop(columns="__val__")
    return merged

# ------------------------------
# Figure building
# ------------------------------


def build_figure_payload(
    df: pd.DataFrame,
    plot_type: str,
    x_col: str,
    y_col: Optional[str],
    color_col: Optional[str],
    details_cols: List[str],
    top_n: Optional[int],
    sort_primary: Optional[str],
    sort_primary_order: str,
    sort_secondary: Optional[str],
    sort_secondary_order: str,
    initial_zoom: Optional[int],
    title: Optional[str],
    show_legend: bool,
    dup_policy: str,
) -> Tuple[Figure, Dict]:
    """
    Build figure + payload dictionary for client-side UI.
    """

    plot_type = plot_type.lower()
    if plot_type not in ("bar", "scatter", "line"):
        raise ValueError("Unsupported --plot-type. Use bar|scatter|line.")

    if x_col not in df.columns:
        raise ValueError(f"X column '{x_col}' not in data.")

    if plot_type == "bar":
        if y_col is None or y_col not in df.columns:
            raise ValueError("Bar plot requires a numeric Y column via --default-y.")
        y_s = coerce_numeric_column(df, y_col)
        df = df.copy()
        df[y_col] = y_s
    else:
        if y_col is None or y_col not in df.columns:
            raise ValueError(f"{plot_type} plot requires numeric --default-y.")
        x_s = coerce_numeric_column(df, x_col)
        y_s = coerce_numeric_column(df, y_col)
        df = df.copy()
        df[x_col] = x_s
        df[y_col] = y_s

    # ---- Sorting ----
    def _sort_df(dfin: pd.DataFrame) -> pd.DataFrame:
        df2 = dfin.copy()
        cols = []
        asc_flags = []
        if sort_primary and sort_primary in df2.columns:
            cols.append(sort_primary)
            asc_flags.append(sort_primary_order.lower() == "asc")
        if sort_secondary and sort_secondary in df2.columns:
            cols.append(sort_secondary)
            asc_flags.append(sort_secondary_order.lower() == "asc")
        if not cols:
            return df2
        for c in cols:
            if not pd.api.types.is_numeric_dtype(df2[c]):
                df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2 = df2.sort_values(by=cols, ascending=asc_flags, kind="mergesort")
        return df2

    df = _sort_df(df)

    # ---- Colors ----
    if color_col and color_col in df.columns:
        cats = sorted(df[color_col].astype(str).unique())      
        from plotly.express import colors
        palette = (
            colors.qualitative.Dark24 +
            colors.qualitative.Light24 +
            colors.qualitative.Set3
        )
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
        bar_colors = df[color_col].astype(str).map(lambda c: color_map.get(c, "#636EFA")).tolist()
    else:
        color_map = {}
        bar_colors = ["#636EFA"] * len(df)

    # ---- Dedupe (collapse where applicable) ----
    if dup_policy.lower() in ("max", "mean", "median", "first", "sum"):
        key_cols = [x_col]
        df = dedupe_rows(df, key_cols=key_cols, value_col=y_col, policy=dup_policy)
    elif dup_policy.lower() == "separate":
        # Split duplicates into distinct x categories by suffixing with an index
        # Only append "#n" for actual duplicates (groups with size > 1)
        df = df.copy()
        dup_count = df.groupby([x_col], dropna=False)[x_col].transform("size")
        df["_dup_index"] = df.groupby([x_col], dropna=False).cumcount() + 1
        df[x_col] = np.where(
            dup_count > 1,
            df[x_col].astype(str) + "_#" + df["_dup_index"].astype(str),
            df[x_col].astype(str)
        )
        df.drop(columns="_dup_index", inplace=True, errors="ignore")




    # ---- Details (customdata) ALWAYS defined as list-of-lists ----
    detail_cols = [c for c in details_cols if c in df.columns]
    if detail_cols:
        customdata = df[detail_cols].to_numpy().tolist()
    else:
        # Ensure downstream code can index safely
        customdata = [[""] * 0 for _ in range(len(df))]

    fig = Figure()

    if plot_type == "bar":
        categories = df[x_col].astype(str).tolist()
        ticktext = categories[:]
        vals = df[y_col].astype(float).tolist()

        if dup_policy.lower() == "stack":
            from collections import defaultdict

            cat_map_vals = defaultdict(list)
            cat_map_colors = defaultdict(list)
            cat_map_cd = defaultdict(list)

            # ✅ Guard: customdata is always a list; fall back to [] per-row
            for i, cat in enumerate(categories):
                cat_map_vals[cat].append(vals[i])
                cat_map_colors[cat].append(bar_colors[i])
                cd_row = customdata[i] if (customdata and i < len(customdata)) else []
                cat_map_cd[cat].append(cd_row)

            max_dups = max((len(vs) for vs in cat_map_vals.values()), default=0)

            for j in range(max_dups):
                xL, yL, colL, cdL = [], [], [], []
                for cat in categories:
                    arr_vals = cat_map_vals.get(cat, [])
                    arr_cols = cat_map_colors.get(cat, [])
                    arr_cds  = cat_map_cd.get(cat, [])
                    if j < len(arr_vals):
                        xL.append(cat); yL.append(arr_vals[j])
                        colL.append(arr_cols[j]); cdL.append(arr_cds[j])
                    else:
                        xL.append(cat); yL.append(0.0)
                        colL.append("#00000000"); cdL.append([])
                fig.add_trace(Bar(
                    x=xL, y=yL, marker=dict(color=colL),
                    customdata=cdL,
                    hovertemplate=_make_hover_template(detail_cols, orientation="v"),
                    name=f"layer {j+1}"
                ))
            fig.update_layout(barmode="stack")

        else:
            fig.add_trace(Bar(
                x=categories,
                y=vals,
                marker=dict(color=bar_colors),
                customdata=customdata,
                hovertemplate=_make_hover_template(detail_cols, orientation="v"),
            ))  
            if dup_policy.lower() == "separate":
                fig.update_layout(barmode="group")
            else:
                fig.update_layout(barmode="overlay")

        fig.update_layout(
            title=title or f"{y_col} by {x_col}",
            template="plotly_white",
            hovermode="closest",
            showlegend=bool(show_legend),
            dragmode="select",
            margin=dict(l=80, r=40, t=70, b=130),
            bargap=0.2,
        )
        fig.update_xaxes(
            title=dict(text=x_col),
            tickangle=-45,
            categoryorder="array",
            categoryarray=categories,
            tickmode="array",
            tickvals=categories,
            ticktext=ticktext,
            automargin=True,
            title_standoff=10,
        )
        fig.update_yaxes(
            title=dict(text=y_col),
            automargin=True,
            title_standoff=10,
        )

        if initial_zoom is not None:
            n = max(1, min(int(initial_zoom), len(categories)))
            fig.update_xaxes(range=[-0.5, n - 0.5])

    else:
        x = df[x_col].astype(float).tolist()
        y = df[y_col].astype(float).tolist()
        fig.add_trace(Scatter(
            x=x, y=y,
            mode="markers" if plot_type == "scatter" else "lines+markers",
            marker=dict(color=bar_colors),
            customdata=customdata,
            hovertemplate=_make_hover_template(detail_cols, orientation="v"),
        ))
        fig.update_layout(
            title=title or f"{y_col} vs {x_col}",
            template="plotly_white",
            hovermode="closest",
            showlegend=bool(show_legend),
            dragmode="select",
            margin=dict(l=80, r=40, t=70, b=80),
        )
        fig.update_xaxes(title=dict(text=x_col), automargin=True, title_standoff=10)
        fig.update_yaxes(title=dict(text=y_col), automargin=True, title_standoff=10)

    payload = {
        "rows": df.to_dict(orient="records"),
        "detail_cols": detail_cols,
        "color_col": color_col,
        "color_map": color_map,
        "plot_type": plot_type,
        "x_col": x_col,
        "y_col": y_col,
        "title": title or "",
        "show_legend": bool(show_legend),
        "dup_policy": dup_policy.lower(),
        "__x__": x_col,
        "__y__": y_col,
    }
    return fig, payload



def _make_hover_template(detail_cols: List[str], orientation: str = "v") -> str:
    """
    Build hover template referencing Plotly's customdata array correctly.
    """
    lines = []
    if orientation == "v":
        lines.append("**%{x}**")
        lines.append("Value: %{y:.4g}")
    else:
        lines.append("**%{y}**")
        lines.append("Value: %{x:.4g}")

    # Proper Plotly syntax for customdata
    for i, c in enumerate(detail_cols):
        lines.append(f"{c}: %{{customdata[{i}]}}")


    return "<br>".join(lines) + "<extra></extra>"

# ------------------------------
# HTML saving + client UI
# ------------------------------

DETAILS_UI = r"""
<div id="controls" style="margin: 0 0 12px 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">
  <div style="display:flex; flex-wrap:wrap; gap:10px;">
    <div>
      <label><strong>Plot type</strong></label><br>
      <select id="plotTypeSelect" aria-label="Plot type">
        <option value="bar">Bar</option>
        <option value="scatter">Scatter</option>
        <option value="line">Line</option>
      </select>
    </div>
    <div>
      <label><strong>Color by</strong></label><br>
      <select id="colorBySelect" aria-label="Color by"></select>
    </div>
    <div>
      <label><strong>X</strong></label><br>
      <select id="xSelect" aria-label="X column"></select>
    </div>
    <div>
      <label><strong>Y</strong></label><br>
      <select id="ySelect" aria-label="Y column"></select>
    </div>
    <div>
      <label><strong>Bars to show</strong></label><br>
      <input id="barsCount" type="number" value="100" min="1" step="1" style="width:80px;">
    </div>
    <div>
      <label><strong>Duplicate policy</strong></label><br>
      <select id="dupPolicySelect" aria-label="Duplicate policy">
        <option value="overlay">overlay</option>
        <option value="stack">stack</option>
        <option value="max">max</option>
        <option value="mean">mean</option>
        <option value="median">median</option>
        <option value="first">first</option>
        <option value="sum">sum</option>
        <option value="separate">separate</option>
      </select>
    </div>
    <div>
      <label><strong>Primary sort</strong></label><br>
      <select id="sortPrimary"></select>
      <select id="sortPrimaryOrder">
        <option value="asc">asc</option>
        <option value="desc" selected>desc</option>
      </select>
    </div>
    <div>
      <label><strong>Secondary sort</strong></label><br>
      <select id="sortSecondary"></select>
      <select id="sortSecondaryOrder">
        <option value="asc">asc</option>
        <option value="desc" selected>desc</option>
      </select>
    </div>
  </div>

  <hr style="margin:8px 0;">

  <div id="filterRow" style="display:flex; flex-wrap:wrap; gap:10px;"></div>

  <div id="searchRow" style="margin-top:8px; display:flex; flex-wrap:wrap; gap:10px;"></div>

  <div style="margin-top:8px; display:flex; gap:8px;">
    <button id="resetBtn" type="button" aria-label="Reset filters">Reset</button>
    <button id="exportBtn" type="button" aria-label="Export TSV">Export TSV</button>
  </div>
</div>

<div id="rowDetails" style="font-size: 13px; color: #333;">
  Click a point/bar to see details here. Use lasso/box-select to export only selected.
</div>

<script>
(function() {
  try {
    var NL = String.fromCharCode(10);
    var TAB = String.fromCharCode(9);
    var payloadEl = document.getElementById('__payload__');
    var P = payloadEl ? JSON.parse(payloadEl.textContent || '{}') : null;
    if (!P) return;

    var plotEl = document.querySelector('div.js-plotly-plot');
    if (!plotEl) return;

    // UI config pulled from payload "ui_cfg"
    var UI = P.ui_cfg || {};
    var xChoices = UI.x_choices || [];
    var yChoices = UI.y_choices || [];
    var filterCols = UI.filter_cols || [];
    var filterDefaults = UI.filter_defaults || {}; // col -> value
    var searchCols = UI.search_cols || [];
    var searchDefaults = UI.search_defaults || {}; // col -> term
    var sortChoices = UI.sort_choices || []; // allowed sort columns

    var dupPolicy = P.dup_policy || "overlay";
    var dupPolicySelect = document.getElementById('dupPolicySelect');
    dupPolicySelect.value = dupPolicy;

    var firstRender = true;
    var plotTypeSelect = document.getElementById('plotTypeSelect');
    var xSelect = document.getElementById('xSelect');
    var ySelect = document.getElementById('ySelect');
    var barsCount = document.getElementById('barsCount');
    var sortPrimary = document.getElementById('sortPrimary');
    var sortPrimaryOrder = document.getElementById('sortPrimaryOrder');
    var sortSecondary = document.getElementById('sortSecondary');
    var sortSecondaryOrder = document.getElementById('sortSecondaryOrder');
    var filterRow = document.getElementById('filterRow');
    var searchRow = document.getElementById('searchRow');
    var resetEl = document.getElementById('resetBtn');
    var exportEl = document.getElementById('exportBtn');
    var detailsEl = document.getElementById('rowDetails');
    var colorBySelect = document.getElementById('colorBySelect');

    // for subplot
    var CURRENT_ROWS = []; // will mirror the rows used for the last Plotly.react

    
    // ---- Present tissues mini-plot (SVG; no external deps) ----
    // Tiny renderer: value (Y) vs label (X); filters non-numeric Y.
    (function(){
      const ns = 'http://www.w3.org/2000/svg';
      function createSVG(w,h){
        const s=document.createElementNS(ns,'svg');
        s.setAttribute('width',w); s.setAttribute('height',h);
        s.setAttribute('role','img');
        return s; }
      function addText(svg,x,y,txt,o={}){
        const t=document.createElementNS(ns,'text');
        t.setAttribute('x',x);
        t.setAttribute('y',y);
        if(o.anchor)t.setAttribute('text-anchor',o.anchor); 
        if(o.rotate)t.setAttribute('transform',`rotate(${o.rotate},${x},${y})`);
        t.style.font=o.font||'12px sans-serif';
        t.setAttribute('fill',o.color||'#333');
        t.textContent=txt;
        svg.appendChild(t);
        return t; }
      function addLine(svg,x1,y1,x2,y2,o={}){
        const l=document.createElementNS(ns,'line');
        l.setAttribute('x1',x1);
        l.setAttribute('y1',y1);
        l.setAttribute('x2',x2);
        l.setAttribute('y2',y2);
        l.setAttribute('stroke',o.color||'#ccc');
        l.setAttribute('stroke-width',o.width||1);
        if(o.dash)l.setAttribute('stroke-dasharray',o.dash);
        svg.appendChild(l);
        return l; }
      function addRect(svg,x,y,w,h,o={}){
        const r=document.createElementNS(ns,'rect');
        r.setAttribute('x',x); r.setAttribute('y',y);
        r.setAttribute('width',Math.max(0,w));
        r.setAttribute('height',Math.max(0,h));
        r.setAttribute('fill',o.fill||'#4e79a7');
        if(o.rx)r.setAttribute('rx',o.rx);
        if(o.ry)r.setAttribute('ry',o.ry);
        svg.appendChild(r);
        return r; }
      function fmt(v){
        if(Math.abs(v)>=1000)return v.toLocaleString();
        if(Math.abs(v)>=1)return (Math.round(v*1000)/1000).toString();
        return Number(v).toPrecision(3); }
      function parseCell(s){
        if(!s || typeof s!=='string') return [];
        return s.split('&').map(x=>x.trim()).map(row=>{
          if(!row) return null;
          const parts=row.split(':'); if(parts.length<2) return null;
          const y=parseFloat(parts[0].trim()); if(!isFinite(y)) return null;
          let xlbl=parts.slice(1).join(':').trim();
          xlbl=xlbl.replace(/\s*with\s*\d+.*$/i,'').trim(); // strip "with N"
          if(!xlbl) return null;
          return {x: xlbl, y};
        }).filter(Boolean);
      }
      function renderPT(cellString, cfg){
        const container=document.getElementById(cfg.containerId||'present-tissues-plot');
        if(!container) return;
        container.innerHTML='';
        const data=parseCell(cellString);
        if(!data.length){ const d=document.createElement('div'); d.textContent='No data to plot for "Present tissues".'; d.style.color='#666'; d.style.font='13px sans-serif'; container.appendChild(d); return; }

        // 1) Layout
        const height = Number(cfg.height || 340);
        const margin = Object.assign({ top: 40, right: 20, bottom: 70, left: 60 }, cfg.margin || {});
        const innerH = Math.max(10, height - margin.top - margin.bottom);

        // 2) Fixed bar pixels + gap (configurable)
        const BAR_PX  = (typeof cfg.barPixelWidth === 'number' ? cfg.barPixelWidth : 14); // px
        const GAP_PX  = (typeof cfg.barGap === 'number' ? cfg.barGap : 6);               // px between bars

        
        // 3) Size the inner plotting width from the data count
        const numBars = data.length;
        const step = BAR_PX + GAP_PX;                               // center-to-center
        const computedInnerW = Math.max(10, numBars * step);        // px


        // Compute width (respect minWidth if auto)
        const MIN_W = (cfg && typeof cfg.minWidth === 'number') ? cfg.minWidth : 420; // px
        const width = (cfg && (cfg.width === 'auto' || cfg.width === undefined))
          ? Math.max(MIN_W, margin.left + computedInnerW + margin.right)
          : Number((cfg && cfg.width) || Math.max(MIN_W, margin.left + computedInnerW + margin.right));


         // Create SVG BEFORE any drawing calls
        const svg = createSVG(width, height);
        container.appendChild(svg);

        // 4) Scales and helpers
        const innerW = Math.max(10, width - margin.left - margin.right);
        const cats = data.map(d => d.x);
        const maxY = Math.max(...data.map(d => d.y), 0) || 1;   // keep the fallback

        const barW = BAR_PX;
        const xPos = (i) => margin.left + i * step;           // left edge of bar i
        const yPos = (v) => margin.top + (innerH - (v / maxY) * innerH);

        // Title
        if (cfg.title) {
          addText(svg, margin.left + innerW / 2, Math.max(18, margin.top * 0.6), cfg.title, {
            anchor: 'middle',
            font: 'bold 14px sans-serif',
            color: '#222'
          });
        }

        // axes
        addLine(svg, margin.left, margin.top, margin.left, margin.top + innerH, { color: '#999' });
        addLine(svg, margin.left, margin.top + innerH, margin.left + innerW, margin.top + innerH, { color: '#999' });

        // y ticks
        const ticks=5, stepV=maxY/ticks;
        for(let i=0;i<=ticks;i++){ const v=i*stepV, y=yPos(v); addLine(svg, margin.left, y, margin.left+innerW, y, {color:'#eee'}); addText(svg, margin.left-8, y+4,  fmt(v), {anchor:'end', font:'11px sans-serif', color:'#555'}); }
        if(cfg.yLabel){ addText(svg, 14, margin.top+innerH/2, cfg.yLabel, {anchor:'middle', rotate:-90, font:'12px sans-serif', color:'#333'}); }
        const rotate=typeof cfg.xLabelRotation==='number'? cfg.xLabelRotation : -40;
        data.forEach((d,i)=>{ const h=(d.y/maxY)*innerH, y=yPos(d.y), x=xPos(i); addRect(svg, x, y, barW, h, {fill: cfg.barColor||'#4e79a7', rx:2, ry:2}); addText(svg,  x+barW/2, margin.top+innerH+16, d.x, {anchor: rotate? 'end':'middle', rotate: rotate||0, font:'11px sans-serif', color:'#555'}); });
        if(cfg.xLabel){ addText(svg, margin.left+innerW/2, margin.top+innerH+margin.bottom-10, cfg.xLabel, {anchor:'middle', font:'12px sans-serif', color:'#333'}); }
      }
      window.PTPlot = { render: renderPT, parse: parseCell };
    })();

    // ---- Create/insert subplot container per UI.pt settings ----
    
    // ---- Create a side-by-side layout: details (left) + subplot (right) ----
    (function(){
      if (!UI || !UI.pt || !UI.pt.enable) return;

      var details = document.getElementById('rowDetails');
      if (!details) return;

      // 1) Create a wrapper around details + subplot
      var wrap = document.createElement('div');
      wrap.id = 'detailsWrap';
      wrap.style.display = 'flex';
      wrap.style.gap = '16px';
      wrap.style.alignItems = 'flex-start';
      wrap.style.marginTop = '12px';

      // Replace the details node in place with the wrapper, then re-append details into it
      var parent = details.parentNode;
      parent.insertBefore(wrap, details);
      parent.removeChild(details);

      // 2) Left column: the existing details table
      var left = details;
      left.style.flex = '1 1 auto';   // details column takes remaining width
      left.style.minWidth = '320px';  // ensure room for the table
      left.style.overflow = 'auto';

      // 3) Right column: subplot container
      var right = document.createElement('div');
      var id = UI.pt.containerId || 'present-tissues-plot';
      right.id = id;

      // Right column sizing rules:
      // - keep a minimum width so title/labels won't get cut
      // - allow horizontal scroll when the SVG is wider (constant bar width)
      var minW = (typeof UI.pt.minWidth === 'number') ? UI.pt.minWidth : 420;
      right.style.minWidth = String(minW) + 'px';
      right.style.maxWidth = '50vw';      // don’t let it consume the entire page
      right.style.overflowX = 'auto';
      right.style.padding = '4px 0';
      right.style.border = '1px solid #eee';
      right.style.borderRadius = '6px';
      right.style.background = 'white';

      // Optional little header to mirror your subplot title (purely visual)
      if (UI.pt && UI.pt.title) {
        var hdr = document.createElement('div');
        hdr.textContent = UI.pt.title;
        hdr.style.font = '600 13px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif';
        hdr.style.color = '#222';
        hdr.style.margin = '0 0 6px 8px';
        right.appendChild(hdr);
      }

      // 4) Append in order: [details left] [subplot right]
      wrap.appendChild(left);
      wrap.appendChild(right);
    })();



    // Selected points tracking (by a canonical composite key)
    var selectedKeys = [];

    // Build select options helpers
    function fillSelect(sel, items, defaultValue) {
      sel.innerHTML = '';
      items.forEach(function(it){
        var opt = document.createElement('option');
        opt.value = it;
        opt.textContent = it;
        sel.appendChild(opt);
      });
      if (defaultValue && items.indexOf(defaultValue) >= 0) {
        sel.value = defaultValue;
      } else if (items.length > 0) {
        sel.value = items[0];
      }
    }

    // Init axis selectors
    fillSelect(plotTypeSelect, ['bar','scatter','line'], P.plot_type || 'bar');
    fillSelect(xSelect, xChoices, P.x_col || (xChoices.length ? xChoices[0] : ''));
    fillSelect(ySelect, yChoices, P.y_col || (yChoices.length ? yChoices[0] : ''));

    // Init barsCount from initial zoom
    if (UI.initial_zoom && UI.initial_zoom > 0) {
      barsCount.value = UI.initial_zoom;
    }

    // Sort choices

    var prevSecondary = sortSecondary.value;
    var prevSecondaryOrder = sortSecondaryOrder.value;

    var prevPrimary = sortPrimary.value;
    var prevPrimaryOrder = sortPrimaryOrder.value;
    
    
    // Always ensure Y column is available as a sort option
    var dynamicSortChoices = [ySelect.value].concat(
        sortChoices.filter(c => c !== ySelect.value)
    );

    // Fill sort menus with dynamic choices
    fillSelect(sortPrimary, dynamicSortChoices, sortPrimary.value || ySelect.value);
    fillSelect(sortSecondary, dynamicSortChoices, sortSecondary.value || '(none)');

    
    //fillselect for color  by
    var prevColorBy = colorBySelect.value;
    fillSelect(colorBySelect, (UI.color_choices || []), prevColorBy || UI.color_default || '');
    fillSelect(colorBySelect, ['(none)'].concat(UI.color_choices || []),
           UI.color_default || '(none)');
    var colorCol = colorBySelect.value;
    if (colorCol === '(none)') colorCol = null;

    if (firstRender) {
        // Use CLI defaults on first load
        sortPrimaryOrder.value = UI.sort_primary_order || 'desc';
        sortSecondaryOrder.value = UI.sort_secondary_order || 'desc';
    } else {
        // Use user-selected values after first render
        sortPrimaryOrder.value = prevPrimaryOrder || 'desc';
        sortSecondaryOrder.value = prevSecondaryOrder || 'desc';
    }


    // Build filter dropdowns
    var filterEls = {}; // col -> select
    filterCols.forEach(function(col){
      var wrap = document.createElement('div');
      var label = document.createElement('label');
      label.innerHTML = '<strong>' + col + '</strong>';
      var sel = document.createElement('select');
      sel.setAttribute('aria-label', 'Filter: ' + col);
      // Collect unique values from rows
      var uniq = new Set();
      (P.rows || []).forEach(function(r){
        var v = (r[col] != null ? String(r[col]) : '');
        uniq.add(v);
      });
      var values = Array.from(uniq).sort();
      // Special 'All' option
      var optAll = document.createElement('option');
      optAll.value = '__ALL__';
      optAll.textContent = 'All';
      sel.appendChild(optAll);
      values.forEach(function(v){
        var opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        sel.appendChild(opt);
      });
      // Default
      var def = filterDefaults[col];
      sel.value = (def && values.indexOf(def) >= 0) ? def : '__ALL__';
      wrap.appendChild(label);
      wrap.appendChild(document.createElement('br'));
      wrap.appendChild(sel);
      filterRow.appendChild(wrap);
      filterEls[col] = sel;
    });

    // Build search inputs
    var searchEls = {}; // col -> input
    searchCols.forEach(function(col){
      var wrap = document.createElement('div');
      var label = document.createElement('label');
      label.innerHTML = '<strong>Search in ' + col + '</strong>';
      var box = document.createElement('input');
      box.type = 'text';
      box.placeholder = 'Type to filter...';
      box.setAttribute('aria-label', 'Search: ' + col);
      box.value = (searchDefaults[col] || '');
      wrap.appendChild(label);
      wrap.appendChild(document.createElement('br'));
      wrap.appendChild(box);
      searchRow.appendChild(wrap);
      searchEls[col] = box;
    });

    // Selection events
    plotEl.on('plotly_selected', function(ev) {
      var pts = (ev && ev.points) ? ev.points : [];
      // Use x (or y for horizontal) as canonical key; for scatter, use JSON of (x,y)
      selectedKeys = pts.map(function(p){
        if ((P.plot_type || 'bar') === 'bar') {
          return String(p.x);
        } else {
          return JSON.stringify({x:p.x, y:p.y});
        }
      });
    });
    plotEl.on('plotly_deselect', function(){ selectedKeys = []; });

    // Click details
    plotEl.on('plotly_click', function(ev) {
      try {
        var p = (ev && ev.points && ev.points[0]) ? ev.points[0] : null;
        if (!p) return;
        var cd = p.customdata || [];
        var cols = P.detail_cols || [];
        if (!cols.length) {
          detailsEl.textContent = 'No detail columns configured.';
          return;
        }
        var html = '<table style="border-collapse:collapse;">';
        for (var i = 0; i < cols.length; i++) {
          var v = cd[i];
          var vv = (typeof v === 'number') ? (Number.isFinite(v) ? v.toPrecision(4) : String(v)) : String(v != null ? v : '');
          html += '<tr><th style="text-align:left;padding:4px 8px;">' + cols[i] + '</th>' +
                  '<td style="padding:4px 8px;">' + vv + '</td></tr>';
        }
        html += '</table>';
        detailsEl.innerHTML = html;
        
        
    // --- Subplot: Present tissues ---
    try {
      if (UI.pt && UI.pt.enable) {
        var idx = Array.isArray(cd) ? cd[cols.length] : null; // use click-scope 'cols'
        if (idx != null && Number.isFinite(idx) && idx >= 0 && idx < CURRENT_ROWS.length) {
          var row = CURRENT_ROWS[idx];
          var colName = UI.pt.col || 'Present tissues';
          var cell = (row && (row[colName] != null ? String(row[colName]) : (row['Present_tissues'] != null ? String(row['Present_tissues']) : ''))) || '';
          window.PTPlot.render(cell, {
            containerId: UI.pt.containerId || 'present-tissues-plot',
            title: UI.pt.title || ('Present tissues — ' + (row['Gene name'] || row['Gene'] || 'selected')),
            xLabel: UI.pt.xLabel || 'Tissue',
            yLabel: UI.pt.yLabel || 'Score',
            barColor: UI.pt.color || '#4e79a7',
            height: UI.pt.height || 340,
            width: (UI.pt.width === undefined ? 'auto' : UI.pt.width),
            xLabelRotation: (typeof UI.pt.rotate === 'number' ? UI.pt.rotate : -40),
            margin: { top: 40, right: 20, bottom: 70, left: 60 },

            // NEW (optional) — only affects subplot:
            barPixelWidth: 14,  // pick your favorite thickness
            barGap: 6,           // gap between bars

            // NEW: enforce a minimum SVG width so title/labels don't get cut
            minWidth: (typeof UI.pt.minWidth === 'number' ? UI.pt.minWidth : 420)


          });
        }
      }
    } catch (e2) { console.error('PTPlot render error:', e2); }


      } catch (e) { console.error('click -> details error:', e); }
    });

    // Filtering function
    function applyFilters(rows) {
      // Dropdown filters
      var filtered = rows.filter(function(r){
        for (var col in filterEls) {
          var selVal = filterEls[col].value;
          if (selVal === '__ALL__') continue;
          var rv = (r[col] != null ? String(r[col]) : '');
          if (rv !== selVal) return false;
        }
        // Search filters (contains, case-insensitive)
        for (var scol in searchEls) {
          var term = (searchEls[scol].value || '').toLowerCase().trim();
          if (term === '') continue;
          var rv2 = String(r[scol] != null ? r[scol] : '').toLowerCase();
          if (rv2.indexOf(term) === -1) return false;
        }
        return true;
      });
      return filtered;
    }
    // Apply client dedupe function
    
    function applyClientDedupe(rows, policy, xcol, colorCol) {
      policy = policy || "overlay";
      // Implement "separate" dynamically in the browser:
      // suffix duplicates as _#1, _#2, ... so each becomes its own category.
      if (policy === "separate") {
        // Shallow copy so we don't mutate P.rows
        var out = rows.map(r => Object.assign({}, r));
        // Count occurrences per X (BEFORE renaming)
        var counts = new Map();
        for (var r0 of out) {
          var k = String(r0[xcol]); // number per X only
          counts.set(k, (counts.get(k) || 0) + 1);
        }
        // For duplicated X's, add a running index; leave singletons untouched
        var seen = new Map(); // key -> count
        for (var i = 0; i < out.length; i++) {
          var r = out[i];
          var key = String(r[xcol]); // number per X only
          if ((counts.get(key) || 0) > 1) {
            var cnt = (seen.get(key) || 0) + 1;
            seen.set(key, cnt);
            r[xcol] = key + "_#" + String(cnt);
         } else {
            r[xcol] = key; // keep singleton label
          }
        }
        return out;
      }


      // overlay/stack: no collapsing, keep categories untouched
      if (policy === "overlay" || policy === "stack")
        return rows;
      // Collapse duplicates for aggregate modes (max/mean/median/first/sum)
      function key(r) {
        return String(r[xcol]);
      }
      var groups = {};
      for (var r of rows) {
        var k = key(r);
        if (!(k in groups)) groups[k] = [];
        groups[k].push(r);
      }
      var valCol = ySelect.value;
      var output = [];
      for (var k in groups) {
        var g = groups[k];
        var base = g[0];
        var nums = g.map(r => Number(r[valCol])).filter(n => !Number.isNaN(n));
        if (nums.length === 0) nums = [0];
        var newVal;
        switch (policy) {
          case "first":  newVal = nums[0]; break;
          case "max":    newVal = Math.max(...nums); break;
          case "mean":   newVal = nums.reduce((a,b)=>a+b,0)/nums.length; break;
          case "median": nums.sort((a,b)=>a-b);
                         newVal = nums[Math.floor(nums.length/2)]; break;
          case "sum":    newVal = nums.reduce((a,b)=>a+b,0); break;
          default:       newVal = nums[0];
        }
        var newRow = JSON.parse(JSON.stringify(base));
        newRow[valCol] = newVal;
        output.push(newRow);
      }
      return output;
    }


    // Sort function
    function sortRows(rows, pcol, pord, scol, sord) {
    
    // Normalize sort column names
    if (pcol) pcol = pcol.trim();
    if (scol) scol = scol.trim();

      if (!Array.isArray(rows)) return [];
      var data = rows.slice();
      function asNum(v) {
        if (typeof v === 'number') return v;
        if (typeof v === 'string') {
          var s = v.replace(',', '.');
          var n = parseFloat(s);
          if (!Number.isNaN(n)) return n;
        }
        return v;
      }
      
    
    
    function cmp(a, b, col, ord) {
      let asc = (ord || 'asc').toLowerCase() === 'asc';

      let va = a[col];
      let vb = b[col];

      // Convert if possible
      let na = Number(va);
      let nb = Number(vb);

      let a_is_num = !Number.isNaN(na);
      let b_is_num = !Number.isNaN(nb);

      // Case 1: both numeric → numeric compare
      if (a_is_num && b_is_num) {
        return asc ? (na - nb) : (nb - na);
      }

      // Case 2: only one is numeric → treat both as strings
      // This avoids inconsistent ordering
      let sa = String(va);
      let sb = String(vb);

      // Case 3: pure string sort A-Z or Z-A
      return asc ? sa.localeCompare(sb) : sb.localeCompare(sa);
    }

      
      data.sort(function(a, b){
        if (
          pcol &&
          Object.prototype.hasOwnProperty.call(a, pcol) &&
          Object.prototype.hasOwnProperty.call(b, pcol)
        ) {

          var c = cmp(a,b,pcol,pord);
          if (c !== 0) return c;
        }
        if (
          scol &&
          scol !== '(none)' &&
          scol.trim() !== '' &&
          Object.prototype.hasOwnProperty.call(a, scol) &&
          Object.prototype.hasOwnProperty.call(b, scol)
        )
        {
          var d = cmp(a,b,scol,sord);
          if (d !== 0) return d;
        }
        return 0;
      });

      return data;
    }

    // Build full hover template dynamically (axis choice may change)
    function makeHover(detailCols, orientation) {
      var lines = [];
      if (orientation === 'v') {
        lines.push("**%{x}**");
        lines.push("Value: %{y:.4g}");
      } else {
        lines.push("**%{y}**");
        lines.push("Value: %{x:.4g}");
      }
      for (var i = 0; i < detailCols.length; i++) {
        lines.push(detailCols[i] + ": %{customdata[" + i + "]}");
      }
      return lines.join("<br>") + "<extra></extra>";
    }

    // Render function: rebuild single/stacked traces
    function render() {
      var rowsAll = Array.isArray(P.rows) ? P.rows.slice() : [];
      var rowsF = applyFilters(rowsAll);

      var ptype = plotTypeSelect.value || 'bar';
      var xcol = xSelect.value || '';
      var ycol = ySelect.value || '';
      var currentDupPolicy = dupPolicySelect.value || "overlay";

      var pcol = sortPrimary.value.trim();
      var scol = sortSecondary.value.trim();


      // First apply duplicate policy → modifies Y to sum/mean/etc
      rowsF = applyClientDedupe(rowsF, currentDupPolicy, xcol, colorCol);

      // Restore CLI default sorting ON FIRST RENDER ONLY
      if (firstRender) {
          // Apply CLI's default primary sort column & order
          sortPrimary.value = UI.sort_primary || '';
          sortPrimaryOrder.value = UI.sort_primary_order || 'desc';

          // Apply CLI's default secondary sort, if any
          sortSecondary.value = UI.sort_secondary || '(none)';
          sortSecondaryOrder.value = UI.sort_secondary_order || 'desc';
      }


      // THEN sort using the aggregated values
      rowsF = sortRows(
          rowsF,
          sortPrimary.value,
          sortPrimaryOrder.value,
          sortSecondary.value,
          sortSecondaryOrder.value
      );

      
      // Bars count (viewport only; do NOT slice data)
      var nBars = parseInt(barsCount.value || '0', 10);
      if (!Number.isFinite(nBars) || nBars <= 0) nBars = null;


      var detailCols = P.detail_cols || [];
      var colorCol = colorBySelect.value || null;
      if (colorCol === '(none)') colorCol = null;

     
      

      // custom color by

      // Build colors for current colorCol (client-side)
      var colorMap = {}; 
      var palette = [
          "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
          "#393b79", "#5254a3", "#6b6ecf", "#9c9ede",
          "#c6d31f", "#bd9e39", "#e7ba52", "#e7cb94",
          "#843c39", "#ad494a", "#d6616b", "#e7969c",
          "#7b4173", "#a55194", "#ce6dbd", "#de9ed6"
      ];
      // Collect unique categories from the rows to be plotted (rowsF)
      if (colorCol) {
          // collect all possible categories from ALL rows (not only filtered)
          var cats = Array.from(new Set(P.rows.map(r =>
              r[colorCol] != null ? String(r[colorCol]) : ""
          ))).sort();
        cats.sort();
        for (var i = 0; i < cats.length; i++) {
          colorMap[cats[i]] = palette[i % palette.length];
        }
      }

      // Build per-point colors
      var colors = [];
      for (var i = 0; i < rowsF.length; i++) {
        var r = rowsF[i];
        var gval = colorCol ? String(r[colorCol]) : null;
        var c = (colorCol && colorMap[gval]) ? colorMap[gval] : '#636EFA';
        colors.push(c);
      }
      

      // Collect data arrays
      var x = [], y = [], customdata = [];
      var categories = [], ticktext = [];

      // Build x,y/customdata based on plot type
      if (ptype === 'bar') {
        // X categorical, Y numeric
        for (var i = 0; i < rowsF.length; i++){
          var r = rowsF[i];
          var xc = String(r[xcol]);
          var yc = r[ycol];
          if (typeof yc === 'string') {
            var s = yc.replace(',', '.');
            yc = parseFloat(s);
          }
          x.push(xc);
          y.push(yc);
          categories.push(xc);
          ticktext.push(xc);
          var cd = [];
          for (var j = 0; j < detailCols.length; j++) cd.push(r[detailCols[j]]);
          cd.push(i); // <-- append row index for subplot
          customdata.push(cd);
        }

        var traces = [];
        if (currentDupPolicy === 'stack') {
          // group by category, stack layers
          var catMap = new Map();
          for (var i = 0; i < x.length; i++) {
            var cat = x[i];
            if (!catMap.has(cat)) catMap.set(cat, []);
            catMap.get(cat).push({y:y[i], color:colors[i], cd:customdata[i]});
          }
          // max depth
          var maxDepth = 0;
          catMap.forEach(function(arr){ if (arr.length > maxDepth) maxDepth = arr.length; });

          for (var layer = 0; layer < maxDepth; layer++) {
            var xL = [], yL = [], colL = [], cdL = [];
            for (var i = 0; i < categories.length; i++) {
              var cat = categories[i];
              var arr = catMap.get(cat) || [];
              if (layer < arr.length) {
                xL.push(cat); yL.push(arr[layer].y);
                colL.push(arr[layer].color);
                cdL.push(arr[layer].cd);
              } else {
                xL.push(cat); yL.push(0);
                colL.push('#00000000');
                cdL.push([]);
              }
            }
            traces.push({
              type: 'bar',
              x: xL,
              y: yL,
              customdata: cdL,
              marker: { color: colL },
              hovertemplate: makeHover(detailCols, 'v'),
              name: ('layer ' + (layer+1))
            });
          }

          CURRENT_ROWS = rowsF.slice(); // keep the order used for plotting
          
          Plotly.react(plotEl, traces, {
            title: P.title || (ycol + ' by ' + xcol),
            template: 'plotly_white',
            hovermode: 'closest',
            showlegend: !!P.show_legend,
            dragmode: 'select',
            margin: {l:80, r:40, t:70, b:130},
            xaxis: {
              title: { text: xcol },
              automargin: true,
              title_standoff: 10,
              tickangle: -45,
              categoryorder: 'array',
              categoryarray: categories,
              tickmode: 'array',
              tickvals: categories,
              ticktext: ticktext,
              // NEW:
              range: (nBars ? [-0.5, Math.min(categories.length, nBars) - 0.5] : undefined),
            },
            yaxis: { title: { text: ycol }, automargin: true, title_standoff: 10 },
            barmode: 'stack'
          });
        } else {
          var trace = {
            type: 'bar',
            x: x,
            y: y,
            customdata: customdata,
            marker: { color: colors },
            selected: { marker: { opacity: 1.0 } },
            unselected: { marker: { opacity: 0.5 } },
            hovertemplate: makeHover(detailCols, 'v'),
          };
          
          CURRENT_ROWS = rowsF.slice();  // keep the order used for plotting
          
          Plotly.react(plotEl, [trace], {
            title: P.title || (ycol + ' by ' + xcol),
            template: 'plotly_white',
            hovermode: 'closest',
            showlegend: !!P.show_legend,
            dragmode: 'select',
            margin: {l:80, r:40, t:70, b:130},
            xaxis: {
              title: { text: xcol },
              automargin: true,
              title_standoff: 10,
              tickangle: -45,
              categoryorder: 'array',
              categoryarray: categories,
              tickmode: 'array',
              tickvals: categories,
              ticktext: ticktext,
              // NEW:
              range: (nBars ? [-0.5, Math.min(categories.length, nBars) - 0.5] : undefined),
            },
            yaxis: { title: { text: ycol }, automargin: true, title_standoff: 10 },
            barmode: (currentDupPolicy === 'separate' ? 'group' : 'overlay')
          });
        }

      } else {
        // scatter/line: both numeric
        for (var i = 0; i < rowsF.length; i++){
          var r = rowsF[i];
          var xv = r[xcol];
          var yv = r[ycol];
          if (typeof xv === 'string') { var s1 = xv.replace(',', '.'); xv = parseFloat(s1); }
          if (typeof yv === 'string') { var s2 = yv.replace(',', '.'); yv = parseFloat(s2); }
          x.push(xv);
          y.push(yv);
          var cd = [];
          for (var j = 0; j < detailCols.length; j++) cd.push(r[detailCols[j]]);
          cd.push(i); // <-- append row index for subplot
          customdata.push(cd);
        }
        var trace2 = {
          type: 'scatter',
          mode: (ptype === 'scatter' ? 'markers' : 'lines+markers'),
          x: x,
          y: y,
          customdata: customdata,
          marker: { color: colors },
          hovertemplate: makeHover(detailCols, 'v'),
        };

        CURRENT_ROWS = rowsF.slice();  // keep the order used for plotting

        Plotly.react(plotEl, [trace2], {
          title: P.title || (ycol + ' vs ' + xcol),
          template: 'plotly_white',
          hovermode: 'closest',
          showlegend: !!P.show_legend,
          dragmode: 'select',
          margin: {l:80, r:40, t:70, b:80},
          xaxis: { title: { text: xcol }, automargin: true, title_standoff: 10 },
          yaxis: { title: { text: ycol }, automargin: true, title_standoff: 10 },
        });
      }
      firstRender = false;
    }

    // Bind changes
    plotTypeSelect.addEventListener('change', render);
    xSelect.addEventListener('change', render);
    ySelect.addEventListener('change', render);
    barsCount.addEventListener('input', render);
    sortPrimary.addEventListener('change', render);
    sortPrimaryOrder.addEventListener('change', render);
    sortSecondary.addEventListener('change', render);
    sortSecondaryOrder.addEventListener('change', render);
    dupPolicySelect.addEventListener('change', render);
    Object.values(filterEls).forEach(function(sel){ sel.addEventListener('change', render); });
    Object.values(searchEls).forEach(function(box){ box.addEventListener('input', render); });
    colorBySelect.addEventListener('change', render);

    resetEl.addEventListener('click', function(){
      plotTypeSelect.value = (P.plot_type || 'bar');
      xSelect.value = (P.x_col || (xChoices.length ? xChoices[0] : ''));
      ySelect.value = (P.y_col || (yChoices.length ? yChoices[0] : ''));
      barsCount.value = (UI.initial_zoom || 100);
      sortPrimary.value = (UI.sort_primary || '');
      sortPrimaryOrder.value = (UI.sort_primary_order || 'desc');
      sortSecondary.value = (UI.sort_secondary || '(none)');
      sortSecondaryOrder.value = (UI.sort_secondary_order || 'desc');
      for (var col in filterEls) {
        var def = filterDefaults[col];
        var sel = filterEls[col];
        if (def && Array.from(sel.options).map(function(o){return o.value;}).indexOf(def) >= 0) {
          sel.value = def;
        } else {
          sel.value = '__ALL__';
        }
      }
      for (var scol in searchEls) {
        searchEls[scol].value = (searchDefaults[scol] || '');
      }
      selectedKeys = [];
      firstRender = false;
      render();
    });

    exportEl.addEventListener('click', function(){
      // Export TSV (filtered OR selected)
      var rowsAll = Array.isArray(P.rows) ? P.rows.slice() : [];
      var rowsF = applyFilters(rowsAll);

      // if selection exists, filter by selection set
      if (selectedKeys && selectedKeys.length > 0) {
        var sset = new Set(selectedKeys.map(String));
        rowsF = rowsF.filter(function(r){
          // bar: key is X category; scatter: JSON of (x,y)
          var ptype = plotTypeSelect.value || 'bar';
          if (ptype === 'bar') {
            var key = String(r[xSelect.value || P.x_col]);
            return sset.has(key);
          } else {
            // best-effort: exact match on (x,y) pair
            var xv = r[xSelect.value || P.x_col];
            var yv = r[ySelect.value || P.y_col];
            var key = JSON.stringify({x:xv, y:yv});
            return sset.has(key);
          }
        });
      }

      var cols = P.detail_cols || (rowsF.length ? Object.keys(rowsF[0]) : []);
      var header = cols.join(TAB);
      var lines = rowsF.map(function(r){
        return cols.map(function(c){ return (r[c] != null ? String(r[c]) : ''); }).join(TAB);
      });
      var tsv = [header].concat(lines).join(NL);
      var blob = new Blob([tsv], { type: 'text/tab-separated-values;charset=utf-8' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'export.tsv';
      a.click();
      URL.revokeObjectURL(a.href);
    });

    // Initial render
    render();

  } catch (e) {
    console.error('UI init error:', e);
  }
})();
</script>
"""
import re
def save_html(fig: Figure, payload: Dict, ui_cfg: Dict, out_path: str, self_contained: bool, lang: str = "en"):
    # Build plotly HTML
    html = fig.to_html(full_html=True, include_plotlyjs="cdn" if not self_contained else True)
    html = html.lstrip("\ufeff\r\n\t")
    if not html.startswith("<!DOCTYPE html>"):
        html = "<!DOCTYPE html>\n" + html

    # Ensure lang attribute
    if not re.search(r"(?is)<html[^>]*>", html):
        html = f'<!DOCTYPE html>\n<html lang="{lang}">\n</html>'
    if not re.search(r"(?is)<html[^>]*\blang\s*=", html):
        html = re.sub(r"(?is)<html(\s*)>", f'<html lang="{lang}"\\1>', html, count=1)

    # Inject basic meta & title if needed
    head_inject = (
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{payload.get("title") or "Interactive plot"}</title>\n'
    )
    if re.search(r"(?is)<head\s*>", html):
        html = re.sub(r"(?is)<head\s*>", "<head>\n" + head_inject, html, count=1)
    else:
        html = re.sub(r"(?is)(<html[^>]*>)", r"\1\n<head>\n" + head_inject + "</head>\n", html, count=1)

    # Attach payload JSON + UI
    payload_json = json.dumps({**payload, "ui_cfg": ui_cfg}, ensure_ascii=False)
    payload_json = payload_json.replace("</script", "<\\/script")  # safe close
    payload_script = f'<script type="application/json" id="__payload__">{payload_json}</script>\n'

    # Insert our controls + client script before </body>
    
    

    # ... inside save_html(...)
    if re.search(r"(?is)</body>", html):
        html = re.sub(
            r"(?is)</body>",
            lambda m: payload_script + DETAILS_UI + "\n</body>",
            html,
            count=1
        )
    else:
        html = html + "\n" + payload_script + DETAILS_UI


    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

# ------------------------------
# CLI
# ------------------------------

def parse_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    # support both '|' and ',' separators
    items = re.split(r"[|,]", s)
    return [it.strip() for it in items if it.strip()]

def parse_kv_defaults(s: Optional[str]) -> Dict[str, str]:
    """
    Parse "col=value; col=value" pairs; supports ';' or '|' as pair separators.
    """
    if not s:
        return {}
    pairs = re.split(r"[;|]", s)
    out = {}
    for p in pairs:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser(
        "Universal interactive plot maker (generalized)"
    )
    ap.add_argument("--file", "-f", required=True, help="Input table (TSV/CSV)")
    ap.add_argument("--out", "-o", default="interactive_plot.html", help="Output HTML file")
    ap.add_argument("--sep", help="Field separator (auto by extension if omitted)")
    ap.add_argument("--plot-type", default="bar", choices=["bar","scatter","line"], help="Plot type")
    ap.add_argument("--title", default="", help="Figure title")
    ap.add_argument("--color-col", help="Column for group coloring")
    ap.add_argument("--color-choices",help="Columns allowed for coloring (pipe or comma separated). If omitted, any column can be chosen.")
    ap.add_argument("--x-choices", help="Columns allowed for X axis (bar categorical; scatter/line numeric). Use '|' or ',' to separate.")
    ap.add_argument("--y-choices", help="Columns allowed for Y axis (numeric). Use '|' or ',' to separate.")
    ap.add_argument("--default-x", help="Default X column at load")
    ap.add_argument("--default-y", help="Default Y column at load")
    ap.add_argument("--filter-cols", help="Columns to add dropdown filters (pipe or comma separated)")
    ap.add_argument("--filter-defaults", help="Default selection for filters: 'col=value;col=value'")
    ap.add_argument("--search-cols", help="Columns to add search boxes")
    ap.add_argument("--search-defaults", help="Default values for search boxes: 'col=term;col=term'")
    ap.add_argument("--details", default="*", help="Columns to include in hover/details/export. '*' means all cols.")
    ap.add_argument("--initial-zoom", type=int, default=100, help="Initial number of bars/points to show")
    ap.add_argument("--sort-primary", help="Primary sort column")
    ap.add_argument("--sort-primary-order", default="desc", choices=["asc","desc"], help="Primary sort order")
    ap.add_argument("--sort-secondary", help="Secondary sort column (optional)")
    ap.add_argument("--sort-secondary-order", default="desc", choices=["asc","desc"], help="Secondary sort order")
    ap.add_argument("--dup-policy", default="overlay",
        choices=["overlay","stack","max","mean","median","first","sum","separate"],
        help="Duplicate policy: overlay/stack for visual; max/mean/median/first/sum collapse pre-plot; separate splits duplicates           into distinct bars.")
    ap.add_argument("--show-legend", action="store_true", help="Show legend (hidden otherwise)")
    ap.add_argument("--self-contained", action="store_true", help="Embed plotly.js for offline HTML")
    ap.add_argument("--lang", default="en", help="HTML lang attribute (e.g., 'en', 'en-CA')")

    # --- Present tissues subplot (PT) options ---
    ap.add_argument("--pt-enable", action="store_true",
                    help="Enable the Present tissues subplot rendered on bar click")
    ap.add_argument("--pt-col", default="Present tissues",
                    help="Column that stores the sub-data (e.g., 'Present tissues')")
    ap.add_argument("--pt-title", default="Present tissues",
                    help="Subplot title")
    ap.add_argument("--pt-x-label", default="Tissue",
                    help="Subplot X axis label")
    ap.add_argument("--pt-y-label", default="Score",
                    help="Subplot Y axis label")
    ap.add_argument("--pt-color", default="#4e79a7",
                    help="Subplot bar color (CSS color)")
    ap.add_argument("--pt-height", type=int, default=340,
                    help="Subplot height in pixels")
    ap.add_argument("--pt-width", default="auto",
                    help="Subplot width in pixels or 'auto'")
    ap.add_argument("--pt-rotate", type=int, default=-40,
                    help="X tick label rotation in degrees")
    ap.add_argument("--pt-container-id", default="present-tissues-plot",
                    help="HTML id of the div container where the subplot will render")
    ap.add_argument("--pt-location",
                    choices=["above-controls","below-controls","above-details","below-details"],
                    default="below-details",
                    help="Where to insert the subplot container in the HTML")
    ap.add_argument("--pt-min-width", type=int, default=420,
                    help="Minimum width (px) for the subplot SVG/column to avoid clipped titles/labels")



    args = ap.parse_args()

    # Load data
    try:
        df = load_table(args.file, sep=args.sep)
    except Exception as e:
        print(f"[error] Failed to load '{args.file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare UI configs
    x_choices = parse_list(args.x_choices) or list(df.columns)
    y_choices = parse_list(args.y_choices) or [c for c in df.columns if is_numeric_series(df[c])]
    filter_cols = parse_list(args.filter_cols)
    search_cols = parse_list(args.search_cols)
    filter_defaults = parse_kv_defaults(args.filter_defaults)
    search_defaults = parse_kv_defaults(args.search_defaults)
    color_choices = (parse_list(args.color_choices) if args.color_choices
                 else list(df.columns))  # fallback: any column
    default_color = args.color_col if (args.color_col in df.columns) else ""


    # Details columns
    if args.details.strip() == "*":
        details_cols = list(df.columns)
    else:
        details_cols = [c for c in parse_list(args.details) if c in df.columns]
        if not details_cols:
            details_cols = list(df.columns)

    # Determine defaults for axis
    default_x = args.default_x or (x_choices[0] if x_choices else None)
    default_y = args.default_y or (y_choices[0] if y_choices else None)

    # Build figure & payload on initial defaults
    try:
        fig, payload = build_figure_payload(
            df=df,
            plot_type=args.plot_type,
            x_col=default_x,
            y_col=default_y,
            color_col=args.color_col if args.color_col in df.columns else None,
            details_cols=details_cols,
            top_n=args.initial_zoom if args.initial_zoom and args.initial_zoom > 0 else None,
            sort_primary=args.sort_primary if args.sort_primary in df.columns else None,
            sort_primary_order=args.sort_primary_order,
            sort_secondary=args.sort_secondary if (args.sort_secondary and args.sort_secondary in df.columns) else None,
            sort_secondary_order=args.sort_secondary_order,
            initial_zoom=args.initial_zoom,
            title=args.title,
            show_legend=args.show_legend,
            dup_policy=args.dup_policy,
        )
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    # UI config object for client
    ui_cfg = {
        "x_choices": x_choices,
        "y_choices": y_choices,
        "filter_cols": filter_cols,
        "filter_defaults": filter_defaults,
        "search_cols": search_cols,
        "search_defaults": search_defaults,
        "initial_zoom": args.initial_zoom,
        "sort_choices": list(df.columns),  # permit any column to sort
        "sort_primary": args.sort_primary if args.sort_primary in df.columns else "",
        "sort_primary_order": args.sort_primary_order,
        "sort_secondary": args.sort_secondary if (args.sort_secondary and args.sort_secondary in df.columns) else "",
        "sort_secondary_order": args.sort_secondary_order,
        "color_choices": color_choices,
        "color_default": default_color,   # initial dropdown default

        # NEW: Present tissues subplot config
        "pt": {
            "enable": bool(args.pt_enable),
            "col": args.pt_col,
            "title": args.pt_title,
            "xLabel": args.pt_x_label,
            "yLabel": args.pt_y_label,
            "color": args.pt_color,
            "height": args.pt_height,
            "width": args.pt_width,
            "rotate": args.pt_rotate,
            "containerId": args.pt_container_id,
            "location": args.pt_location,
            "minWidth": args.pt_min_width
        }

    }

    # Persist HTML
    save_html(fig, payload, ui_cfg, out_path=args.out, self_contained=bool(args.self_contained), lang=args.lang)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
```

## CLI help
```txt
universal_plot_maker_plus.py — CLI Reference
1) Input / Output

--file, -f <PATH>
Required. Input table (TSV/CSV). Separator auto‑detected by extension unless --sep is provided. [phrica-my....epoint.com]
--out, -o <PATH>
Output HTML file. Default: interactive_plot.html. [phrica-my....epoint.com]
--sep <STR>
Field separator override (e.g., \t, ,). If omitted, inferred from file extension (.tsv/.tab→tab, .csv→comma). [phrica-my....epoint.com]
--self-contained
Embed Plotly.js in the HTML for fully offline viewing. If omitted, uses CDN. [phrica-my....epoint.com]
--lang <BCP47>
HTML language attribute (e.g., en, en-CA). Default: en. [phrica-my....epoint.com]

2) Plot Basics

--plot-type <bar|scatter|line>
Initial plot type. Default: bar. [phrica-my....epoint.com]
--title <STR>
Initial figure title (shown above the Plotly chart). [phrica-my....epoint.com]

3) Axis Choices & Defaults
These flags define which columns are available in the UI dropdowns and what loads first.

--x-choices "<C1|C2|…>"
Columns allowed for X. For bar, X is categorical; for scatter/line, X must be numeric. Falls back to all columns if omitted. [phrica-my....epoint.com]
--y-choices "<C1|C2|…>"
Columns allowed for Y (numeric). Falls back to all numeric columns if omitted. [phrica-my....epoint.com]
--default-x "<COL>"
Initial X column at load (must be in --x-choices). If omitted, uses the first in x-choices. [phrica-my....epoint.com]
--default-y "<COL>"
Initial Y column at load (must be in --y-choices). If omitted, uses the first numeric candidate. [phrica-my....epoint.com]

4) Coloring (group by)

--color-col "<COL>"
Column used to color points/bars by category (legend optional; see --show-legend). [phrica-my....epoint.com]
--color-choices "<C1|C2|…>"
Columns the UI will allow for “Color by”. If omitted, any column can be chosen. [phrica-my....epoint.com]
--show-legend
Show legend (hidden if not set). [phrica-my....epoint.com]

5) Filtering & Search (client‑side UI)

--filter-cols "<C1|C2|…>"
Adds dropdown filters for these columns (plus an “All” option). [phrica-my....epoint.com]
--filter-defaults "col=value; col=value"
Preselect default values in those filter dropdowns. [phrica-my....epoint.com]
--search-cols "<C1|C2|…>"
Adds text search boxes (case‑insensitive “contains”). [phrica-my....epoint.com]
--search-defaults "col=term; col=term"
Prefill the search boxes with initial terms. [phrica-my....epoint.com]

6) Sorting (client‑side UI)

--sort-primary "<COL>"
Primary sort column to use on first render (must exist in data). [phrica-my....epoint.com]
--sort-primary-order <asc|desc>
Default: desc. [phrica-my....epoint.com]
--sort-secondary "<COL>"
Optional secondary sort (applies when primary ties). [phrica-my....epoint.com]
--sort-secondary-order <asc|desc>
Default: desc. [phrica-my....epoint.com]


Sorting is applied after duplicates are handled (see next section) and respects numeric values where possible. [phrica-my....epoint.com]

7) Duplicates (pre‑plot or visual)

--dup-policy <overlay|stack|max|mean|median|first|sum|separate>
How duplicate X categories are handled:

overlay / stack: keep all rows; bar mode renders single or stacked layers.
max|mean|median|first|sum: collapse duplicates before plotting using that aggregation for Y.
separate: split duplicates into X_#1, X_#2, … so each gets its own bar. [phrica-my....epoint.com]



8) Zoom & Details / Export

--initial-zoom <INT>
Initial number of bars/points shown (viewport range; data are not sliced). Default: 100. [phrica-my....epoint.com]
--details "<C1|C2|…>" or "*"
Columns included in hover, click‑details table, and TSV export. * = all columns. Default: *. [phrica-my....epoint.com]


Interactions: click a bar/point to populate the details table; use lasso/box to select and then Export TSV to save only the selected rows (otherwise exports all filtered rows). [phrica-my....epoint.com]

9) “Present tissues” Subplot (SVG micro‑chart on click)
Enable and control the auxiliary bar chart that renders from a cell string (e.g., "0.41: Tissue A & 0.12: Tissue B").

--pt-enable
Turns the subplot feature on. (If off, only the main Plotly figure & details UI are rendered.) [phrica-my....epoint.com]
--pt-col "<COL>"
Column that contains the encoded sub‑data (e.g., Present tissues). Default: "Present tissues". [phrica-my....epoint.com]
--pt-title "<STR>"
Title for the subplot box. Default: "Present tissues". [phrica-my....epoint.com]
--pt-x-label "<STR>" / --pt-y-label "<STR>"
Axis labels shown under/left of the mini chart. Defaults: Tissue, Score. [phrica-my....epoint.com]
--pt-color "<CSS_COLOR>"
Bar color (e.g., #2a9d8f). Default: #4e79a7. [phrica-my....epoint.com]
--pt-height <INT> / --pt-width <INT|'auto'>
Subplot SVG height; width can be a number or 'auto' (auto uses a constant bar width + gap to decide total width with horizontal scroll when needed). Defaults: 340, auto. [phrica-my....epoint.com]
--pt-rotate <INT>
X‑tick label rotation in degrees. Default: -40. [phrica-my....epoint.com]
--pt-container-id "<ID>"
HTML id for the subplot container div. Default: present-tissues-plot. [phrica-my....epoint.com]
--pt-min-width <INT>
Minimum column/SVG width to protect labels/title from clipping. Default: 420. [phrica-my....epoint.com]

Subplot placement (CLI‑driven)

--pt-mode <flow|absolute|fixed>
Placement strategy.

flow: Inserts a flex row with “details” on the left and subplot on the right (side‑by‑side).
absolute: Places the subplot absolutely within a positioned wrapper near the anchor.
fixed: Pins the subplot to the viewport (independent of page flow).
Default: flow. [phrica-my....epoint.com]


--pt-anchor "<CSS_SELECTOR>"
Anchor element used to place the subplot (e.g., #rowDetails, #controls). Default: #rowDetails. [phrica-my....epoint.com]
--pt-position <before|after|inside>
Placement relative to the anchor (supported in flow and absolute modes). Default: after. [phrica-my....epoint.com]
--pt-offset-x <INT> / --pt-offset-y <INT>
Pixel offsets to nudge the subplot position. Positive X → right, negative X → left. Positive Y → down, negative Y → up. (In flow, applied as a visual translate on the subplot column; in absolute/fixed, applied to the absolutely/fixed‑positioned box.) Defaults: 0, 0
```

## Examples

### Basic bar chart with defaults:
```bash
python universal_plot_maker_plus.py \
  --file data.tsv \
  --out plot.html \
  --plot-type bar \
  --default-x "Gene name" \
  --default-y "log2_enrichment_penalized"
```
### Enable filters, search, and color; set zoom & sorting:

```bash
python universal_plot_maker_plus.py \
  --file data.tsv --out plot.html --plot-type bar \
  --x-choices "Gene name|Gene" \
  --y-choices "Enrichment score|log2_enrichment|log2_enrichment_penalized" \
  --color-col "Cell type" \
  --filter-cols "Cell type class|Cell type group|Cell type" \
  --search-cols "Gene|Gene name" \
  --initial-zoom 150 \
  --sort-primary "overall_rank_by_Cell_type" --sort-primary-order asc \
  --sort-secondary "log2_enrichment_penalized" --sort-secondary-order desc
```
### Turn on the subplot and place it precisely

```
python universal_plot_maker_plus.py \
  --file data.tsv --out plot.html \
  --pt-enable --pt-col "Present tissues" \
  --pt-title "Enrichment per present tissue" \
  --pt-x-label "Tissue" --pt-y-label "log2 Enrichment Penalized" \
  --pt-color "#2a9d8f" --pt-height 360 --pt-width auto --pt-rotate -35 \
  --pt-container-id "present-tissues-plot" --pt-min-width 420 \
  --pt-mode absolute --pt-anchor "#rowDetails" --pt-position after \
  --pt-offset-x 240 --pt-offset-y 0
```

### Notes

If you pick an aggregate duplicate policy (max|mean|median|first|sum), the script collapses duplicates before building the Plotly figure; overlay/stack keep all rows and handle layering in the client.

--self-contained can make large HTMLs (Plotly.js embedded); use the default CDN mode for smaller files.

The TSV Export button saves filtered rows; if you have an active selection (lasso/box), it exports only the selected subset.
