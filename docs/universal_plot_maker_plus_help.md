# universal_plot_maker_plus

# 1) Introduction

universal_plot_maker_plus.py is a flexible and general‑purpose interactive plotting tool for exploring large tabular datasets (TSV/CSV). It produces a self‑contained HTML file with dynamic controls that allow end‑users to switch axes, filter data, search, sort, zoom, and export selected subsets — all without requiring Python or Plotly installed.
This tool is especially useful for high‑dimensional biological datasets (gene‑level, cell‑type‑level, enrichment tables, marker tables, scoring matrices, etc.) where the user needs:

Dynamic X/Y axis switching
Multiple Y metrics (raw/log/penalized/etc.)
Drop‑down filters (cell type / group / class / cluster…)
Multiple search fields
Sorting by any column
Duplicate handling
Click‑to‑inspect rows
Client‑side TSV export of selected points
Optional embedding of Plotly.js for offline sharing

You control the initial state of the figure entirely through the CLI, and the resulting HTML contains a fully reactive UI that lets end‑users interact with the dataset in real time.

# 2)Run
```

python universal_plot_maker_plus.py \
  --file top_10k.tsv \
  --out Celltype_Enrichment_V2_1_top_10k.html \
  --plot-type bar \
  --x-choices "Gene name | Gene" \
  --y-choices "Enrichment score|log2_enrichment| specificity_tau | Enrichment score (tau penalized)|log2_enrichment_penalized" \
  --default-x "Gene name" \
  --default-y "log2_enrichment_penalized" \
  --color-col "Cell type" \
  --color-choices "Cell type|Cell type group|Cell type class" \
  --filter-cols "Cell type class|Cell type group|Cell type" \
  --search-cols "Gene|Gene name" \
  --details "Gene|Gene name|Cell type|Cell type group|Cell type class|Enrichment score|log2_enrichment| specificity_tau |log2_enrichment_penalized|top_percent_Cell_type_count|top_percent_Cell_type_group_count|top_percent_Cell_type_class_count|overall_rank_by_Cell_type|overall_rank_by_Cell_type_group|overall_rank_by_Cell_type_class|rank_within_Cell_type|rank_within_Cell_type_group|rank_within_Cell_type_class|top_percent_Cell_types|top_percent_Cell_type_groups|top_percent_Cell_type_classes" \
  --title "Celltype Enrichmnt V 2.1" \
  --dup-policy overlay \
  --sort-primary "overall_rank_by_Cell_type" \
  --sort-primary-order asc \
  --sort-secondary "log2_enrichment_penalized" \
  --sort-secondary-order desc \
  --initial-zoom 100 \
  --self-contained \
  --lang en
```
# 3) 🧰 Command‑Line Interface (CLI) — Full Help

Below is the complete list of CLI options supported by `universal_plot_maker_plus.py`, with explanations of what each command does and how it affects the resulting interactive HTML plot.

---

## 📥 Input / Output

### `--file`, `-f`
Path to the input TSV/CSV file.

### `--out`, `-o`
Output HTML file path.  
Default: `interactive_plot.html`

### `--sep`
Manually specify a field separator.  
If omitted, auto‑detected based on file extension (`.tsv`, `.csv`, etc.).

---

## 📊 Plot Configuration

### `--plot-type {bar,scatter,line}`
Initial plot type shown in the viewer.  
End users can still change plot type later.

### `--title`
Title displayed at the top of the plot.

### `--color-col`
Column used for coloring points or bars.  
Each unique category is mapped to a unique color.

---

## 🧭 Axis Selection

### `--x-choices`
List of allowed X‑axis columns.  
Use `|` or `,` to separate multiple options.

### `--y-choices`
List of allowed Y‑axis columns (typically numeric).  
Use `|` or `,` to separate multiple options.

### `--default-x`
The X‑axis column selected at initial load.

### `--default-y`
The Y‑axis column selected at initial load.

---

## 🎚️ Sorting

### `--sort-primary`
Primary sort column used before plotting.

### `--sort-primary-order {asc,desc}`
Sorting direction for primary sort.

### `--sort-secondary`
Optional secondary sort column.

### `--sort-secondary-order {asc,desc}`
Sorting direction for secondary sort.

---

## 🔍 Filtering & Searching

### `--filter-cols`
List of columns exposed as dropdown filters in the HTML viewer.

### `--filter-defaults`
Default filter selections in the format:
``
"Column1=Value1;Column2=Value2"
Use `__ALL__` to default to "no filtering".

### `--search-cols`
Columns that get search bars in the UI.

### `--search-defaults`
Default starting values for search inputs:

"Column1=query;Column2=query"

---

## 📝 Details Panel / Hover Information

### `--details`
Columns included in:
- hover tooltip  
- click‑to‑inspect details panel  
- TSV export of selected points  

Use:
- `"*"` to include all columns  
- `"Col1|Col2|Col3"` for specific columns  

---

## 🔁 Duplicate Handling

### `--dup-policy {overlay,stack,max,mean,median,first,sum}`
Defines how duplicate X‑values (or X+color pairs) are handled:

| Policy | Description |
|--------|-------------|
| `overlay` | Plot duplicates on top of each other (default) |
| `stack` | Stack duplicates as multiple bars |
| `max` | Use only the maximum value |
| `mean` | Use the mean of duplicates |
| `median` | Use the median |
| `first` | Keep first occurrence |
| `sum` | Sum all duplicates |
| `separate` | adds seperate bars for duplicates |

---

## 🔍 Initial Zoom / Data Window

### `--initial-zoom`
Number of rows/bars initially shown.  
More rows remain accessible through sorting or increasing the zoom in the UI.

---

## 🎨 Legend & Layout

### `--show-legend`
If supplied, legend is visible by default.  
Omit to hide the legend.

### `--lang`
Set the HTML `<html lang="...">` attribute.

---

## 📦 Self‑contained HTML

### `--self-contained`
Embed Plotly.js directly in the output HTML.  
Use this when sharing the HTML offline.

---

## 📤 Example Command

```bash
python universal_plot_maker_plus.py \
    --file top_10k.tsv \
    --out top_10k.html \
    --plot-type bar \
    --x-choices "Gene name" \
    --y-choices "Enrichment score|log2_enrichment|log2_enrichment_penalized" \
    --default-x "Gene name" \
    --default-y "log2_enrichment_penalized" \
    --color-col "Cell type" \
    --filter-cols "Cell type class|Cell type group|Cell type" \
    --search-cols "Gene|Gene name" \
    --details "*" \
    --sort-primary "overall_rank_by_Cell_type" \
    --sort-primary-order asc \
    --initial-zoom 100 \
    --self-contained
```
