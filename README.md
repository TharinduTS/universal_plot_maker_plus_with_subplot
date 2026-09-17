# universal_plot_maker_plus_with_subplot

###Please read the help file in docs for more info

This updated script introduces the ability to use sublots, so you can plot dataframes with more than 3 layers.

## Notes

1. If you pick an aggregate duplicate policy (max|mean|median|first|sum) as default --dup-policy, the script collapses duplicates before building the Plotly figure; overlay/stack keep all rows and handle layering in the client.
Therefore *it is always advisable to use overlay/stack as the starting dup-policy*

2. --self-contained can make large HTMLs (Plotly.js embedded); use the default CDN mode for smaller files.

3. The TSV Export button saves filtered rows; if you have an active selection (lasso/box), it exports only the selected subset.





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

