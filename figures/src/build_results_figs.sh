#!/usr/bin/env bash
# Build the forest, load and workload-level figures (fig6_workloads.pdf: supplement Fig. S1) from the analysis
# directory written by scripts/analyze_campaign.py (contrasts.csv, descriptive.csv, trials.csv, summary.json).
#
#   Final:        bash src/build_results_figs.sh --final ANALYSIS_DIR [ROBUST_DIR ...]
#                 writes ../fig4_forest.pdf ../fig5_load.pdf ../fig6_workloads.pdf, no watermark;
#                 refuses rehearsal (phase tune, validation block) and synthetic data.
#   Rehearsal:    bash src/build_results_figs.sh ANALYSIS_DIR [ROBUST_DIR ...]
#                 writes ../rehearsal/fig*_<KIND>.pdf with a watermark and PNG previews (color and
#                 gray, 300 dpi) in ../rehearsal/png/.
#
# ROBUST_DIR: analysis directory of the robustness family (MMPP, mobility, on-off) when it was
# analysed on its own; it feeds fig6_workloads.py only. Never pass rehearsal or synthetic data with real data.
# Then checks every PDF: page size, fonts embedded, no Type 3 font, info dictionary = title only.
# Needs python3 with numpy, scipy, matplotlib 3.11 (STIX fonts ship with matplotlib) and
# poppler-utils (pdfinfo, pdffonts, pdftoppm). Writes nothing else; no other figure is touched.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
SRC="$(cd "$(dirname "$0")" && pwd)"; OUT="$(cd "$SRC/.." && pwd)"
FINAL=()
if [[ "${1:-}" == "--final" ]]; then FINAL=(--final); shift; fi
if [[ $# -lt 1 ]]; then sed -n '2,13p' "$0"; exit 2; fi
DIR="$(cd "$1" && pwd)"; shift
ROB=()
for r in "$@"; do ROB+=(--robust "$(cd "$r" && pwd)"); done
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

cd "$SRC"
python3 fig4_forest.py "$DIR" ${FINAL[@]+"${FINAL[@]}"} | tee "$TMP/log4"
python3 fig5_load.py "$DIR" ${FINAL[@]+"${FINAL[@]}"} | tee "$TMP/log5"
python3 fig6_workloads.py "$DIR" ${ROB[@]+"${ROB[@]}"} ${FINAL[@]+"${FINAL[@]}"} | tee "$TMP/log6"
mapfile -t PDFS < <(awk '/^wrote /{print $2}' "$TMP/log4" "$TMP/log5" "$TMP/log6")

echo
echo "check (page size; fonts; info dictionary):"
bad=0
for f in "${PDFS[@]}"; do
  size="$(pdfinfo "$f" | awk '/^Page size/{printf "%.3f x %.2f in", $3/72, $5/72}')"
  nfont="$(pdffonts "$f" | awk 'NR>2' | wc -l)"
  t3="$(pdffonts "$f" | awk 'NR>2 && /Type 3/' | wc -l)"
  notemb="$(pdffonts "$f" | awk 'NR>2 && $(NF-4)!="yes"' | wc -l)"
  info="$(pdfinfo "$f" | awk -F: '/^(Title|Author|Creator|Producer|CreationDate|ModDate|Subject|Keywords):/{print $1}' | paste -sd, -)"
  printf "  %-36s %s; %s fonts, %s not embedded, %s Type 3; info: %s\n" \
    "$(basename "$f")" "$size" "$nfont" "$notemb" "$t3" "$info"
  if [[ "$t3" != 0 || "$notemb" != 0 || "$info" != "Title" ]]; then bad=1; fi
done
if [[ $bad != 0 ]]; then echo "CHECK FAILED"; exit 1; fi

if [[ ${#FINAL[@]} -eq 0 ]]; then
  mkdir -p "$OUT/rehearsal/png"
  for f in "${PDFS[@]}"; do
    b="$OUT/rehearsal/png/$(basename "${f%.pdf}")"
    pdftoppm -r 300 -png -singlefile "$f" "$b"
    pdftoppm -r 300 -gray -png -singlefile "$f" "${b}_gray"
  done
  echo "previews: $OUT/rehearsal/png/"
fi
