#!/usr/bin/env bash
# Render a publication-style detail snapshot of one complete crosslink
# site extracted from a cured htpolynet project.
#
# Usage:
#   ./scripts/render-detail.sh <gro> <out.tga> <node-resname> "<partner-resnames>" [face-on-atoms]
#
# Example (BADCy):
#   ./scripts/render-detail.sh \
#       proj-0/systems/final-results/final.gro \
#       badcy-detail.tga \
#       TAZ "BPA CYN" "N1 N2 N3"
#
# Example (PACM/DGEBA — no face-on rotation):
#   ./scripts/render-detail.sh \
#       proj-0/systems/final-results/final.gro \
#       dge-pac-detail.tga \
#       PAC "DGE"
#
# Arguments:
#   <gro>            cured final-results gro; must have a sibling .viz.psf
#   <out.tga>        output tachyon TGA path
#   <node-resname>   the crosslink-node residue (TAZ, PAC, DFA, ...).  The
#                    centre-most occurrence in the box is picked.
#   <partner-resnames>
#                    whitespace-separated list of partner residue names
#                    that the node is bonded to via cure bonds (e.g.
#                    "BPA CYN" or just "DGE").
#   [face-on-atoms]  optional whitespace-separated list of atom names that
#                    define the node's ring plane (e.g. "N1 N2 N3").  When
#                    given, the camera rotates so the ring normal aligns
#                    with the view direction (face-on rendering).
#
# The output is rendered via Tachyon (internal to VMD) at 1200x900;
# convert to PNG afterwards with `magick out.tga -trim +repage out.png`.

set -euo pipefail

GRO="${1:?Usage: render-detail.sh <gro> <out.tga> <node-resname> '<partner-resnames>' [face-on-atoms]}"
OUT="${2:?Usage: render-detail.sh <gro> <out.tga> <node-resname> '<partner-resnames>' [face-on-atoms]}"
NODE="${3:?Usage: render-detail.sh <gro> <out.tga> <node-resname> '<partner-resnames>' [face-on-atoms]}"
PARTNERS="${4:?Usage: render-detail.sh <gro> <out.tga> <node-resname> '<partner-resnames>' [face-on-atoms]}"
FACE_ATOMS="${5:-}"

if [[ ! -f "$GRO" ]]; then
    echo "ERROR: $GRO not found" >&2
    exit 1
fi

GRO="$(realpath "$GRO")"
DIR="$(dirname "$GRO")"
STEM="$(basename "$GRO" .gro)"
PSF="$DIR/$STEM.viz.psf"

# Walk up to proj-N/systems/final-results to find a viz.psf if the
# colocated one is missing.  Detail rendering specifically needs the
# PSF because we follow the bond graph to find partners — autobonds
# from the gro alone won't give us the cure bonds.
if [[ ! -f "$PSF" ]]; then
    ROOT="$DIR"
    while [[ "$ROOT" != "/" && "$ROOT" != "." ]]; do
        if [[ -f "$ROOT/systems/final-results/final.viz.psf" ]]; then
            PSF="$ROOT/systems/final-results/final.viz.psf"
            break
        fi
        ROOT="$(dirname "$ROOT")"
    done
fi

if [[ ! -f "$PSF" ]]; then
    echo "ERROR: no .viz.psf found alongside $GRO — detail rendering needs the bond topology" >&2
    exit 1
fi

# Set up env vars for the TCL driver
export PSF_PATH="$PSF"
export GRO_PATH="$GRO"
export SNAPSHOT_OUT="$OUT"
export NODE_RESNAME="$NODE"
export PARTNER_RESNAMES="$PARTNERS"
if [[ -n "$FACE_ATOMS" ]]; then
    export FACE_ON=1
    export FACE_ATOMS="$FACE_ATOMS"
else
    export FACE_ON=0
fi

# Pick up the local tachyon binary; fall back to "tachyon" on PATH.
TACHYON_BIN="${TACHYON:-$(command -v tachyon 2>/dev/null || echo tachyon)}"
export TACHYON="$TACHYON_BIN"

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
TCL="$SCRIPT_DIR/render-detail.tcl"

echo "rendering detail of $NODE residue + bonded $PARTNERS partners (via $PSF) → $OUT"
vmd -dispdev text -e "$TCL" </dev/null > /tmp/render-detail.log 2>&1

if [[ ! -f "$OUT" ]]; then
    echo "ERROR: detail render failed; see /tmp/render-detail.log" >&2
    tail -20 /tmp/render-detail.log >&2
    exit 1
fi

echo "wrote $OUT"
