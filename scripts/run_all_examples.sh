#!/usr/bin/env bash
# Run every example in the depot, each in its own working directory.
#
# Usage:
#   ./scripts/run_all_examples.sh [-d <root>] [--force-reparameterize] [-- <htpolynet run flags>]
#
# Options:
#   -d <root>               Parent directory under which one subdirectory
#                           per example is created.  Defaults to
#                           ./examples-runs/.
#   --force-reparameterize  Forward --force-parameterization --force-checkin
#                           to every `htpolynet run` call.  Each example
#                           re-runs antechamber/parmchk/tleap on its
#                           monomers and overwrites the user cache
#                           (~/.htpolynet/molecules/parameterized/).
#                           Right level of rigor when the cache may be
#                           stale across YAMLs with different reaction
#                           sets — single source of truth per fresh run.
#   --                      Everything after this is forwarded verbatim as
#                           extra flags to `htpolynet run` (in addition to
#                           `-diag diagnostics.log`).
#
# Example:
#   ./scripts/run_all_examples.sh -d /tmp/htp-runs --force-reparameterize
#   ./scripts/run_all_examples.sh -- --loglevel debug
#
# Behavior:
#   - For each example N, runs `htpolynet fetch-example N` inside its own
#     subdirectory, then `htpolynet run -diag diagnostics.log <yaml>`,
#     redirecting stdout/stderr into console.log.
#   - Examples run sequentially (they share GPU/CPU; parallelizing would
#     fight for the same resources).
#   - Per-example exit status is reported at the end; a non-zero overall
#     exit code is returned if any example failed.

set -uo pipefail

ROOT='./examples-runs'
EXTRA_RUN_ARGS=()
FORCE_REPARAM=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d)
            ROOT="$2"
            shift 2
            ;;
        --force-reparameterize)
            FORCE_REPARAM=1
            shift
            ;;
        --)
            shift
            EXTRA_RUN_ARGS=("$@")
            break
            ;;
        -h|--help)
            sed -n '2,32p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [[ $FORCE_REPARAM -eq 1 ]]; then
    EXTRA_RUN_ARGS=(--force-parameterization --force-checkin "${EXTRA_RUN_ARGS[@]}")
fi

if ! command -v htpolynet >/dev/null 2>&1; then
    echo "error: htpolynet is not on PATH" >&2
    exit 1
fi

mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"

# Discover example IDs by parsing `htpolynet fetch-example --help`.
# Fall back to the canonical 0..4 set if discovery fails.
IDS=()
if help_text="$(htpolynet fetch-example --help 2>&1)"; then
    while read -r id; do
        IDS+=("$id")
    done < <(printf '%s\n' "$help_text" \
                 | grep -oE "'[0-9]+'" \
                 | tr -d "'" \
                 | sort -u)
fi
if [[ ${#IDS[@]} -eq 0 ]]; then
    IDS=(0 1 2 3 4 5 6)
fi

echo "Running examples ${IDS[*]} under $ROOT"
echo

declare -A STATUS
declare -A DURATION

for id in "${IDS[@]}"; do
    run_dir="$ROOT/example-$id"
    mkdir -p "$run_dir"
    echo "=== example $id  ($run_dir) ==="
    (
        cd "$run_dir"
        # If the YAML is already here from a previous run, skip the fetch.
        if ! ls *.yaml >/dev/null 2>&1; then
            htpolynet fetch-example "$id"
        fi
        yaml="$(ls -1 *.yaml | head -n1)"
        if [[ -z "$yaml" ]]; then
            echo "error: no YAML found after fetch-example $id" >&2
            exit 1
        fi
        echo "  yaml:    $yaml"
        echo "  logfile: $run_dir/console.log"
        SECONDS=0
        htpolynet run -diag diagnostics.log "${EXTRA_RUN_ARGS[@]}" "$yaml" \
            > console.log 2>&1
        rc=$?
        echo "  elapsed: ${SECONDS}s"
        exit $rc
    )
    rc=$?
    STATUS["$id"]=$rc
    if [[ $rc -eq 0 ]]; then
        echo "  -> ok"
    else
        echo "  -> FAILED (rc=$rc)  see $run_dir/console.log"
    fi
    echo
done

echo "=== summary ==="
overall=0
for id in "${IDS[@]}"; do
    rc="${STATUS[$id]:-?}"
    if [[ "$rc" == "0" ]]; then
        printf '  example %s : ok\n' "$id"
    else
        printf '  example %s : FAIL (rc=%s)\n' "$id" "$rc"
        overall=1
    fi
done

exit "$overall"
