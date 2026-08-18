#!/usr/bin/env bash
# Tier-3 mechanism for the "preserve full run logs" rule (see CLAUDE.md, the
# preserve-full-run-logs memory, and working_with_ai.md's tiers).
#
# The ground-truth loss it prevents: piping a run through `... | tee log | tail`
# truncates the stream (and BSD tee buffers), so the committed log ends up a few
# lines instead of the whole run - the receipts (request_id/batch_id, verbatim
# usage) are exactly what gets cut. This wrapper makes that impossible BY
# CONSTRUCTION: the FULL stdout+stderr always lands in evidence/raw-logs/<name>.log;
# the tail you see is read back FROM that complete file, so legibility never costs
# preservation.
#
# Usage:  scripts/run-evidence.sh <log-name> -- <command...>
# Example: scripts/run-evidence.sh batch-sweep-haiku -- uv run python -m llm_benchmark.sweep --html reports/x.html
set -uo pipefail

name="${1:?usage: run-evidence.sh <log-name> -- <command...>}"; shift
[ "${1:-}" = "--" ] && shift
[ "$#" -gt 0 ] || { echo "run-evidence.sh: no command given" >&2; exit 2; }

mkdir -p evidence/raw-logs
log="evidence/raw-logs/${name}.log"

# FULL stream to the durable file. No pipe, no tail, nothing that can truncate.
"$@" > "$log" 2>&1
rc=$?

lines=$(wc -l < "$log" | tr -d ' ')
echo "[run-evidence] full log preserved -> $log (${lines} lines, exit ${rc})"
echo "[run-evidence] tail of the COMPLETE file (nothing was truncated on disk):"
tail -n 12 "$log"
exit "$rc"
