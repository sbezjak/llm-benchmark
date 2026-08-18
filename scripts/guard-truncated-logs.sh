#!/usr/bin/env bash
# Tier-3 guard for the "preserve full run logs" rule (PreToolUse/Bash hook).
#
# Reads the hook's stdin JSON, and if the Bash command about to run pipes a REAL
# run (uv run / python -m / pytest) or a `tee` through tail/head - the exact
# pattern that truncated and lost the batch run's receipts - it emits a
# non-blocking warning + additionalContext nudging toward scripts/run-evidence.sh.
# It never blocks: plain `tail`-of-a-file and clean `> file 2>&1` redirects pass
# silently. See the preserve-full-run-logs memory and CLAUDE.md.
set -uo pipefail

cmd=$(jq -r '.tool_input.command // ""' 2>/dev/null || echo "")
[ -n "$cmd" ] || exit 0

# Anti-pattern: a run (or a tee) whose output is piped into tail/head, which
# discards everything but the last lines - the ground-truth receipts included.
pattern='((uv run|python -m|pytest)[^|]*\|[^|]*(tail|head)\b)|(tee[^|]*\|[^|]*(tail|head)\b)'
printf '%s' "$cmd" | grep -Eq "$pattern" || exit 0

msg='WARN: piping a run through tail/head loses the full log (the receipts). Preserve it with scripts/run-evidence.sh <name> -- <cmd> (full stream -> evidence/raw-logs/).'
ctx='This Bash command pipes a run through tail/head, discarding the full log - the exact mistake that lost the batch run receipts. If this run backs an evidence claim, re-run it via scripts/run-evidence.sh <log-name> -- <command> so the complete stdout+stderr lands in evidence/raw-logs/<log-name>.log.'
jq -cn --arg m "$msg" --arg c "$ctx" \
  '{systemMessage:$m, hookSpecificOutput:{hookEventName:"PreToolUse", additionalContext:$c}}'
