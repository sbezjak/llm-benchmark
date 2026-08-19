#!/usr/bin/env bash
# Pre-commit guard: block a commit that stages a secret.
#
# Encodes the repo's manual pre-push security pass: this project spends real money
# against paid providers, so the one realistic leak
# is a provider key, a Langfuse key, or a captured Authorization/x-api-key header
# ending up in a committed file, report, or evidence log. It scans only the
# ADDED lines of the staged diff (not the whole tree - pre-existing benign matches
# never re-fire) and refuses the commit if any look like a real credential. It
# also refuses to commit a real `.env`.
#
# Install as the git hook once (also wires a fresh clone):  scripts/install-hooks.sh
# Escape hatch for a false positive:  SKIP_SECRET_GUARD=1 git commit ...  (or git commit --no-verify)
#
# Deliberately NOT hardcoded: no literal key lives here - the provider adapters'
# `{"x-api-key": self._api_key}` / `f"Bearer {self._api_key}"` lines read from the
# env, and their `{...}` template value never matches the length gate below.
set -uo pipefail

[ "${SKIP_SECRET_GUARD:-0}" = "1" ] && exit 0

# Real-credential shapes: known provider/tool prefixes, private keys, and an
# Authorization/x-api-key header carrying an actual >=20-char value (a `{var}`
# template or the empty `KEY=` placeholder in .env.example can't reach the gate).
patterns='sk-[A-Za-z0-9_-]{20,}'   # OpenAI (incl. sk-proj-...), Anthropic sk-ant-..., classic sk-...
patterns+='|AKIA[0-9A-Z]{16}'
patterns+='|AIza[0-9A-Za-z_-]{30,}'
patterns+='|gh[pousr]_[A-Za-z0-9]{30,}'
patterns+='|xox[baprs]-[0-9A-Za-z-]{12,}'
patterns+='|(pk|sk)-lf-[A-Za-z0-9-]{16,}'
patterns+='|-----BEGIN [A-Z ]*PRIVATE KEY-----'
patterns+='|(authorization|x-api-key)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"']?(bearer[[:space:]]+)?[A-Za-z0-9._-]{20,}'

staged=$(git diff --cached --name-only --diff-filter=ACM)
[ -n "$staged" ] || exit 0

fail=0

# 1. A real .env (any .env / .env.* except .env.example) must never be staged.
env_staged=$(printf '%s\n' "$staged" | grep -E '(^|/)\.env(\.|$)' | grep -vE '(^|/)\.env\.example$' || true)
if [ -n "$env_staged" ]; then
  echo "BLOCKED: refusing to commit an environment file (holds real keys):" >&2
  printf '  %s\n' $env_staged >&2
  fail=1
fi

# 2. Secret-shaped strings in the ADDED lines of the staged diff.
#    -I skips binary; -U0 keeps only changed hunks; keep '^+' adds, drop the '+++' header.
hits=$(git diff --cached -U0 --diff-filter=ACM -- $staged \
        | grep -aE '^\+' | grep -avE '^\+\+\+ ' \
        | grep -aniE "$patterns" || true)
if [ -n "$hits" ]; then
  echo "BLOCKED: a staged line looks like a live secret:" >&2
  printf '%s\n' "$hits" | sed 's/^/  /' >&2
  fail=1
fi

if [ "$fail" = 1 ]; then
  echo "" >&2
  echo "Nothing was committed. Move the value into .env (git-ignored) and read it from the env," >&2
  echo "or if this is a false positive: SKIP_SECRET_GUARD=1 git commit ...  (or git commit --no-verify)." >&2
  exit 1
fi
exit 0
