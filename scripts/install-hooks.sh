#!/usr/bin/env bash
# Wire this repo's git hooks. Run once after cloning:  scripts/install-hooks.sh
# (Git hooks live in .git/hooks, which is never committed, so a clone needs this.)
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
hook="$repo_root/.git/hooks/pre-commit"
cat > "$hook" <<'SH'
#!/usr/bin/env bash
exec "$(git rev-parse --show-toplevel)/scripts/guard-secrets.sh"
SH
chmod +x "$hook"
echo "installed pre-commit -> scripts/guard-secrets.sh"
