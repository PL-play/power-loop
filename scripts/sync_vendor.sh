#!/usr/bin/env bash
# Re-vendor llm_client into power_loop/_vendor/llm_client (ECO-2).
#
# Copies an upstream `llm_client` package into the vendored location, strips modules
# power-loop does not use (so the core stays dependency-free), and rewrites imports to
# the vendored path. Idempotent. Review the diff + update _vendor/llm_client/VENDOR.md
# with the source commit before committing.
#
# Usage:
#   LLM_CLIENT_SRC=/path/to/llm_client scripts/sync_vendor.sh
#
# After: ruff check . && mypy power_loop && pytest -q
set -euo pipefail

SRC="${LLM_CLIENT_SRC:-}"
if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo "error: set LLM_CLIENT_SRC to the upstream llm_client package directory" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/power_loop/_vendor/llm_client"

# Modules power-loop does not import — keep the vendored copy minimal + zero-dependency.
PRUNE=(qwen_image.py web_search.py)

echo "vendoring $SRC -> $DEST"
mkdir -p "$DEST"
rm -f "$DEST"/*.py
cp "$SRC"/*.py "$DEST"/
[[ -f "$DEST/__init__.py" ]] || : > "$DEST/__init__.py"

for f in "${PRUNE[@]}"; do
  rm -f "$DEST/$f"
done

# Rewrite any top-level `llm_client` imports to the vendored path.
# (BSD/GNU sed compatible: write to a temp then move.)
while IFS= read -r -d '' py; do
  sed -e 's/\bfrom llm_client/from power_loop._vendor.llm_client/g' \
      -e 's/\bimport llm_client\b/import power_loop._vendor.llm_client as llm_client/g' \
      "$py" > "$py.tmp" && mv "$py.tmp" "$py"
done < <(find "$DEST" -name '*.py' -print0)

echo "done. Now: update VENDOR.md (source commit), then run ruff/mypy/pytest."
