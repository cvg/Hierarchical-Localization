#!/usr/bin/env bash
# Archive the old open-SuperPoint feature caches (and stale matches/sfm/results that
# depended on them) instead of deleting, then leave the run dirs clean for the
# corrected superpoint_mg_aachen re-run.
set -euo pipefail

REPO=/home/ubuntu/work/MambaGlue/Hierarchical-Localization
cd "$REPO"

STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="outputs/_archive_open_sp_${STAMP}"
mkdir -p "$ARCHIVE"
echo "Archiving stale artifacts to: $ARCHIVE"

for tag in nc2 nc3 nc4 mg; do
  src="outputs/aachen_v1.1_${tag}"
  [[ -d "$src" ]] || { echo "  (skip $tag: $src not found)"; continue; }

  dst="$ARCHIVE/aachen_v1.1_${tag}"
  mkdir -p "$dst"

  # move anything tied to the old open-SuperPoint features:
  #   - the 1024/1600 open feature h5s
  #   - matches, sfm models, results, retrieval pairs built on top of them
  #   - the local-feature pairs are matcher-independent but cheap; archive too for a clean slate
  shopt -s nullglob
  for f in \
    "$src"/feats-superpoint-open-*.h5 \
    "$src"/matches-superpoint-*.h5 \
    "$src"/global-feats-*.h5 \
    "$src"/pairs-query-netvlad*.txt \
    "$src"/pairs-db-covis*.txt \
    "$src"/sfm_superpoint-open+* \
    "$src"/Aachen-v1.1_hloc_superpoint-open+*; do
    echo "  mv $f"
    mv "$f" "$dst"/
  done
  shopt -u nullglob
done

echo "Done. Archived under $ARCHIVE"
echo "Remaining in run dirs (should be empty or only new artifacts):"
for tag in nc2 nc3 nc4 mg; do
  echo "--- outputs/aachen_v1.1_${tag} ---"
  ls -1 "outputs/aachen_v1.1_${tag}" 2>/dev/null || true
done