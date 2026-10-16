#!/usr/bin/env bash
# Attaches a built zip to a GitHub Release. Called by both platform workflows
# after their tests pass, so the Windows and macOS zips land on the same release.
#
#   v* tag push    -> that tag's release (hand-cut versions, e.g. v0.3.0)
#   push to master -> a build-YYYYMMDD-<sha7> release marked Latest, so the
#                     Releases page always offers the newest master build
#
# The build tag comes from the commit itself (commit date + short sha), not the
# run number, so the two workflows agree on it without talking to each other.
# Whichever finishes first creates the release; the other just uploads.
# Releases made with GITHUB_TOKEN never trigger workflows, so the tag this
# creates does not start another build.
#
# Only the newest KEEP build-* releases are kept; v* releases are never touched.
#
# Needs GH_TOKEN, GITHUB_REF, GITHUB_REF_NAME and GITHUB_SHA (all set by Actions).
set -euo pipefail

zip="$1"
KEEP=10

if [[ "$GITHUB_REF" == refs/tags/v* ]]; then
  tag="$GITHUB_REF_NAME"
  title="raw-editor $tag"
  create_args=(--generate-notes)
else
  date=$(git show -s --format=%cd --date=format:%Y%m%d "$GITHUB_SHA")
  tag="build-$date-${GITHUB_SHA:0:7}"
  subject=$(git show -s --format=%s "$GITHUB_SHA")
  title="raw-editor $tag"
  create_args=(--target "$GITHUB_SHA" --latest
               --notes "Automatic build of master at ${GITHUB_SHA:0:7}: $subject")
fi

gh release view "$tag" >/dev/null 2>&1 \
  || gh release create "$tag" --title "$title" "${create_args[@]}" \
  || true

uploaded=0
for i in 1 2 3 4 5; do
  if gh release upload "$tag" "$zip" --clobber; then uploaded=1; break; fi
  sleep 5
done
[[ $uploaded == 1 ]] || { echo "::error::upload to $tag failed"; exit 1; }
echo "published $zip to $tag"

# Prune old automatic builds. Both workflows run this; deleting an already
# deleted release just fails quietly.
if [[ "$tag" == build-* ]]; then
  gh release list --limit 200 --json tagName,createdAt \
      --jq '[.[] | select(.tagName | startswith("build-"))] | sort_by(.createdAt) | reverse | .['"$KEEP"':][].tagName' |
    while read -r old; do
      gh release delete "$old" --cleanup-tag --yes || true
    done
fi
